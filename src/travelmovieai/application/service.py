"""Use-case facade called by the CLI."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal, cast

from pydantic import ValidationError

from travelmovieai.analysis.vision import VisionProvider
from travelmovieai.application.context import ProjectContext
from travelmovieai.application.diagnostics import SystemDiagnosticReport, run_system_diagnostics
from travelmovieai.application.disk_space import ensure_render_disk_space
from travelmovieai.application.project_archive import (
    export_project_archive,
    restore_project_archive,
)
from travelmovieai.application.reporting import generate_project_report
from travelmovieai.application.resource_estimates import (
    ProjectResourceEstimate,
    estimate_project_resources,
)
from travelmovieai.application.semantic_search import search_project_scenes
from travelmovieai.application.validation import (
    ProjectPaths,
    validate_output_path,
    validate_project_paths,
)
from travelmovieai.application.variants import safe_variant_slug, validate_variant_name
from travelmovieai.application.workspace_identity import (
    default_workspace_path,
    legacy_workspace_path,
    validate_existing_workspace_identity,
    workspace_proves_source,
)
from travelmovieai.application.workspace_lease import WorkspaceLease
from travelmovieai.core.config import Settings
from travelmovieai.core.exceptions import MontageError
from travelmovieai.domain.enums import PipelineStage, StageStatus, StoryStyle
from travelmovieai.domain.models import (
    MediaScanReport,
    MontageQualityReport,
    MusicPlan,
    QuickMontagePlan,
    QuickMontageResult,
    QuickMontageSettings,
    Scene,
    SemanticSearchReport,
    StageResult,
)
from travelmovieai.editing.publication import (
    publish_render_candidate,
    render_candidate_path,
)
from travelmovieai.editing.quality_report import (
    build_montage_quality_report,
    enforce_montage_quality,
    enrich_montage_quality_report_with_render,
)
from travelmovieai.editing.renderer import QuickMontageRenderer
from travelmovieai.editing.timeline import (
    apply_music_directing,
    build_quick_montage_plan,
)
from travelmovieai.infrastructure.artifacts import write_json_atomic
from travelmovieai.infrastructure.database import MediaAssetRepository
from travelmovieai.infrastructure.music_generation import (
    AceStepMusicGenerator,
    resolve_local_music_model,
)
from travelmovieai.infrastructure.system import ResourceProfile, detect_resource_profile
from travelmovieai.pipeline.progress import (
    LegacyProgressCallback,
    ProgressEventCallback,
)
from travelmovieai.pipeline.registry import build_default_pipeline
from travelmovieai.pipeline.runner import PipelineRunner
from travelmovieai.pipeline.stages.music_selection import MusicGeneratorFactory
from travelmovieai.pipeline.stages.vision_analysis import VisionProviderFactory
from travelmovieai.story.music import NeuralMusicGenerator, build_music_plan


class TravelMovieService:
    def __init__(
        self,
        settings: Settings,
        vision_provider_factory: Callable[[Settings], VisionProvider] | None = None,
        music_generator_factory: Callable[[Settings, str], NeuralMusicGenerator] | None = None,
    ) -> None:
        self.settings = settings
        self._vision_provider_factory = vision_provider_factory
        self._music_generator_factory = music_generator_factory
        self._resource_profile: ResourceProfile | None = None

    def get_resource_profile(self, *, refresh: bool = False) -> ResourceProfile:
        if self._resource_profile is None or refresh:
            self._resource_profile = detect_resource_profile(
                self.settings.ffmpeg_binary,
                worker_override=self.settings.workers,
                batch_override=self.settings.batch_size,
                resource_mode=self.settings.resource_mode,
                gpu_memory_reserve_mb=self.settings.gpu_memory_reserve_mb,
                max_gpu_processes=self.settings.max_gpu_processes,
            )
        return self._resource_profile

    def diagnostics(self) -> SystemDiagnosticReport:
        return run_system_diagnostics(self.settings)

    def estimate(
        self,
        *,
        input_path: Path,
        workspace: Path | None,
        montage_settings: QuickMontageSettings | None = None,
    ) -> ProjectResourceEstimate:
        """Scan metadata and return a bounded runtime/disk preflight estimate."""

        self.analyze(input_path=input_path, workspace=workspace)
        context = self._context(input_path=input_path, workspace=workspace)
        analysis_path = context.artifacts_dir / "analysis.json"
        try:
            report = MediaScanReport.model_validate_json(analysis_path.read_text(encoding="utf-8"))
        except (OSError, ValidationError) as error:
            raise MontageError("Could not read media metadata for the project estimate.") from error
        repository = MediaAssetRepository(context.database_path)
        repository.initialize()
        known_scene_count = len(repository.list_scenes()) or None
        repository.close()
        return estimate_project_resources(
            report.assets,
            settings=self.settings,
            montage_settings=montage_settings or QuickMontageSettings(),
            known_scene_count=known_scene_count,
        )

    def create(
        self,
        *,
        input_path: Path,
        output_path: Path,
        workspace: Path | None,
        style: StoryStyle,
        semantic: bool = False,
        montage_settings: QuickMontageSettings | None = None,
        variant_name: str = "Default",
        progress: LegacyProgressCallback | None = None,
        progress_events: ProgressEventCallback | None = None,
    ) -> StageResult:
        if montage_settings is None:
            montage_settings = QuickMontageSettings(
                semantic_analysis=semantic,
                story_style=style,
                vision_provider=self.settings.vision_provider,
                vision_model=(
                    None if self.settings.vision_model == "auto" else self.settings.vision_model
                ),
            )
        else:
            montage_settings = montage_settings.model_copy(
                update={"semantic_analysis": semantic, "story_style": style}
            )
        _validate_montage_feature_requests(montage_settings, self.settings)
        if semantic:
            return self.run_until(
                PipelineStage.RENDERING,
                input_path=input_path,
                output_path=output_path,
                workspace=workspace,
                style=style,
                montage_settings=montage_settings,
                variant_name=variant_name,
                progress=progress,
                progress_events=progress_events,
            )

        result = self.create_quick_montage(
            input_path=input_path,
            workspace=workspace,
            settings=montage_settings,
            output_path=output_path,
            variant_name=variant_name,
            progress=progress,
            progress_events=progress_events,
        )
        return StageResult(
            stage=PipelineStage.RENDERING,
            artifacts=[result.timeline_path, result.output_path],
            message=(
                f"{result.selection_mode.title()} montage created "
                f"{result.clip_count} clip(s), "
                f"{result.duration_seconds:.1f}s: {result.output_path}"
            ),
        )

    def analyze(
        self,
        *,
        input_path: Path,
        workspace: Path | None,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> StageResult:
        return self.run_until(
            PipelineStage.MEDIA_SCAN,
            input_path=input_path,
            workspace=workspace,
            progress=progress,
        )

    def create_quick_montage(
        self,
        *,
        input_path: Path,
        workspace: Path | None,
        settings: QuickMontageSettings,
        output_path: Path | None = None,
        progress: Callable[[int, int, str], None] | None = None,
        progress_events: ProgressEventCallback | None = None,
        variant_name: str = "Default",
    ) -> QuickMontageResult:
        settings = _effective_montage_settings(settings)
        _validate_montage_feature_requests(settings, self.settings)
        project_paths = self.resolve_project_paths(input_path, workspace)
        with WorkspaceLease(project_paths.workspace, operation="quick_montage"):
            return self._create_quick_montage_locked(
                input_path=project_paths.input_path,
                workspace=project_paths.workspace,
                settings=settings,
                output_path=output_path,
                progress=progress,
                progress_events=progress_events,
                variant_name=variant_name,
            )

    def _create_quick_montage_locked(
        self,
        *,
        input_path: Path,
        workspace: Path,
        settings: QuickMontageSettings,
        output_path: Path | None,
        progress: Callable[[int, int, str], None] | None,
        progress_events: ProgressEventCallback | None,
        variant_name: str,
    ) -> QuickMontageResult:
        settings = _effective_montage_settings(settings)
        _validate_montage_feature_requests(settings, self.settings)
        try:
            normalized_variant_name = validate_variant_name(variant_name)
        except ValueError as error:
            raise MontageError(str(error)) from error
        variant_slug = safe_variant_slug(normalized_variant_name)
        resources = self.get_resource_profile(refresh=True)
        tracker = _ProgressTracker(progress)
        tracker.emit(0, f"Resource profile: {resources.summary}")
        context = self._context(
            input_path=input_path,
            workspace=workspace,
            variant_name=normalized_variant_name,
            variant_slug=variant_slug,
            resources=resources,
        )
        default_name = "preview.mp4" if settings.preview_mode else "final.mp4"
        resolved_output = validate_output_path(
            output_path or context.artifacts_dir / default_name,
            context.input_path,
            workspace=context.workspace,
            database_path=context.database_path,
        )
        if settings.semantic_analysis:
            semantic_context = self._context(
                input_path=context.input_path,
                workspace=context.workspace,
                output_path=resolved_output,
                style=settings.story_style,
                montage_settings=settings,
                variant_name=normalized_variant_name,
                variant_slug=variant_slug,
                resources=resources,
            )
            return self._create_semantic_montage(
                semantic_context,
                resolved_output,
                tracker,
                progress_events,
            )

        tracker.emit(1, "Checking media library and updating index")
        self.analyze(
            input_path=context.input_path,
            workspace=context.workspace,
            progress=tracker.range(1, 5),
        )
        tracker.emit(5, "Media scan complete, reading metadata")
        analysis_path = context.artifacts_dir / "analysis.json"
        try:
            report = MediaScanReport.model_validate_json(analysis_path.read_text(encoding="utf-8"))
        except (OSError, ValidationError) as error:
            raise MontageError("Could not read media scan results.") from error

        quality_report_path = context.artifacts_dir / "montage_quality_report.json"
        quality_report: MontageQualityReport | None = None
        tracker.emit(10, "Selecting quick clips by duration")
        plan = build_quick_montage_plan(report.assets, settings)
        tracker.emit(78, "Building a music map from clip boundaries")
        music_plan = self._build_music_plan(
            context,
            report,
            [],
            settings,
            plan,
            tracker.range(78, 80),
        )
        plan = plan.model_copy(
            update={
                "music_plan": music_plan,
                "music_path": music_plan.source_path,
            }
        )
        plan = apply_music_directing(plan)
        quality_report = build_montage_quality_report(plan, [])
        tracker.emit(80, "Quick edit plan created")
        timeline_path = context.artifacts_dir / "quick_timeline.json"
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        tracker.emit(84, f"Timeline ready: {len(plan.clips)} clip(s)")
        ensure_render_disk_space(
            workspace=context.workspace,
            output_path=resolved_output,
            settings=settings,
            plan=plan,
            reserve_mb=self.settings.render_disk_reserve_mb,
            safety_factor=self.settings.render_disk_safety_factor,
        )
        candidate_path = render_candidate_path(resolved_output)
        try:
            render_encoder = QuickMontageRenderer(
                self.settings.ffmpeg_binary,
                self.settings.ffprobe_binary,
                workers=resources.render_workers,
                ffmpeg_threads=resources.ffmpeg_threads,
                timeout_seconds=self.settings.render_timeout_seconds,
            ).render(
                plan,
                candidate_path,
                context.cache_dir,
                tracker.range(85, 100),
            )
            quality_report = enrich_montage_quality_report_with_render(
                quality_report,
                candidate_path,
                ffprobe_binary=self.settings.ffprobe_binary,
                ffmpeg_binary=self.settings.ffmpeg_binary,
                timeout_seconds=self.settings.render_timeout_seconds,
                require_full_scan=plan.settings.validate_full_render_decode,
            )
            quality_report = quality_report.model_copy(update={"render_encoder": render_encoder})
            enforce_montage_quality(quality_report)
            quality_report = quality_report.model_copy(update={"rendered_path": resolved_output})
            publish_render_candidate(candidate_path, resolved_output)
            write_json_atomic(timeline_path, plan)
            write_json_atomic(quality_report_path, quality_report)
            _record_timeline_version(
                context,
                plan,
                phase="built",
            )
            _record_timeline_version(
                context,
                plan,
                phase="rendered",
                output_path=resolved_output,
            )
        finally:
            candidate_path.unlink(missing_ok=True)
        tracker.emit(100, "Film ready and validated with FFprobe")
        return QuickMontageResult(
            output_path=resolved_output,
            timeline_path=timeline_path,
            clip_count=len(plan.clips),
            duration_seconds=plan.total_duration_seconds,
            selection_mode=plan.selection_mode,
            render_encoder=render_encoder,
            music_mode=plan.music_plan.mode if plan.music_plan else None,
            music_profile=plan.music_plan.profile if plan.music_plan else None,
            music_generator=plan.music_plan.generator if plan.music_plan else None,
            music_model=plan.music_plan.model if plan.music_plan else None,
            quality_score=quality_report.score,
            quality_issue_count=len(quality_report.issues),
            quality_gate_status=quality_report.gate_status,
            semantic_score_p10=quality_report.semantic_score_p10,
            dominant_event_ratio=quality_report.dominant_event_ratio,
            adjacent_source_repeat_ratio=quality_report.adjacent_source_repeat_ratio,
            center_cut_ratio=quality_report.center_cut_ratio,
            full_media_qa_completed=(
                quality_report.rendered_media_metrics is not None
                and quality_report.rendered_media_metrics.scan_completed
            ),
        )

    def _create_semantic_montage(
        self,
        context: ProjectContext,
        output_path: Path,
        tracker: _ProgressTracker,
        progress_events: ProgressEventCallback | None,
    ) -> QuickMontageResult:
        result = self._pipeline_runner().run_until(
            context,
            PipelineStage.RENDERING,
            progress=tracker.range(1, 100),
            progress_events=progress_events,
        )
        if result.status is StageStatus.NO_INPUT:
            raise MontageError(result.message or "Semantic montage has no usable media.")

        timeline_path = context.artifacts_dir / "quick_timeline.json"
        quality_report_path = context.artifacts_dir / "montage_quality_report.json"
        try:
            plan = QuickMontagePlan.model_validate_json(timeline_path.read_text(encoding="utf-8"))
            quality_report = MontageQualityReport.model_validate_json(
                quality_report_path.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError) as error:
            raise MontageError("Semantic pipeline produced invalid montage artifacts.") from error
        if not output_path.is_file():
            raise MontageError("Semantic pipeline did not produce the requested movie.")

        tracker.emit(100, "Film ready and validated with FFprobe")
        return QuickMontageResult(
            output_path=output_path,
            timeline_path=timeline_path,
            clip_count=len(plan.clips),
            duration_seconds=plan.total_duration_seconds,
            selection_mode=plan.selection_mode,
            render_encoder=quality_report.render_encoder,
            music_mode=plan.music_plan.mode if plan.music_plan else None,
            music_profile=plan.music_plan.profile if plan.music_plan else None,
            music_generator=plan.music_plan.generator if plan.music_plan else None,
            music_model=plan.music_plan.model if plan.music_plan else None,
            quality_score=quality_report.score,
            quality_issue_count=len(quality_report.issues),
            quality_gate_status=quality_report.gate_status,
            semantic_score_p10=quality_report.semantic_score_p10,
            dominant_event_ratio=quality_report.dominant_event_ratio,
            adjacent_source_repeat_ratio=quality_report.adjacent_source_repeat_ratio,
            center_cut_ratio=quality_report.center_cut_ratio,
            full_media_qa_completed=(
                quality_report.rendered_media_metrics is not None
                and quality_report.rendered_media_metrics.scan_completed
            ),
        )

    def _build_music_plan(
        self,
        context: ProjectContext,
        report: MediaScanReport,
        scenes: list[Scene],
        settings: QuickMontageSettings,
        montage_plan: QuickMontagePlan,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> MusicPlan:
        generator = self._music_generator(settings, progress=progress)
        music_plan = build_music_plan(
            report.assets,
            scenes,
            settings,
            self.settings.music_library.expanduser().resolve(),
            context.artifacts_dir / self.settings.generated_music_filename,
            montage_plan,
            neural_generator=generator,
            ffmpeg_binary=self.settings.ffmpeg_binary,
            progress=progress,
        )
        write_json_atomic(context.artifacts_dir / "music_plan.json", music_plan)
        return music_plan

    def _music_generator(
        self,
        settings: QuickMontageSettings,
        *,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> NeuralMusicGenerator | None:
        if (
            settings.music_mode not in {"auto", "generated"}
            or settings.music_engine == "procedural"
        ):
            return None
        model = settings.music_model or self.settings.music_model
        if self._music_generator_factory is not None:
            return self._music_generator_factory(self.settings, model)
        resources = self.get_resource_profile()
        resolved_model = resolve_local_music_model(
            model,
            gpu_memory_mb=resources.gpu_memory_mb,
        )
        return cast(
            NeuralMusicGenerator,
            AceStepMusicGenerator(
                resolved_model,
                runtime_dir=Path(".cache/ace-step").resolve(),
                model_cache=(self.settings.model_cache / "ace-step").expanduser().resolve(),
                ffmpeg_binary=self.settings.ffmpeg_binary,
                allow_download=self.settings.allow_model_download,
                device=self.settings.device,
                gpu_memory_mb=resources.gpu_memory_mb,
                ffmpeg_timeout_seconds=self.settings.render_timeout_seconds,
                cancel_requested=(
                    (lambda: _music_progress_heartbeat(progress)) if progress is not None else None
                ),
            ),
        )

    def resolve_workspace(self, input_path: Path, workspace: Path | None) -> Path:
        if workspace is not None:
            return workspace.expanduser().resolve()
        legacy_workspace = legacy_workspace_path(self.settings.workspace, input_path)
        if workspace_proves_source(legacy_workspace, input_path):
            return legacy_workspace
        return default_workspace_path(self.settings.workspace, input_path)

    def resolve_project_paths(self, input_path: Path, workspace: Path | None) -> ProjectPaths:
        resolved_workspace = self.resolve_workspace(input_path, workspace)
        project_paths = validate_project_paths(input_path, resolved_workspace)
        validate_existing_workspace_identity(project_paths.input_path, project_paths.workspace)
        return project_paths

    def run_until(
        self,
        target: PipelineStage,
        *,
        input_path: Path,
        workspace: Path | None,
        output_path: Path | None = None,
        style: StoryStyle = StoryStyle.CINEMATIC,
        montage_settings: QuickMontageSettings | None = None,
        progress: Callable[[int, int, str], None] | None = None,
        progress_events: ProgressEventCallback | None = None,
        variant_name: str = "Default",
    ) -> StageResult:
        try:
            normalized_variant_name = validate_variant_name(variant_name)
        except ValueError as error:
            raise MontageError(str(error)) from error
        context = self._context(
            input_path=input_path,
            output_path=output_path,
            workspace=workspace,
            style=style,
            montage_settings=montage_settings,
            variant_name=normalized_variant_name,
            variant_slug=safe_variant_slug(normalized_variant_name),
            resources=(None if target is PipelineStage.MEDIA_SCAN else self.get_resource_profile()),
        )
        return self._pipeline_runner().run_until(
            context,
            target,
            progress=progress,
            progress_events=progress_events,
        )

    def _pipeline_runner(self) -> PipelineRunner:
        vision_factory: VisionProviderFactory | None = None
        if self._vision_provider_factory is not None:
            injected_vision_factory = self._vision_provider_factory

            def build_injected_vision(
                _context: ProjectContext,
                _provider: str,
                _model: str,
                _resources: ResourceProfile,
            ) -> VisionProvider:
                return injected_vision_factory(self.settings)

            vision_factory = build_injected_vision
        music_factory: MusicGeneratorFactory | None = None
        if self._music_generator_factory is not None:

            def build_injected_music(
                _context: ProjectContext,
                montage_settings: QuickMontageSettings,
            ) -> NeuralMusicGenerator | None:
                return self._music_generator(montage_settings)

            music_factory = build_injected_music
        return PipelineRunner(
            build_default_pipeline(
                vision_provider_factory=vision_factory,
                music_generator_factory=music_factory,
            )
        )

    def report(self, *, input_path: Path, workspace: Path | None) -> StageResult:
        context = self._context(input_path=input_path, workspace=workspace)
        with WorkspaceLease(context.workspace, operation="report"):
            report = generate_project_report(context)
        return StageResult(
            stage=PipelineStage.EVENT_DETECTION,
            artifacts=[report.path],
            message=(
                f"HTML report generated for {report.asset_count} asset(s), "
                f"{report.scene_count} scene(s), and {report.event_count} event(s): "
                f"{report.path}"
            ),
        )

    def search(
        self,
        *,
        input_path: Path,
        workspace: Path | None,
        query: str,
        limit: int = 10,
    ) -> SemanticSearchReport:
        context = self._context(input_path=input_path, workspace=workspace)
        return search_project_scenes(context, query, limit=limit)

    def export_project(
        self,
        *,
        input_path: Path,
        workspace: Path | None,
        output_path: Path,
        include_rendered_media: bool = False,
        overwrite: bool = False,
    ) -> StageResult:
        context = self._context(input_path=input_path, workspace=workspace)
        with WorkspaceLease(context.workspace, operation="export"):
            result = export_project_archive(
                context,
                output_path,
                include_rendered_media=include_rendered_media,
                overwrite=overwrite,
            )
        return StageResult(
            stage=PipelineStage.MEDIA_SCAN,
            artifacts=[result.archive_path],
            message=(
                f"Project archive created with {result.file_count} file(s), "
                f"{result.total_bytes} byte(s): {result.archive_path}"
            ),
        )

    def restore_project(self, *, archive_path: Path, workspace: Path) -> StageResult:
        restored = restore_project_archive(archive_path, workspace)
        return StageResult(
            stage=PipelineStage.MEDIA_SCAN,
            artifacts=[restored],
            message=f"Project archive restored to {restored}",
        )

    def _context(
        self,
        *,
        input_path: Path,
        workspace: Path | None,
        output_path: Path | None = None,
        style: StoryStyle = StoryStyle.CINEMATIC,
        montage_settings: QuickMontageSettings | None = None,
        variant_name: str = "Default",
        variant_slug: str = "default",
        resources: ResourceProfile | None = None,
    ) -> ProjectContext:
        project_paths = self.resolve_project_paths(input_path, workspace)
        return ProjectContext(
            input_path=project_paths.input_path,
            workspace=project_paths.workspace,
            output_path=output_path,
            settings=self.settings,
            style=style,
            montage_settings=montage_settings,
            variant_name=variant_name,
            variant_slug=variant_slug,
            resources=resources,
        )


def _effective_montage_settings(
    settings: QuickMontageSettings,
) -> QuickMontageSettings:
    if not settings.preview_mode:
        return settings
    width = min(settings.width, 854)
    height = min(settings.height, 480)
    if width % 2:
        width -= 1
    if height % 2:
        height -= 1
    return settings.model_copy(
        update={
            "width": width,
            "height": height,
            "fps": min(settings.fps, 24),
        }
    )


def _validate_montage_feature_requests(
    montage_settings: QuickMontageSettings,
    application_settings: Settings,
) -> None:
    semantic_render_features = []
    if montage_settings.framing_mode == "smart":
        semantic_render_features.append("smart crop")
    if montage_settings.color_normalization:
        semantic_render_features.append("color normalization")
    if montage_settings.event_titles_enabled:
        semantic_render_features.append("event titles")
    if montage_settings.scene_subtitles_enabled:
        semantic_render_features.append("scene subtitles")
    if semantic_render_features and not montage_settings.semantic_analysis:
        raise MontageError(
            f"{', '.join(semantic_render_features)} require semantic scene analysis."
        )
    if montage_settings.speech_analysis and not montage_settings.semantic_analysis:
        raise MontageError("Speech analysis requires semantic scene selection.")
    if montage_settings.narration_enabled and not montage_settings.semantic_analysis:
        raise MontageError("Narration requires semantic scene selection.")
    if montage_settings.narration_enabled and application_settings.voice_provider == "disabled":
        raise MontageError(
            "Narration was requested, but voice_provider is disabled. Configure the local "
            "Piper executable, model, and voice_provider='piper', or disable narration."
        )


def _music_progress_heartbeat(
    progress: Callable[[int, int, str], None],
) -> bool:
    progress(1, 4, "ACE-Step: generation is still running")
    return False


class _ProgressTracker:
    def __init__(
        self,
        callback: Callable[[int, int, str], None] | None,
    ) -> None:
        self._callback = callback

    def emit(self, percent: float, message: str) -> None:
        if self._callback is not None:
            self._callback(round(max(0, min(100, percent)) * 10), 1000, message)

    def range(
        self,
        start_percent: float,
        end_percent: float,
    ) -> Callable[[int, int, str], None]:
        def report(current: int, total: int, message: str) -> None:
            fraction = current / total if total > 0 else 0
            self.emit(
                start_percent + (end_percent - start_percent) * fraction,
                message,
            )

        return report


def _record_timeline_version(
    context: ProjectContext,
    plan: QuickMontagePlan,
    *,
    phase: Literal["built", "rendered"],
    output_path: Path | None = None,
) -> None:
    repository = MediaAssetRepository(context.database_path)
    try:
        repository.initialize()
        repository.record_timeline_version(
            plan,
            phase=phase,
            variant_name=context.variant_name,
            variant_slug=context.variant_slug,
            output_path=output_path,
        )
    finally:
        repository.close()
