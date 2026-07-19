"""Use-case facade called by the CLI."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import cast
from uuid import UUID

from pydantic import ValidationError

from travelmovieai.analysis.audio import analyze_audio
from travelmovieai.analysis.duplicates import detect_duplicate_scenes
from travelmovieai.analysis.quality import analyze_scene_quality
from travelmovieai.analysis.scenes import RepresentativeFrameExtractor, frame_sample_count_for_mode
from travelmovieai.analysis.speech import analyze_speech
from travelmovieai.analysis.vision import VisionProvider, analyze_scenes
from travelmovieai.application.context import ProjectContext
from travelmovieai.application.validation import (
    ProjectPaths,
    validate_output_path,
    validate_project_paths,
)
from travelmovieai.core.config import Settings
from travelmovieai.core.exceptions import MontageError
from travelmovieai.domain.enums import PipelineStage, StoryStyle
from travelmovieai.domain.models import (
    MediaAsset,
    MediaScanReport,
    MontageQualityReport,
    MusicPlan,
    QualityAnalysisReport,
    QuickMontagePlan,
    QuickMontageResult,
    QuickMontageSettings,
    Scene,
    SceneDetectionReport,
    StageResult,
    VisionAnalysisReport,
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
    build_selection_report,
    build_semantic_montage_plan,
)
from travelmovieai.infrastructure.artifacts import write_json_atomic
from travelmovieai.infrastructure.database import MediaAssetRepository
from travelmovieai.infrastructure.music_generation import (
    AceStepMusicGenerator,
    resolve_local_music_model,
)
from travelmovieai.infrastructure.system import ResourceProfile, detect_resource_profile
from travelmovieai.infrastructure.vision import build_vision_provider
from travelmovieai.infrastructure.whisper import FasterWhisperProvider
from travelmovieai.pipeline.registry import build_default_pipeline
from travelmovieai.pipeline.runner import PipelineRunner
from travelmovieai.pipeline.stages.scene_detection import SceneDetectionStage
from travelmovieai.story.builder import (
    build_multimodal_descriptions,
    build_storyboard,
)
from travelmovieai.story.events import detect_events
from travelmovieai.story.music import NeuralMusicGenerator, build_music_plan
from travelmovieai.story.optimizer import apply_story_structure


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

    def create(
        self,
        *,
        input_path: Path,
        output_path: Path,
        workspace: Path | None,
        style: StoryStyle,
        semantic: bool = False,
    ) -> StageResult:
        result = self.create_quick_montage(
            input_path=input_path,
            workspace=workspace,
            settings=QuickMontageSettings(
                semantic_analysis=semantic,
                story_style=style,
                vision_provider=self.settings.vision_provider,
                vision_model=(
                    None if self.settings.vision_model == "auto" else self.settings.vision_model
                ),
            ),
            output_path=output_path,
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

    def analyze(self, *, input_path: Path, workspace: Path | None) -> StageResult:
        return self.run_until(
            PipelineStage.MEDIA_SCAN,
            input_path=input_path,
            workspace=workspace,
        )

    def create_quick_montage(
        self,
        *,
        input_path: Path,
        workspace: Path | None,
        settings: QuickMontageSettings,
        output_path: Path | None = None,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> QuickMontageResult:
        settings = _effective_montage_settings(settings)
        resources = self.get_resource_profile(refresh=True)
        tracker = _ProgressTracker(progress)
        tracker.emit(0, f"Resource profile: {resources.summary}")
        context = self._context(input_path=input_path, workspace=workspace)
        default_name = "preview.mp4" if settings.preview_mode else "final.mp4"
        resolved_output = validate_output_path(
            output_path or context.artifacts_dir / default_name,
            context.input_path,
            workspace=context.workspace,
            database_path=context.database_path,
        )
        tracker.emit(1, "Checking media library and updating index")
        self.analyze(input_path=context.input_path, workspace=context.workspace)
        tracker.emit(5, "Media scan complete, reading metadata")
        analysis_path = context.artifacts_dir / "analysis.json"
        try:
            report = MediaScanReport.model_validate_json(analysis_path.read_text(encoding="utf-8"))
        except (OSError, ValidationError) as error:
            raise MontageError("Could not read media scan results.") from error

        quality_report_path = context.artifacts_dir / "montage_quality_report.json"
        quality_report: MontageQualityReport | None = None
        if settings.semantic_analysis:
            plan = self._build_semantic_plan(
                context,
                report,
                settings,
                tracker,
                resources,
            )
        else:
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
            write_json_atomic(quality_report_path, quality_report)
            tracker.emit(80, "Quick edit plan created")
        timeline_path = context.artifacts_dir / "quick_timeline.json"
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(timeline_path, plan)
        tracker.emit(84, f"Timeline saved: {len(plan.clips)} clip(s)")
        render_encoder = QuickMontageRenderer(
            self.settings.ffmpeg_binary,
            self.settings.ffprobe_binary,
            workers=resources.render_workers,
            ffmpeg_threads=resources.ffmpeg_threads,
            timeout_seconds=self.settings.render_timeout_seconds,
        ).render(
            plan,
            resolved_output,
            context.cache_dir,
            tracker.range(85, 100),
        )
        if quality_report is None:
            try:
                quality_report = MontageQualityReport.model_validate_json(
                    quality_report_path.read_text(encoding="utf-8")
                )
            except (OSError, ValidationError):
                quality_report = build_montage_quality_report(plan, [])
        if not quality_report_path.is_file():
            write_json_atomic(quality_report_path, quality_report)
        quality_report = enrich_montage_quality_report_with_render(
            quality_report,
            resolved_output,
            ffprobe_binary=self.settings.ffprobe_binary,
            ffmpeg_binary=self.settings.ffmpeg_binary,
            timeout_seconds=self.settings.render_timeout_seconds,
        )
        write_json_atomic(quality_report_path, quality_report)
        enforce_montage_quality(quality_report)
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
        )

    def _build_semantic_plan(
        self,
        context: ProjectContext,
        report: MediaScanReport,
        settings: QuickMontageSettings,
        tracker: _ProgressTracker,
        resources: ResourceProfile,
    ) -> QuickMontagePlan:
        tracker.emit(6, "Scene detection")
        SceneDetectionStage(settings=settings).run(context)
        scenes_path = context.artifacts_dir / "scenes.json"
        try:
            scene_report = SceneDetectionReport.model_validate_json(
                scenes_path.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError) as error:
            raise MontageError("Could not read scene detection results.") from error

        assets_by_id = {asset.id: asset for asset in report.assets}
        use_cuda_decode = resources.nvenc and self.settings.device in {"auto", "cuda"}
        frame_workers = (
            min(
                resources.frame_workers,
                1 if resources.resource_mode == "safe" else self.settings.max_gpu_processes,
            )
            if use_cuda_decode
            else resources.frame_workers
        )
        extractor = RepresentativeFrameExtractor(
            self.settings.ffmpeg_binary,
            self.settings.ffprobe_binary,
            use_cuda_decode=use_cuda_decode,
            frame_sample_count=frame_sample_count_for_mode(settings.analysis_quality_mode),
            timeout_seconds=self.settings.frame_extraction_timeout_seconds,
        )
        tracker.emit(
            12,
            f"Scenes found: {len(scene_report.scenes)}. "
            f"Extracting frames with {frame_workers} worker(s), "
            f"decode={'NVDEC' if use_cuda_decode else 'CPU'}",
        )
        prepared_scenes = _extract_scene_frames(
            scene_report.scenes,
            assets_by_id,
            extractor,
            context.frames_dir,
            frame_workers,
            tracker.range(12, 32),
        )
        tracker.emit(32, f"Frame preparation complete: {extractor.backend_summary}")

        quality_report = (
            analyze_scene_quality(
                prepared_scenes,
                workers=resources.analysis_workers,
                progress=tracker.range(32, 45),
            )
            if settings.quality_analysis
            else QualityAnalysisReport(
                created_at=scene_report.created_at,
                scenes=prepared_scenes,
            )
        )
        quality_path = context.artifacts_dir / "quality_analysis.json"
        write_json_atomic(quality_path, quality_report)
        quality_backend = next(
            (
                scene.metadata.get("quality_metrics", {}).get("backend")
                for scene in quality_report.scenes
                if scene.metadata.get("quality_metrics")
            ),
            "disabled",
        )
        tracker.emit(45, f"Quality analysis saved, backend={quality_backend}")
        vision_path = context.artifacts_dir / "vision_analysis.json"
        try:
            cached_vision_report = VisionAnalysisReport.model_validate_json(
                vision_path.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError):
            cached_vision_report = None
        vision_provider = self._vision_provider(
            settings.vision_provider,
            settings.vision_model,
        )
        tracker.emit(
            45,
            f"Vision AI: checking cache for {vision_provider.model}. "
            "The model loads only when a scene still needs analysis",
        )
        try:
            vision_report = analyze_scenes(
                quality_report.scenes,
                vision_provider,
                settings.story_style,
                tracker.range(45, 70),
                cached_report=cached_vision_report,
                checkpoint=lambda partial: write_json_atomic(vision_path, partial),
            )
            runtime = getattr(vision_provider, "runtime_description", "ready")
            tracker.emit(
                70,
                f"Vision AI complete: {vision_report.analyzed_count} analyzed, "
                f"{vision_report.cached_count} cached ({runtime})",
            )
        finally:
            release = getattr(vision_provider, "release", None)
            if callable(release):
                release()
        write_json_atomic(vision_path, vision_report)
        speech_scenes = vision_report.scenes
        if settings.speech_analysis:
            tracker.emit(70, "Whisper: transcribing speech and important lines")
            whisper_provider = FasterWhisperProvider(
                self.settings.whisper_model,
                self.settings.device,
            )
            try:
                speech_report = analyze_speech(
                    vision_report.scenes,
                    report.assets,
                    whisper_provider,
                    self.settings.ffmpeg_binary,
                    context.cache_dir / "speech",
                    timeout_seconds=self.settings.frame_extraction_timeout_seconds,
                    progress=tracker.range(70, 74),
                )
            finally:
                whisper_provider.release()
            speech_scenes = speech_report.scenes
            write_json_atomic(
                context.artifacts_dir / "speech_analysis.json",
                speech_report,
            )
        else:
            tracker.emit(74, "Speech recognition disabled")
        if settings.audio_analysis:
            tracker.emit(74, "Audio Analysis: classifying speech, silence, and noise")
            audio_report = analyze_audio(
                speech_scenes,
                report.assets,
                self.settings.ffmpeg_binary,
                timeout_seconds=self.settings.frame_extraction_timeout_seconds,
                progress=tracker.range(74, 76),
            )
            speech_scenes = audio_report.scenes
            write_json_atomic(
                context.artifacts_dir / "audio_analysis.json",
                audio_report,
            )
        else:
            tracker.emit(76, "Audio Analysis disabled")
        if settings.duplicate_detection:
            tracker.emit(76, "Finding similar and duplicate scenes")
            duplicate_report, deduplicated_scenes = detect_duplicate_scenes(
                speech_scenes,
                settings.duplicate_similarity_threshold,
            )
            write_json_atomic(
                context.artifacts_dir / "duplicates.json",
                duplicate_report,
            )
        else:
            deduplicated_scenes = speech_scenes
        tracker.emit(77, "Merging Vision AI, OpenCV, and audio metadata")
        descriptions = build_multimodal_descriptions(deduplicated_scenes)
        write_json_atomic(
            context.artifacts_dir / "scene_descriptions.json",
            descriptions,
        )
        event_report, event_scenes = detect_events(
            deduplicated_scenes,
            report.assets,
        )
        tracker.emit(79, f"Trip events grouped: {len(event_report.events)}")
        events_path = context.artifacts_dir / "events.json"
        write_json_atomic(events_path, event_report)
        storyboard = build_storyboard(
            event_report.events,
            event_scenes,
            settings.story_style,
        )
        event_scenes = apply_story_structure(event_scenes, storyboard)
        tracker.emit(81, "Story structure and scene order created")
        write_json_atomic(context.artifacts_dir / "storyboard.json", storyboard)
        repository = MediaAssetRepository(context.database_path)
        repository.initialize()
        repository.synchronize_scenes(event_scenes)
        repository.synchronize_events(event_report.events)
        plan = build_semantic_montage_plan(
            report.assets,
            event_scenes,
            settings,
        )
        tracker.emit(82, f"AI selection complete: {len(plan.clips)} clip(s)")
        write_json_atomic(
            context.artifacts_dir / "selection_decisions.json",
            build_selection_report(event_scenes, plan, settings),
        )
        tracker.emit(82, "Building a music map from scene and event importance")
        music_plan = self._build_music_plan(
            context,
            report,
            event_scenes,
            settings,
            plan,
            tracker.range(82, 84),
        )
        tracker.emit(
            83,
            f"Music: {music_plan.mode}, profile {music_plan.profile}, "
            f"{len(music_plan.accents)} accent(s)",
        )
        final_plan = plan.model_copy(
            update={
                "music_plan": music_plan,
                "music_path": music_plan.source_path,
            }
        )
        final_plan = apply_music_directing(final_plan, event_scenes)
        montage_quality_report = build_montage_quality_report(final_plan, event_scenes)
        write_json_atomic(
            context.artifacts_dir / "montage_quality_report.json",
            montage_quality_report,
        )
        return final_plan

    def _build_music_plan(
        self,
        context: ProjectContext,
        report: MediaScanReport,
        scenes: list[Scene],
        settings: QuickMontageSettings,
        montage_plan: QuickMontagePlan,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> MusicPlan:
        generator = self._music_generator(settings)
        music_plan = build_music_plan(
            report.assets,
            scenes,
            settings,
            self.settings.music_library.expanduser().resolve(),
            context.artifacts_dir / self.settings.generated_music_filename,
            montage_plan,
            neural_generator=generator,
            progress=progress,
        )
        write_json_atomic(context.artifacts_dir / "music_plan.json", music_plan)
        return music_plan

    def _music_generator(
        self,
        settings: QuickMontageSettings,
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
            ),
        )

    def _vision_provider(
        self,
        provider: str = "local",
        model: str | None = None,
    ) -> VisionProvider:
        if self._vision_provider_factory is not None:
            return self._vision_provider_factory(self.settings)
        resources = self.get_resource_profile()
        return build_vision_provider(
            provider=provider,
            model=model or self.settings.vision_model,
            device=self.settings.device,
            cache_dir=self.settings.model_cache.expanduser().resolve(),
            allow_download=self.settings.allow_model_download,
            gpu_memory_mb=resources.gpu_memory_mb,
            system_memory_mb=resources.memory_mb,
            model_batch_size=resources.model_batch_size,
        )

    def resolve_workspace(self, input_path: Path, workspace: Path | None) -> Path:
        return (workspace or self.settings.workspace / input_path.name).resolve()

    def resolve_project_paths(self, input_path: Path, workspace: Path | None) -> ProjectPaths:
        resolved_workspace = self.resolve_workspace(input_path, workspace)
        return validate_project_paths(input_path, resolved_workspace)

    def run_until(
        self,
        target: PipelineStage,
        *,
        input_path: Path,
        workspace: Path | None,
        output_path: Path | None = None,
        style: StoryStyle = StoryStyle.CINEMATIC,
        montage_settings: QuickMontageSettings | None = None,
    ) -> StageResult:
        context = self._context(
            input_path=input_path,
            output_path=output_path,
            workspace=workspace,
            style=style,
            montage_settings=montage_settings,
        )
        return PipelineRunner(build_default_pipeline()).run_until(context, target)

    def report(self, *, input_path: Path, workspace: Path | None) -> StageResult:
        context = self._context(input_path=input_path, workspace=workspace)
        context.prepare()
        report_path = context.artifacts_dir / "report.html"
        return StageResult(
            stage=PipelineStage.EVENT_DETECTION,
            skipped=True,
            artifacts=[],
            message=(
                "Project structure is ready. Report generation will be implemented "
                f"in a later milestone: {report_path}"
            ),
        )

    def _context(
        self,
        *,
        input_path: Path,
        workspace: Path | None,
        output_path: Path | None = None,
        style: StoryStyle = StoryStyle.CINEMATIC,
        montage_settings: QuickMontageSettings | None = None,
    ) -> ProjectContext:
        project_paths = self.resolve_project_paths(input_path, workspace)
        return ProjectContext(
            input_path=project_paths.input_path,
            workspace=project_paths.workspace,
            output_path=output_path,
            settings=self.settings,
            style=style,
            montage_settings=montage_settings,
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


def _extract_scene_frames(
    scenes: list[Scene],
    assets_by_id: dict[UUID, MediaAsset],
    extractor: RepresentativeFrameExtractor,
    frames_dir: Path,
    workers: int,
    progress: Callable[[int, int, str], None],
) -> list[Scene]:
    jobs = [(index, scene, assets_by_id.get(scene.asset_id)) for index, scene in enumerate(scenes)]
    valid_jobs = [(index, scene, asset) for index, scene, asset in jobs if asset is not None]
    prepared: dict[int, Scene] = {}
    worker_count = min(max(1, workers), max(1, len(valid_jobs)))

    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="travelmovieai-frames",
    ) as executor:
        futures = {
            executor.submit(extractor.extract, scene, asset, frames_dir): (index, scene)
            for index, scene, asset in valid_jobs
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            index, scene = futures[future]
            prepared[index] = scene.model_copy(update={"keyframe_path": future.result()})
            progress(
                completed,
                len(valid_jobs),
                f"Frames: {completed}/{len(valid_jobs)}, workers={worker_count}",
            )
    return [prepared[index] for index in sorted(prepared)]
