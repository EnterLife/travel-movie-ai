"""Use-case facade called by the CLI."""

from collections.abc import Callable
from pathlib import Path

from pydantic import ValidationError

from travelmovieai.analysis.scenes import RepresentativeFrameExtractor
from travelmovieai.analysis.vision import VisionProvider, analyze_scenes
from travelmovieai.application.context import ProjectContext
from travelmovieai.application.validation import ProjectPaths, validate_project_paths
from travelmovieai.core.config import Settings
from travelmovieai.core.exceptions import MontageError
from travelmovieai.domain.enums import PipelineStage, StoryStyle
from travelmovieai.domain.models import (
    MediaScanReport,
    QuickMontagePlan,
    QuickMontageResult,
    QuickMontageSettings,
    SceneDetectionReport,
    StageResult,
)
from travelmovieai.editing.renderer import QuickMontageRenderer
from travelmovieai.editing.timeline import (
    build_quick_montage_plan,
    build_semantic_montage_plan,
)
from travelmovieai.infrastructure.artifacts import write_json_atomic
from travelmovieai.infrastructure.database import MediaAssetRepository
from travelmovieai.infrastructure.vision import LMStudioVisionProvider
from travelmovieai.pipeline.registry import build_default_pipeline
from travelmovieai.pipeline.runner import PipelineRunner
from travelmovieai.pipeline.stages.scene_detection import SceneDetectionStage
from travelmovieai.story.music import select_music


class TravelMovieService:
    def __init__(
        self,
        settings: Settings,
        vision_provider_factory: Callable[[Settings], VisionProvider] | None = None,
    ) -> None:
        self.settings = settings
        self._vision_provider_factory = vision_provider_factory

    def create(
        self,
        *,
        input_path: Path,
        output_path: Path,
        workspace: Path | None,
        style: StoryStyle,
        cloud: bool,
        semantic: bool = False,
    ) -> StageResult:
        result = self.create_quick_montage(
            input_path=input_path,
            workspace=workspace,
            settings=QuickMontageSettings(
                semantic_analysis=semantic,
                story_style=style,
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
        context = self._context(input_path=input_path, workspace=workspace)
        self.analyze(input_path=context.input_path, workspace=context.workspace)
        analysis_path = context.artifacts_dir / "analysis.json"
        try:
            report = MediaScanReport.model_validate_json(analysis_path.read_text(encoding="utf-8"))
        except (OSError, ValidationError) as error:
            raise MontageError("Не удалось прочитать результаты Media Scan.") from error

        music_path = select_music(
            report.assets,
            settings,
            self.settings.music_library.expanduser().resolve(),
        )
        if settings.semantic_analysis:
            plan = self._build_semantic_plan(
                context,
                report,
                settings,
                music_path,
                progress,
            )
        else:
            plan = build_quick_montage_plan(report.assets, settings, music_path)
        timeline_path = context.artifacts_dir / "quick_timeline.json"
        resolved_output = (output_path or context.artifacts_dir / "final.mp4").resolve()
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(timeline_path, plan)
        QuickMontageRenderer(self.settings.ffmpeg_binary).render(
            plan,
            resolved_output,
            context.cache_dir,
            progress,
        )
        return QuickMontageResult(
            output_path=resolved_output,
            timeline_path=timeline_path,
            clip_count=len(plan.clips),
            duration_seconds=plan.total_duration_seconds,
            selection_mode=plan.selection_mode,
        )

    def _build_semantic_plan(
        self,
        context: ProjectContext,
        report: MediaScanReport,
        settings: QuickMontageSettings,
        music_path: Path | None,
        progress: Callable[[int, int, str], None] | None,
    ) -> QuickMontagePlan:
        if progress:
            progress(0, 1, "Детектирование сцен")
        SceneDetectionStage(settings=settings).run(context)
        scenes_path = context.artifacts_dir / "scenes.json"
        try:
            scene_report = SceneDetectionReport.model_validate_json(
                scenes_path.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError) as error:
            raise MontageError("Не удалось прочитать результаты Scene Detection.") from error

        assets_by_id = {asset.id: asset for asset in report.assets}
        extractor = RepresentativeFrameExtractor(self.settings.ffmpeg_binary)
        prepared_scenes = []
        for index, scene in enumerate(scene_report.scenes, start=1):
            asset = assets_by_id.get(scene.asset_id)
            if asset is None:
                continue
            if progress:
                progress(
                    index - 1,
                    len(scene_report.scenes),
                    f"Кадр сцены {index}/{len(scene_report.scenes)}",
                )
            frame_path = extractor.extract(scene, asset, context.frames_dir)
            prepared_scenes.append(scene.model_copy(update={"keyframe_path": frame_path}))

        provider = self._vision_provider()
        vision_report = analyze_scenes(
            prepared_scenes,
            provider,
            settings.story_style,
            progress,
        )
        vision_path = context.artifacts_dir / "vision_analysis.json"
        write_json_atomic(vision_path, vision_report)
        repository = MediaAssetRepository(context.database_path)
        repository.initialize()
        repository.synchronize_scenes(vision_report.scenes)
        return build_semantic_montage_plan(
            report.assets,
            vision_report.scenes,
            settings,
            music_path,
        )

    def _vision_provider(self) -> VisionProvider:
        if self._vision_provider_factory is not None:
            return self._vision_provider_factory(self.settings)
        return LMStudioVisionProvider(
            base_url=self.settings.lm_studio_url,
            model=self.settings.vision_model,
            timeout_seconds=self.settings.vision_timeout_seconds,
            api_key=self.settings.lm_studio_api_key,
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
    ) -> StageResult:
        context = self._context(
            input_path=input_path,
            output_path=output_path,
            workspace=workspace,
            style=style,
        )
        return PipelineRunner(build_default_pipeline()).run_until(context, target)

    def report(self, *, input_path: Path, workspace: Path | None) -> StageResult:
        context = self._context(input_path=input_path, workspace=workspace)
        context.prepare()
        report_path = context.artifacts_dir / "report.html"
        return StageResult(
            stage=PipelineStage.EVENT_DETECTION,
            skipped=True,
            artifacts=[report_path],
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
        cloud: bool = False,
    ) -> ProjectContext:
        project_paths = self.resolve_project_paths(input_path, workspace)
        return ProjectContext(
            input_path=project_paths.input_path,
            workspace=project_paths.workspace,
            output_path=output_path,
            settings=self.settings,
            style=style,
            cloud=cloud or self.settings.cloud_enabled,
        )
