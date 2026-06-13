"""Pipeline stage for structured local Vision AI scene understanding."""

from travelmovieai.analysis.vision import VisionProvider, analyze_scenes
from travelmovieai.application.context import ProjectContext
from travelmovieai.domain.enums import PipelineStage
from travelmovieai.domain.models import StageResult
from travelmovieai.infrastructure.artifacts import write_json_atomic
from travelmovieai.infrastructure.database import MediaAssetRepository
from travelmovieai.infrastructure.lm_studio import (
    list_lm_studio_models,
    resolve_vision_model,
)
from travelmovieai.infrastructure.vision import (
    Florence2VisionProvider,
    LMStudioVisionProvider,
)
from travelmovieai.pipeline.base import Stage


class VisionAnalysisStage(Stage):
    name = PipelineStage.VISION_ANALYSIS

    def run(self, context: ProjectContext) -> StageResult:
        repository = MediaAssetRepository(context.database_path)
        repository.initialize()
        provider: VisionProvider
        if context.settings.vision_provider == "florence":
            provider = Florence2VisionProvider(
                model=(
                    context.settings.vision_model
                    if context.settings.vision_model != "auto"
                    else "microsoft/Florence-2-large"
                ),
                device=context.settings.device,
            )
        else:
            discovered = list_lm_studio_models(
                context.settings.lm_studio_url,
                context.settings.lm_studio_api_key,
                5,
            )
            model = resolve_vision_model(discovered, context.settings.vision_model)
            provider = LMStudioVisionProvider(
                base_url=context.settings.lm_studio_url,
                model=model,
                timeout_seconds=context.settings.vision_timeout_seconds,
                api_key=context.settings.lm_studio_api_key,
            )
        report = analyze_scenes(repository.list_scenes(), provider, context.style)
        repository.synchronize_scenes(report.scenes)
        artifact = context.artifacts_dir / "vision_analysis.json"
        write_json_atomic(artifact, report)
        return StageResult(
            stage=self.name,
            skipped=report.analyzed_count == 0,
            artifacts=[context.database_path, artifact],
            message=(
                f"Vision AI analyzed {report.analyzed_count} scene(s), "
                f"{report.cached_count} cached, model {report.model}."
            ),
        )
