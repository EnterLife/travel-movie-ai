"""Pipeline stage for structured local Vision AI scene understanding."""

from travelmovieai.analysis.vision import VisionProvider, analyze_scenes
from travelmovieai.application.context import ProjectContext
from travelmovieai.domain.enums import PipelineStage
from travelmovieai.domain.models import StageResult
from travelmovieai.infrastructure.artifacts import write_json_atomic
from travelmovieai.infrastructure.database import MediaAssetRepository
from travelmovieai.infrastructure.system import detect_resource_profile
from travelmovieai.infrastructure.vision import build_vision_provider
from travelmovieai.pipeline.base import Stage


class VisionAnalysisStage(Stage):
    name = PipelineStage.VISION_ANALYSIS

    def run(self, context: ProjectContext) -> StageResult:
        repository = MediaAssetRepository(context.database_path)
        repository.initialize()
        resources = detect_resource_profile(
            context.settings.ffmpeg_binary,
            worker_override=context.settings.workers,
            batch_override=context.settings.batch_size,
        )
        provider: VisionProvider = build_vision_provider(
            provider=context.settings.vision_provider,
            model=context.settings.vision_model,
            device=context.settings.device,
            cache_dir=context.settings.model_cache.expanduser().resolve(),
            allow_download=context.settings.allow_model_download,
            gpu_memory_mb=resources.gpu_memory_mb,
            system_memory_mb=resources.memory_mb,
            lm_studio_url=context.settings.lm_studio_url,
            lm_studio_api_key=context.settings.lm_studio_api_key,
            timeout_seconds=context.settings.vision_timeout_seconds,
            model_batch_size=resources.model_batch_size,
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
