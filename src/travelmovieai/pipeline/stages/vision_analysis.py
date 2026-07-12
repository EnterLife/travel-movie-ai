"""Pipeline stage for structured local Vision AI scene understanding."""

from pathlib import Path

from travelmovieai.analysis.vision import VisionProvider, analyze_scenes
from travelmovieai.application.context import ProjectContext
from travelmovieai.domain.enums import PipelineStage
from travelmovieai.domain.models import Scene, StageResult, VisionAnalysisReport
from travelmovieai.infrastructure.artifacts import (
    artifact_fingerprint,
    stage_cache_manifest_matches,
    write_json_atomic,
    write_stage_cache_manifest,
)
from travelmovieai.infrastructure.database import MediaAssetRepository
from travelmovieai.infrastructure.system import detect_resource_profile
from travelmovieai.infrastructure.vision import build_vision_provider
from travelmovieai.pipeline.base import Stage

ARTIFACT_SCHEMA_VERSION = "vision-analysis-v1"


class VisionAnalysisStage(Stage):
    name = PipelineStage.VISION_ANALYSIS

    def run(self, context: ProjectContext) -> StageResult:
        montage_settings = context.montage_settings
        provider_name = (
            montage_settings.vision_provider
            if montage_settings is not None
            else context.settings.vision_provider
        )
        model_name = (
            montage_settings.vision_model
            if montage_settings is not None and montage_settings.vision_model is not None
            else context.settings.vision_model
        )
        repository = MediaAssetRepository(context.database_path)
        repository.initialize()
        scenes = repository.list_scenes()
        artifact = context.artifacts_dir / "vision_analysis.json"
        resources = detect_resource_profile(
            context.settings.ffmpeg_binary,
            worker_override=context.settings.workers,
            batch_override=context.settings.batch_size,
            resource_mode=context.settings.resource_mode,
            gpu_memory_reserve_mb=context.settings.gpu_memory_reserve_mb,
            max_gpu_processes=context.settings.max_gpu_processes,
        )
        input_fingerprint = artifact_fingerprint(_vision_inputs(scenes))
        config_fingerprint = artifact_fingerprint(
            {
                "provider": provider_name,
                "model": model_name,
                "device": context.settings.device,
                "allow_model_download": context.settings.allow_model_download,
                "model_batch_size": resources.model_batch_size,
                "style": context.style,
                "schema": ARTIFACT_SCHEMA_VERSION,
            }
        )
        cache_artifact = context.artifacts_dir / "vision_analysis.cache.json"
        if stage_cache_manifest_matches(
            cache_artifact,
            stage=self.name,
            artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
            input_fingerprint=input_fingerprint,
            config_fingerprint=config_fingerprint,
            artifacts=[artifact],
        ) and _cached_vision_analysis_valid(artifact, scenes):
            return StageResult(
                stage=self.name,
                skipped=True,
                artifacts=[context.database_path, artifact, cache_artifact],
                message="Vision AI reused cached analysis artifacts.",
            )

        provider: VisionProvider = build_vision_provider(
            provider=provider_name,
            model=model_name,
            device=context.settings.device,
            cache_dir=context.settings.model_cache.expanduser().resolve(),
            allow_download=context.settings.allow_model_download,
            gpu_memory_mb=resources.gpu_memory_mb,
            system_memory_mb=resources.memory_mb,
            model_batch_size=resources.model_batch_size,
        )
        try:
            report = analyze_scenes(scenes, provider, context.style)
        finally:
            release = getattr(provider, "release", None)
            if callable(release):
                release()
        repository.synchronize_scenes(report.scenes)
        write_json_atomic(artifact, report)
        write_stage_cache_manifest(
            cache_artifact,
            stage=self.name,
            artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
            input_fingerprint=input_fingerprint,
            config_fingerprint=config_fingerprint,
            artifacts=[artifact],
        )
        return StageResult(
            stage=self.name,
            skipped=report.analyzed_count == 0,
            artifacts=[context.database_path, artifact, cache_artifact],
            message=(
                f"Vision AI analyzed {report.analyzed_count} scene(s), "
                f"{report.cached_count} cached, model {report.model}."
            ),
        )


def _cached_vision_analysis_valid(artifact: Path, scenes: list[Scene]) -> bool:
    try:
        report = VisionAnalysisReport.model_validate_json(artifact.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if {scene.id for scene in report.scenes} != {scene.id for scene in scenes}:
        return False
    return all(scene.caption and scene.importance_score is not None for scene in scenes)


def _vision_inputs(scenes: list[Scene]) -> list[dict[str, object]]:
    return [
        {
            "id": str(scene.id),
            "asset_id": str(scene.asset_id),
            "start_seconds": scene.start_seconds,
            "end_seconds": scene.end_seconds,
            "keyframe_path": scene.keyframe_path,
            "quality_score": scene.quality_score,
            "transcript": scene.transcript,
            "scene_cache_key": scene.metadata.get("cache_key"),
            "quality_metrics": scene.metadata.get("quality_metrics"),
            "speech_cache_key": scene.metadata.get("speech_cache_key"),
        }
        for scene in sorted(scenes, key=lambda item: str(item.id))
    ]
