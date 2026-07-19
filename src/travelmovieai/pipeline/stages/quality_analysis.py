"""Pipeline stage for OpenCV visual quality analysis."""

from pathlib import Path

from travelmovieai.analysis.quality import (
    analyze_scene_quality,
    create_quality_analyzer,
    resolve_quality_backend,
)
from travelmovieai.application.context import ProjectContext
from travelmovieai.domain.enums import PipelineStage, StageStatus
from travelmovieai.domain.models import QualityAnalysisReport, Scene, StageResult
from travelmovieai.infrastructure.artifacts import (
    artifact_fingerprint,
    stage_cache_manifest_matches,
    write_json_atomic,
    write_stage_cache_manifest,
)
from travelmovieai.infrastructure.database import MediaAssetRepository
from travelmovieai.infrastructure.system import detect_resource_profile
from travelmovieai.pipeline.base import Stage
from travelmovieai.pipeline.state import QUALITY_STATE, clear_stage_owned_state

ARTIFACT_SCHEMA_VERSION = "quality-analysis-v2"


class QualityAnalysisStage(Stage):
    name = PipelineStage.QUALITY_ANALYSIS

    def run(self, context: ProjectContext) -> StageResult:
        if context.montage_settings is not None and not context.montage_settings.quality_analysis:
            clear_stage_owned_state(context, QUALITY_STATE)
            return StageResult(
                stage=self.name,
                status=StageStatus.DISABLED,
                message="Visual quality analysis disabled by montage settings.",
            )

        repository = MediaAssetRepository(context.database_path)
        repository.initialize()
        scenes = repository.list_scenes()
        artifact = context.artifacts_dir / "quality_analysis.json"
        cache_artifact = context.artifacts_dir / "quality_analysis.cache.json"
        if not any(scene.keyframe_path is not None for scene in scenes):
            clear_stage_owned_state(context, QUALITY_STATE)
            return StageResult(
                stage=self.name,
                status=StageStatus.NO_INPUT,
                message="Visual quality analysis needs sampled scene frames.",
            )
        input_fingerprint = artifact_fingerprint(_quality_inputs(scenes))
        quality_backend = resolve_quality_backend(context.settings.device)
        config_fingerprint = artifact_fingerprint(
            {
                "requested_device": context.settings.device,
                "resolved_backend": quality_backend.fingerprint_payload(),
                "schema": ARTIFACT_SCHEMA_VERSION,
            }
        )
        if stage_cache_manifest_matches(
            cache_artifact,
            stage=self.name,
            artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
            input_fingerprint=input_fingerprint,
            config_fingerprint=config_fingerprint,
            artifacts=[artifact],
        ) and _cached_quality_analysis_valid(artifact, scenes):
            return StageResult(
                stage=self.name,
                status=StageStatus.CACHED,
                artifacts=[context.database_path, artifact, cache_artifact],
                message="Visual quality reused cached analysis artifacts.",
            )

        resources = detect_resource_profile(
            context.settings.ffmpeg_binary,
            worker_override=context.settings.workers,
            batch_override=context.settings.batch_size,
            resource_mode=context.settings.resource_mode,
            gpu_memory_reserve_mb=context.settings.gpu_memory_reserve_mb,
            max_gpu_processes=context.settings.max_gpu_processes,
        )
        report = analyze_scene_quality(
            scenes,
            analyzer=create_quality_analyzer(quality_backend),
            workers=resources.analysis_workers,
            progress=context.progress,
        )
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
            artifacts=[context.database_path, artifact, cache_artifact],
            message=(
                f"Visual quality analyzed for {len(report.scenes)} scene(s), "
                f"backend={quality_backend.name}, "
                f"workers={min(max(1, resources.analysis_workers), max(1, len(scenes)))}."
            ),
        )


def _cached_quality_analysis_valid(artifact: Path, scenes: list[Scene]) -> bool:
    try:
        report = QualityAnalysisReport.model_validate_json(artifact.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if {scene.id for scene in report.scenes} != {scene.id for scene in scenes}:
        return False
    return all(
        scene.quality_score is not None and isinstance(scene.metadata.get("quality_metrics"), dict)
        for scene in scenes
    )


def _quality_inputs(scenes: list[Scene]) -> list[dict[str, object]]:
    return [
        {
            "id": str(scene.id),
            "asset_id": str(scene.asset_id),
            "start_seconds": scene.start_seconds,
            "end_seconds": scene.end_seconds,
            "keyframe_path": scene.keyframe_path,
            "scene_cache_key": scene.metadata.get("cache_key"),
        }
        for scene in sorted(scenes, key=lambda item: str(item.id))
    ]
