"""Pipeline stage for perceptual duplicate scene detection."""

from travelmovieai.analysis.duplicates import detect_duplicate_scenes
from travelmovieai.application.context import ProjectContext
from travelmovieai.domain.enums import PipelineStage, StageStatus
from travelmovieai.domain.models import StageResult
from travelmovieai.infrastructure.artifacts import (
    artifact_fingerprint,
    stage_cache_manifest_matches,
    write_json_atomic,
    write_stage_cache_manifest,
)
from travelmovieai.infrastructure.database import MediaAssetRepository
from travelmovieai.pipeline.base import Stage
from travelmovieai.pipeline.state import DUPLICATE_STATE, clear_stage_owned_state

ARTIFACT_SCHEMA_VERSION = "duplicate-detection-v1"


class DuplicateDetectionStage(Stage):
    name = PipelineStage.DUPLICATE_DETECTION

    def run(self, context: ProjectContext) -> StageResult:
        if (
            context.montage_settings is not None
            and not context.montage_settings.duplicate_detection
        ):
            clear_stage_owned_state(context, DUPLICATE_STATE)
            return StageResult(
                stage=self.name,
                status=StageStatus.DISABLED,
                message="Duplicate detection disabled by montage settings.",
            )

        repository = MediaAssetRepository(context.database_path)
        repository.initialize()
        threshold = (
            context.montage_settings.duplicate_similarity_threshold
            if context.montage_settings is not None
            else 0.92
        )
        scenes = repository.list_scenes()
        artifact = context.artifacts_dir / "duplicates.json"
        cache_artifact = context.artifacts_dir / "duplicates.cache.json"
        if not scenes:
            clear_stage_owned_state(context, DUPLICATE_STATE)
            return StageResult(
                stage=self.name,
                status=StageStatus.NO_INPUT,
                message="Duplicate detection needs scene metadata.",
            )
        input_fingerprint = artifact_fingerprint(
            [
                {
                    "id": str(scene.id),
                    "keyframe_path": scene.keyframe_path,
                    "quality_score": scene.quality_score,
                    "importance_score": scene.importance_score,
                    "selection_override": scene.metadata.get("selection_override"),
                    "embedding_backend": scene.metadata.get("embedding_backend"),
                }
                for scene in scenes
            ]
        )
        config_fingerprint = artifact_fingerprint(threshold, ARTIFACT_SCHEMA_VERSION)
        if stage_cache_manifest_matches(
            cache_artifact,
            stage=self.name,
            artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
            input_fingerprint=input_fingerprint,
            config_fingerprint=config_fingerprint,
            artifacts=[artifact],
        ) and all("duplicate_status" in scene.metadata for scene in scenes):
            return StageResult(
                stage=self.name,
                status=StageStatus.CACHED,
                artifacts=[context.database_path, artifact, cache_artifact],
                message="Duplicate detection reused cached groups.",
            )
        report, scenes = detect_duplicate_scenes(scenes, threshold)
        repository.synchronize_scenes(scenes)
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
                f"Duplicate detection found {report.duplicate_count} duplicate "
                f"scene(s) in {len(report.groups)} group(s)."
            ),
        )
