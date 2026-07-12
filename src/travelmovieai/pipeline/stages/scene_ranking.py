"""Pipeline stage for explainable semantic scene ranking."""

from datetime import UTC, datetime

from travelmovieai.application.context import ProjectContext
from travelmovieai.domain.enums import PipelineStage
from travelmovieai.domain.models import SceneDetectionReport, StageResult
from travelmovieai.infrastructure.artifacts import (
    artifact_fingerprint,
    stage_cache_manifest_matches,
    write_json_atomic,
    write_stage_cache_manifest,
)
from travelmovieai.infrastructure.database import MediaAssetRepository
from travelmovieai.pipeline.base import Stage
from travelmovieai.story.ranking import rank_scenes

ARTIFACT_SCHEMA_VERSION = "scene-ranking-v1"


class SceneRankingStage(Stage):
    name = PipelineStage.SCENE_RANKING

    def run(self, context: ProjectContext) -> StageResult:
        repository = MediaAssetRepository(context.database_path)
        repository.initialize()
        scenes = repository.list_scenes()
        artifact = context.artifacts_dir / "ranked_scenes.json"
        cache_artifact = context.artifacts_dir / "ranked_scenes.cache.json"
        input_fingerprint = artifact_fingerprint(
            [
                {
                    "id": str(scene.id),
                    "importance": scene.importance_score,
                    "quality": scene.quality_score,
                    "event_importance": scene.metadata.get("event_importance"),
                    "landmarks": scene.metadata.get("landmarks"),
                    "people_count": scene.metadata.get("people_count"),
                    "people_groups": scene.metadata.get("people_groups"),
                    "audio_features": scene.metadata.get("audio_features"),
                    "technical": scene.metadata.get("technical_rejection_reasons"),
                    "duplicate_status": scene.metadata.get("duplicate_status"),
                    "location_type": scene.metadata.get("location_type"),
                    "activity": scene.metadata.get("activity"),
                    "emotion": scene.metadata.get("emotion"),
                    "tags": scene.metadata.get("tags"),
                    "story_role": scene.metadata.get("story_section_role"),
                }
                for scene in scenes
            ]
        )
        config_fingerprint = artifact_fingerprint(ARTIFACT_SCHEMA_VERSION)
        if stage_cache_manifest_matches(
            cache_artifact,
            stage=self.name,
            artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
            input_fingerprint=input_fingerprint,
            config_fingerprint=config_fingerprint,
            artifacts=[artifact],
        ) and all("ranking_score" in scene.metadata for scene in scenes):
            return StageResult(
                stage=self.name,
                skipped=True,
                artifacts=[context.database_path, artifact, cache_artifact],
                message="Scene ranking reused cached scores.",
            )
        ranked_scenes = rank_scenes(scenes)
        repository.synchronize_scenes(ranked_scenes)
        report = SceneDetectionReport(
            created_at=datetime.now(UTC),
            scenes=ranked_scenes,
            detected_count=len(ranked_scenes),
        )
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
            skipped=not ranked_scenes,
            artifacts=[context.database_path, artifact, cache_artifact],
            message=f"Scene ranking scored {len(ranked_scenes)} scene(s).",
        )
