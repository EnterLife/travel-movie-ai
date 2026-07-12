"""Pipeline stage for deterministic local semantic scene embeddings."""

from pathlib import Path

from travelmovieai.analysis.embeddings import EMBEDDING_BACKEND, embed_scenes
from travelmovieai.application.context import ProjectContext
from travelmovieai.domain.enums import PipelineStage
from travelmovieai.domain.models import EmbeddingAnalysisReport, Scene, StageResult
from travelmovieai.infrastructure.artifacts import (
    artifact_fingerprint,
    stage_cache_manifest_matches,
    write_json_atomic,
    write_stage_cache_manifest,
)
from travelmovieai.infrastructure.database import MediaAssetRepository
from travelmovieai.pipeline.base import Stage

ARTIFACT_SCHEMA_VERSION = "embeddings-v1"


class EmbeddingsStage(Stage):
    name = PipelineStage.EMBEDDINGS

    def run(self, context: ProjectContext) -> StageResult:
        repository = MediaAssetRepository(context.database_path)
        repository.initialize()
        scenes = repository.list_scenes()
        artifact = context.artifacts_dir / "embeddings.json"
        cache_artifact = context.artifacts_dir / "embeddings.cache.json"
        input_fingerprint = artifact_fingerprint(_embedding_inputs(scenes))
        config_fingerprint = artifact_fingerprint(
            {"backend": EMBEDDING_BACKEND, "schema": ARTIFACT_SCHEMA_VERSION}
        )
        if stage_cache_manifest_matches(
            cache_artifact,
            stage=self.name,
            artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
            input_fingerprint=input_fingerprint,
            config_fingerprint=config_fingerprint,
            artifacts=[artifact],
        ) and _cached_embeddings_valid(artifact, scenes):
            return StageResult(
                stage=self.name,
                skipped=True,
                artifacts=[context.database_path, artifact, cache_artifact],
                message="Embeddings reused cached semantic features.",
            )

        report, updated = embed_scenes(scenes)
        repository.synchronize_scenes(updated)
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
            skipped=not report.embeddings,
            artifacts=[context.database_path, artifact, cache_artifact],
            message=f"Embeddings prepared {len(report.embeddings)} semantic vector(s).",
        )


def _cached_embeddings_valid(path: Path, scenes: list[Scene]) -> bool:
    try:
        report = EmbeddingAnalysisReport.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return {item.scene_id for item in report.embeddings} == {scene.id for scene in scenes} and all(
        scene.metadata.get("embedding_backend") == EMBEDDING_BACKEND for scene in scenes
    )


def _embedding_inputs(scenes: list[Scene]) -> list[dict[str, object]]:
    return [
        {
            "id": str(scene.id),
            "caption": scene.caption,
            "transcript": scene.transcript,
            "description": scene.metadata.get("detailed_description"),
            "location_type": scene.metadata.get("location_type"),
            "activity": scene.metadata.get("activity"),
            "emotion": scene.metadata.get("emotion"),
            "tags": scene.metadata.get("tags"),
        }
        for scene in sorted(scenes, key=lambda item: str(item.id))
    ]
