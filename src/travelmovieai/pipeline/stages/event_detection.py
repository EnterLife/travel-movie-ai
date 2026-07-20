"""Pipeline stage for semantic event clustering."""

import math
from pathlib import Path
from uuid import UUID

from travelmovieai.application.context import ProjectContext
from travelmovieai.domain.enums import PipelineStage, StageStatus
from travelmovieai.domain.models import (
    EmbeddingAnalysisReport,
    Event,
    EventDetectionReport,
    Scene,
    StageResult,
)
from travelmovieai.infrastructure.artifacts import (
    artifact_fingerprint,
    stage_cache_manifest_matches,
    write_json_atomic,
    write_stage_cache_manifest,
)
from travelmovieai.infrastructure.database import MediaAssetRepository
from travelmovieai.pipeline.base import Stage
from travelmovieai.story.events import detect_events

ARTIFACT_SCHEMA_VERSION = "event-detection-v4-artifact-embeddings"


class EventDetectionStage(Stage):
    name = PipelineStage.EVENT_DETECTION

    def run(self, context: ProjectContext) -> StageResult:
        repository = MediaAssetRepository(context.database_path)
        repository.initialize()
        scenes = repository.list_scenes()
        assets = repository.list_assets()
        artifact = context.artifacts_dir / "events.json"
        cache_artifact = context.artifacts_dir / "events.cache.json"
        embedding_vectors = _load_embedding_vectors(
            context.artifacts_dir / "embeddings.json",
            scenes,
        )
        input_fingerprint = artifact_fingerprint(
            [
                {
                    "id": str(scene.id),
                    "asset_id": str(scene.asset_id),
                    "start": scene.start_seconds,
                    "end": scene.end_seconds,
                    "caption": scene.caption,
                    "importance": scene.importance_score,
                    "location": scene.metadata.get("location_type"),
                    "activity": scene.metadata.get("activity"),
                    "landmarks": scene.metadata.get("landmarks"),
                    "semantic_embedding": (
                        embedding_vectors.get(scene.id)
                        if embedding_vectors is not None
                        else scene.metadata.get("semantic_embedding")
                    ),
                    "embedding_backend": scene.metadata.get("embedding_backend"),
                    "embedding_model": scene.metadata.get("embedding_model"),
                }
                for scene in scenes
            ],
            [
                {
                    "id": str(asset.id),
                    "created_at": asset.created_at,
                    "modified_at": asset.modified_at,
                    "latitude": asset.latitude,
                    "longitude": asset.longitude,
                }
                for asset in assets
            ],
        )
        config_fingerprint = artifact_fingerprint(ARTIFACT_SCHEMA_VERSION)
        if stage_cache_manifest_matches(
            cache_artifact,
            stage=self.name,
            artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
            input_fingerprint=input_fingerprint,
            config_fingerprint=config_fingerprint,
            artifacts=[artifact],
        ) and _cached_events_valid(artifact, scenes, repository.list_events()):
            if embedding_vectors is not None:
                cleaned_scenes = _without_persisted_vectors(scenes)
                if cleaned_scenes != scenes:
                    repository.synchronize_scenes(cleaned_scenes)
            return StageResult(
                stage=self.name,
                status=StageStatus.CACHED,
                artifacts=[context.database_path, artifact, cache_artifact],
                message="Event detection reused cached event groups.",
            )
        analysis_scenes = (
            _with_semantic_embeddings(scenes, embedding_vectors)
            if embedding_vectors is not None
            else scenes
        )
        report, updated_scenes = detect_events(analysis_scenes, assets)
        repository.synchronize_scenes(
            _without_persisted_vectors(updated_scenes)
            if embedding_vectors is not None
            else updated_scenes
        )
        repository.synchronize_events(report.events)
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
            status=StageStatus.COMPLETED if report.events else StageStatus.NO_INPUT,
            artifacts=[context.database_path, artifact, cache_artifact],
            message=f"Event detection produced {len(report.events)} event(s).",
        )


def _cached_events_valid(path: Path, scenes: list[Scene], events: list[Event]) -> bool:
    try:
        report = EventDetectionReport.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not scenes:
        return not report.events and not events
    reported_by_id = {event.id: event for event in report.events}
    if set(reported_by_id) != {event.id for event in events}:
        return False
    expected_scene_ids = {scene.id for scene in scenes}
    if {scene_id for event in report.events for scene_id in event.scene_ids} != expected_scene_ids:
        return False
    reported_by_text_id = {str(event.id): event for event in report.events}
    for scene in scenes:
        event_id = scene.metadata.get("event_id")
        if not isinstance(event_id, str):
            return False
        event = reported_by_text_id.get(event_id)
        if event is None or scene.id not in event.scene_ids:
            return False
    return True


def _load_embedding_vectors(
    path: Path,
    scenes: list[Scene],
) -> dict[UUID, list[float]] | None:
    try:
        report = EmbeddingAnalysisReport.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if len(report.embeddings) != len(scenes) or any(
        len(embedding.vector) != report.dimensions
        or any(not math.isfinite(value) for value in embedding.vector)
        for embedding in report.embeddings
    ):
        return None
    vectors = {embedding.scene_id: embedding.vector for embedding in report.embeddings}
    if set(vectors) != {scene.id for scene in scenes}:
        return None
    return vectors


def _with_semantic_embeddings(
    scenes: list[Scene],
    vectors: dict[UUID, list[float]],
) -> list[Scene]:
    return [
        scene.model_copy(
            update={
                "metadata": {
                    **scene.metadata,
                    "semantic_embedding": vectors[scene.id],
                }
            }
        )
        for scene in scenes
    ]


def _without_persisted_vectors(scenes: list[Scene]) -> list[Scene]:
    cleaned: list[Scene] = []
    for scene in scenes:
        metadata = dict(scene.metadata)
        metadata.pop("semantic_embedding", None)
        cleaned.append(scene.model_copy(update={"metadata": metadata}))
    return cleaned
