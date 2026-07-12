"""Pipeline stage for semantic event clustering."""

from pathlib import Path

from travelmovieai.application.context import ProjectContext
from travelmovieai.domain.enums import PipelineStage
from travelmovieai.domain.models import Event, EventDetectionReport, Scene, StageResult
from travelmovieai.infrastructure.artifacts import (
    artifact_fingerprint,
    stage_cache_manifest_matches,
    write_json_atomic,
    write_stage_cache_manifest,
)
from travelmovieai.infrastructure.database import MediaAssetRepository
from travelmovieai.pipeline.base import Stage
from travelmovieai.story.events import detect_events

ARTIFACT_SCHEMA_VERSION = "event-detection-v1"


class EventDetectionStage(Stage):
    name = PipelineStage.EVENT_DETECTION

    def run(self, context: ProjectContext) -> StageResult:
        repository = MediaAssetRepository(context.database_path)
        repository.initialize()
        scenes = repository.list_scenes()
        assets = repository.list_assets()
        artifact = context.artifacts_dir / "events.json"
        cache_artifact = context.artifacts_dir / "events.cache.json"
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
                }
                for scene in scenes
            ],
            [
                {
                    "id": str(asset.id),
                    "created_at": asset.created_at,
                    "modified_at": asset.modified_at,
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
            return StageResult(
                stage=self.name,
                skipped=True,
                artifacts=[context.database_path, artifact, cache_artifact],
                message="Event detection reused cached event groups.",
            )
        report, scenes = detect_events(scenes, assets)
        repository.synchronize_scenes(scenes)
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
            skipped=not report.events,
            artifacts=[context.database_path, artifact, cache_artifact],
            message=f"Event detection produced {len(report.events)} event(s).",
        )


def _cached_events_valid(path: Path, scenes: list[Scene], events: list[Event]) -> bool:
    if not events:
        return False
    try:
        EventDetectionReport.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return all(scene.metadata.get("event_id") for scene in scenes)
