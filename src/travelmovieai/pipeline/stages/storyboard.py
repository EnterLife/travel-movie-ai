"""Pipeline stage that creates a deterministic event-based storyboard."""

from pathlib import Path

from travelmovieai.application.context import ProjectContext
from travelmovieai.domain.enums import PipelineStage
from travelmovieai.domain.models import Scene, StageResult, Storyboard
from travelmovieai.infrastructure.artifacts import (
    artifact_fingerprint,
    stage_cache_manifest_matches,
    write_json_atomic,
    write_stage_cache_manifest,
)
from travelmovieai.infrastructure.database import MediaAssetRepository
from travelmovieai.pipeline.base import Stage
from travelmovieai.story.builder import build_storyboard
from travelmovieai.story.optimizer import apply_story_structure

ARTIFACT_SCHEMA_VERSION = "story-builder-v1"


class StoryBuilderStage(Stage):
    name = PipelineStage.STORY_BUILDER

    def run(self, context: ProjectContext) -> StageResult:
        repository = MediaAssetRepository(context.database_path)
        repository.initialize()
        events = repository.list_events()
        scenes = repository.list_scenes()
        artifact = context.artifacts_dir / "storyboard.json"
        cache_artifact = context.artifacts_dir / "storyboard.cache.json"
        input_fingerprint = artifact_fingerprint(
            events,
            [{"id": str(scene.id), "event_id": scene.metadata.get("event_id")} for scene in scenes],
        )
        config_fingerprint = artifact_fingerprint(context.style, ARTIFACT_SCHEMA_VERSION)
        if stage_cache_manifest_matches(
            cache_artifact,
            stage=self.name,
            artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
            input_fingerprint=input_fingerprint,
            config_fingerprint=config_fingerprint,
            artifacts=[artifact],
        ) and _cached_storyboard_valid(artifact, scenes):
            return StageResult(
                stage=self.name,
                skipped=True,
                artifacts=[context.database_path, artifact, cache_artifact],
                message="Story builder reused cached story structure.",
            )
        storyboard = build_storyboard(events, scenes, context.style)
        scenes = apply_story_structure(scenes, storyboard)
        repository.synchronize_scenes(scenes)
        write_json_atomic(artifact, storyboard)
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
            skipped=not storyboard.sections,
            artifacts=[context.database_path, artifact, cache_artifact],
            message=f"Story builder created {len(storyboard.sections)} section(s).",
        )


def _cached_storyboard_valid(path: Path, scenes: list[Scene]) -> bool:
    try:
        storyboard = Storyboard.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not storyboard.sections:
        return not scenes
    return all(scene.metadata.get("story_section_role") for scene in scenes)
