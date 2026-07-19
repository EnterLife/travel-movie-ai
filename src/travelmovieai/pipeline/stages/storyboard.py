"""Pipeline stage that creates a deterministic event-based storyboard."""

from pathlib import Path

from travelmovieai.application.context import ProjectContext
from travelmovieai.core.exceptions import DependencyUnavailableError, StoryGenerationError
from travelmovieai.domain.enums import PipelineStage, StageStatus
from travelmovieai.domain.models import Scene, StageResult, Storyboard
from travelmovieai.infrastructure.artifacts import (
    artifact_fingerprint,
    stage_cache_manifest_matches,
    write_json_atomic,
    write_stage_cache_manifest,
)
from travelmovieai.infrastructure.database import MediaAssetRepository
from travelmovieai.infrastructure.story import StoryProvider, build_story_provider
from travelmovieai.pipeline.base import Stage
from travelmovieai.story.builder import build_storyboard
from travelmovieai.story.optimizer import apply_story_structure

ARTIFACT_SCHEMA_VERSION = "story-builder-v2"


class StoryBuilderStage(Stage):
    name = PipelineStage.STORY_BUILDER

    def run(self, context: ProjectContext) -> StageResult:
        repository = MediaAssetRepository(context.database_path)
        repository.initialize()
        events = repository.list_events()
        scenes = repository.list_scenes()
        artifact = context.artifacts_dir / "storyboard.json"
        cache_artifact = context.artifacts_dir / "storyboard.cache.json"
        provider_name = context.settings.story_provider
        input_fingerprint = artifact_fingerprint(
            events,
            _story_scene_inputs(scenes, include_multimodal=provider_name == "local"),
        )
        provider_config: dict[str, object] = {"provider": provider_name}
        if provider_name == "local":
            provider_config.update(
                {
                    "model": context.settings.story_model,
                    "device": context.settings.device,
                    "max_new_tokens": context.settings.story_max_new_tokens,
                }
            )
        config_fingerprint = artifact_fingerprint(
            context.style,
            provider_config,
            ARTIFACT_SCHEMA_VERSION,
        )
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
                status=StageStatus.CACHED,
                artifacts=[context.database_path, artifact, cache_artifact],
                message="Story builder reused cached story structure.",
            )
        fallback_used = False
        if provider_name == "local" and events:
            if context.progress is not None:
                context.progress(0, 1, "Story model: generating structured local story")
            provider: StoryProvider = build_story_provider(
                model=context.settings.story_model,
                device=context.settings.device,
                cache_dir=(context.settings.model_cache / "story").expanduser().resolve(),
                allow_download=context.settings.allow_model_download,
                max_new_tokens=context.settings.story_max_new_tokens,
            )
            try:
                storyboard = provider.build(events, scenes, context.style)
            except (DependencyUnavailableError, StoryGenerationError):
                fallback_used = True
                storyboard = build_storyboard(events, scenes, context.style).model_copy(
                    update={
                        "provider": provider.name,
                        "model": provider.model,
                        "prompt_version": None,
                        "fallback_used": True,
                    }
                )
            finally:
                provider.release()
            if context.progress is not None:
                context.progress(1, 1, "Story model: complete")
        else:
            storyboard = build_storyboard(events, scenes, context.style)
        scenes = apply_story_structure(scenes, storyboard)
        repository.synchronize_scenes(scenes)
        write_json_atomic(artifact, storyboard)
        if fallback_used:
            cache_artifact.unlink(missing_ok=True)
        else:
            write_stage_cache_manifest(
                cache_artifact,
                stage=self.name,
                artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
                input_fingerprint=input_fingerprint,
                config_fingerprint=config_fingerprint,
                artifacts=[artifact],
            )
        artifacts = [context.database_path, artifact]
        if not fallback_used:
            artifacts.append(cache_artifact)
        return StageResult(
            stage=self.name,
            status=StageStatus.COMPLETED if storyboard.sections else StageStatus.NO_INPUT,
            artifacts=artifacts,
            message=(
                f"Story builder created {len(storyboard.sections)} section(s)"
                + (
                    " with the deterministic fallback; the local model will be retried."
                    if fallback_used
                    else f" using {storyboard.provider}."
                )
            ),
        )


def _cached_storyboard_valid(path: Path, scenes: list[Scene]) -> bool:
    try:
        storyboard = Storyboard.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if storyboard.fallback_used:
        return False
    if not storyboard.sections:
        return not scenes
    return all(scene.metadata.get("story_section_role") for scene in scenes)


def _story_scene_inputs(
    scenes: list[Scene],
    *,
    include_multimodal: bool,
) -> list[dict[str, object]]:
    inputs: list[dict[str, object]] = []
    for scene in scenes:
        item: dict[str, object] = {
            "id": str(scene.id),
            "event_id": scene.metadata.get("event_id"),
        }
        if include_multimodal:
            item.update(
                {
                    "caption": scene.caption,
                    "transcript": scene.transcript,
                    "detailed_description": scene.metadata.get("detailed_description"),
                    "emotion": scene.metadata.get("emotion"),
                    "shot_scale": scene.metadata.get("shot_scale"),
                    "camera_motion": scene.metadata.get("camera_motion"),
                }
            )
        inputs.append(item)
    return inputs
