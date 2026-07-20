"""Pipeline stage that assembles multimodal scene descriptions."""

from pathlib import Path

from travelmovieai.application.context import ProjectContext
from travelmovieai.domain.enums import PipelineStage, StageStatus
from travelmovieai.domain.models import MultimodalDescriptionReport, StageResult
from travelmovieai.infrastructure.artifacts import (
    artifact_fingerprint,
    stage_cache_manifest_matches,
    write_json_atomic,
    write_stage_cache_manifest,
)
from travelmovieai.infrastructure.database import MediaAssetRepository
from travelmovieai.pipeline.base import Stage
from travelmovieai.story.builder import build_multimodal_descriptions

ARTIFACT_SCHEMA_VERSION = "scene-captioning-v2"


class SceneCaptioningStage(Stage):
    name = PipelineStage.SCENE_CAPTIONING

    def run(self, context: ProjectContext) -> StageResult:
        repository = MediaAssetRepository(context.database_path)
        repository.initialize()
        scenes = repository.list_scenes()
        artifact = context.artifacts_dir / "scene_descriptions.json"
        cache_artifact = context.artifacts_dir / "scene_descriptions.cache.json"
        input_fingerprint = artifact_fingerprint(
            [
                {
                    "id": str(scene.id),
                    "caption": scene.caption,
                    "description": scene.metadata.get("detailed_description"),
                    "transcript": scene.transcript,
                    "quality_score": scene.quality_score,
                    "audio_context": scene.metadata.get("audio_context"),
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
        ) and _cached_descriptions_valid(artifact):
            return StageResult(
                stage=self.name,
                status=StageStatus.CACHED,
                artifacts=[artifact, cache_artifact],
                message="Story builder reused cached multimodal descriptions.",
            )
        report = build_multimodal_descriptions(scenes)
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
            status=StageStatus.COMPLETED if report.descriptions else StageStatus.NO_INPUT,
            artifacts=[artifact, cache_artifact],
            message=(
                f"Story builder prepared {len(report.descriptions)} "
                "multimodal scene description(s)."
            ),
        )


def _cached_descriptions_valid(path: Path) -> bool:
    try:
        MultimodalDescriptionReport.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return True
