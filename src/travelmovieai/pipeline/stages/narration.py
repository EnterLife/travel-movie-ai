"""Pipeline stage that creates a local text narration artifact."""

from pathlib import Path

from pydantic import ValidationError

from travelmovieai.application.context import ProjectContext
from travelmovieai.core.exceptions import PipelineStageError
from travelmovieai.domain.enums import PipelineStage, StageStatus
from travelmovieai.domain.models import NarrationReport, StageResult, Storyboard
from travelmovieai.infrastructure.artifacts import (
    artifact_fingerprint,
    stage_cache_manifest_matches,
    write_json_atomic,
    write_stage_cache_manifest,
)
from travelmovieai.infrastructure.database import MediaAssetRepository
from travelmovieai.pipeline.base import Stage
from travelmovieai.story.narration import build_narration

ARTIFACT_SCHEMA_VERSION = "narration-v1"


class NarrationStage(Stage):
    name = PipelineStage.NARRATION

    def run(self, context: ProjectContext) -> StageResult:
        storyboard_path = context.artifacts_dir / "storyboard.json"
        if not storyboard_path.is_file():
            return StageResult(
                stage=self.name,
                status=StageStatus.NO_INPUT,
                message="Narration needs storyboard.json.",
            )
        storyboard = _read_storyboard(storyboard_path)
        repository = MediaAssetRepository(context.database_path)
        repository.initialize()
        events = repository.list_events()
        artifact = context.artifacts_dir / "narration.json"
        cache_artifact = context.artifacts_dir / "narration.cache.json"
        input_fingerprint = artifact_fingerprint(
            storyboard.model_copy(update={"narration": []}),
            events,
        )
        config_fingerprint = artifact_fingerprint(ARTIFACT_SCHEMA_VERSION)
        cached_report = _read_cached_narration(artifact)
        if (
            stage_cache_manifest_matches(
                cache_artifact,
                stage=self.name,
                artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
                input_fingerprint=input_fingerprint,
                config_fingerprint=config_fingerprint,
                artifacts=[artifact],
            )
            and cached_report is not None
        ):
            narration = [line.text for line in cached_report.lines]
            if storyboard.narration != narration:
                write_json_atomic(
                    storyboard_path,
                    storyboard.model_copy(update={"narration": narration}),
                )
            return StageResult(
                stage=self.name,
                status=StageStatus.CACHED,
                artifacts=[storyboard_path, artifact, cache_artifact],
                message="Narration reused cached story text.",
            )
        report = build_narration(storyboard, events)
        write_json_atomic(artifact, report)
        write_json_atomic(
            storyboard_path,
            storyboard.model_copy(update={"narration": [line.text for line in report.lines]}),
        )
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
            status=StageStatus.COMPLETED if report.lines else StageStatus.NO_INPUT,
            artifacts=[storyboard_path, artifact, cache_artifact],
            message=f"Narration prepared {len(report.lines)} story line(s).",
        )


def _read_storyboard(path: Path) -> Storyboard:
    try:
        return Storyboard.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as error:
        raise PipelineStageError("Could not read storyboard.json for narration.") from error


def _read_cached_narration(path: Path) -> NarrationReport | None:
    try:
        return NarrationReport.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError):
        return None
