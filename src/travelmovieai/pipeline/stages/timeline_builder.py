"""Pipeline stage that builds a declarative semantic montage timeline."""

from pathlib import Path

from pydantic import ValidationError

from travelmovieai.application.context import ProjectContext
from travelmovieai.core.exceptions import MontageError
from travelmovieai.domain.enums import PipelineStage
from travelmovieai.domain.models import MusicPlan, QuickMontageSettings, StageResult
from travelmovieai.editing.timeline import (
    apply_music_directing,
    build_selection_report,
    build_semantic_montage_plan,
)
from travelmovieai.infrastructure.artifacts import (
    artifact_fingerprint,
    stage_cache_manifest_matches,
    write_json_atomic,
    write_stage_cache_manifest,
)
from travelmovieai.infrastructure.database import MediaAssetRepository
from travelmovieai.pipeline.base import Stage

ARTIFACT_SCHEMA_VERSION = "timeline-builder-v2"


class TimelineBuilderStage(Stage):
    name = PipelineStage.TIMELINE_BUILDER

    def run(self, context: ProjectContext) -> StageResult:
        repository = MediaAssetRepository(context.database_path)
        repository.initialize()
        assets = repository.list_assets()
        scenes = repository.list_scenes()
        if not assets or not scenes:
            return StageResult(
                stage=self.name,
                skipped=True,
                message="Timeline builder needs media assets and ranked scenes.",
            )

        settings = QuickMontageSettings(
            semantic_analysis=True,
            story_style=context.style,
        )
        music_artifact = context.artifacts_dir / "music_plan.json"
        music_plan = _read_music_plan(music_artifact)
        timeline_artifact = context.artifacts_dir / "quick_timeline.json"
        decisions_artifact = context.artifacts_dir / "selection_decisions.json"
        cache_artifact = context.artifacts_dir / "quick_timeline.cache.json"
        input_fingerprint = artifact_fingerprint(assets, scenes, music_plan)
        config_fingerprint = artifact_fingerprint(settings, ARTIFACT_SCHEMA_VERSION)
        if stage_cache_manifest_matches(
            cache_artifact,
            stage=self.name,
            artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
            input_fingerprint=input_fingerprint,
            config_fingerprint=config_fingerprint,
            artifacts=[timeline_artifact, decisions_artifact],
        ):
            return StageResult(
                stage=self.name,
                skipped=True,
                artifacts=[timeline_artifact, decisions_artifact, cache_artifact],
                message="Timeline builder reused cached timeline artifacts.",
            )

        plan = build_semantic_montage_plan(assets, scenes, settings, music_plan)
        if music_plan is not None:
            plan = apply_music_directing(plan, scenes)
        write_json_atomic(timeline_artifact, plan)
        write_json_atomic(
            decisions_artifact,
            build_selection_report(scenes, plan, settings),
        )
        write_stage_cache_manifest(
            cache_artifact,
            stage=self.name,
            artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
            input_fingerprint=input_fingerprint,
            config_fingerprint=config_fingerprint,
            artifacts=[timeline_artifact, decisions_artifact],
        )
        return StageResult(
            stage=self.name,
            artifacts=[timeline_artifact, decisions_artifact, cache_artifact],
            message=f"Timeline builder selected {len(plan.clips)} clip(s).",
        )


def _read_music_plan(path: Path) -> MusicPlan | None:
    if not path.is_file():
        return None
    try:
        return MusicPlan.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as error:
        raise MontageError("Не удалось прочитать music_plan.json для timeline.") from error
