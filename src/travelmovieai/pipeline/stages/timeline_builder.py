"""Pipeline stage that builds a declarative semantic montage timeline."""

from travelmovieai.application.context import ProjectContext
from travelmovieai.domain.enums import PipelineStage
from travelmovieai.domain.models import QuickMontageSettings, StageResult
from travelmovieai.editing.timeline import (
    build_selection_report,
    build_semantic_montage_plan,
)
from travelmovieai.infrastructure.artifacts import write_json_atomic
from travelmovieai.infrastructure.database import MediaAssetRepository
from travelmovieai.pipeline.base import Stage


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
        plan = build_semantic_montage_plan(assets, scenes, settings)
        timeline_artifact = context.artifacts_dir / "quick_timeline.json"
        decisions_artifact = context.artifacts_dir / "selection_decisions.json"
        write_json_atomic(timeline_artifact, plan)
        write_json_atomic(
            decisions_artifact,
            build_selection_report(scenes, plan, settings),
        )
        return StageResult(
            stage=self.name,
            artifacts=[timeline_artifact, decisions_artifact],
            message=f"Timeline builder selected {len(plan.clips)} clip(s).",
        )
