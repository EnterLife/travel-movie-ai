"""Pipeline stage that creates a deterministic event-based storyboard."""

from travelmovieai.application.context import ProjectContext
from travelmovieai.domain.enums import PipelineStage
from travelmovieai.domain.models import StageResult
from travelmovieai.infrastructure.artifacts import write_json_atomic
from travelmovieai.infrastructure.database import MediaAssetRepository
from travelmovieai.pipeline.base import Stage
from travelmovieai.story.builder import build_storyboard


class StoryBuilderStage(Stage):
    name = PipelineStage.STORY_BUILDER

    def run(self, context: ProjectContext) -> StageResult:
        repository = MediaAssetRepository(context.database_path)
        repository.initialize()
        storyboard = build_storyboard(
            repository.list_events(),
            repository.list_scenes(),
            context.style,
        )
        artifact = context.artifacts_dir / "storyboard.json"
        write_json_atomic(artifact, storyboard)
        return StageResult(
            stage=self.name,
            skipped=not storyboard.sections,
            artifacts=[artifact],
            message=f"Story builder created {len(storyboard.sections)} section(s).",
        )
