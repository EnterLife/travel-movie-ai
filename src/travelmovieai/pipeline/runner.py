"""Sequential pipeline runner with a stable stage order."""

from collections.abc import Sequence

from travelmovieai.application.context import ProjectContext
from travelmovieai.domain.enums import PipelineStage
from travelmovieai.domain.models import StageResult
from travelmovieai.pipeline.base import Stage


class PipelineRunner:
    def __init__(self, stages: Sequence[Stage]) -> None:
        self.stages = stages

    def run_until(self, context: ProjectContext, target: PipelineStage) -> StageResult:
        context.prepare()
        for stage in self.stages:
            result = stage.run(context)
            if stage.name == target:
                return result
        raise ValueError(f"Pipeline target is not registered: {target}")
