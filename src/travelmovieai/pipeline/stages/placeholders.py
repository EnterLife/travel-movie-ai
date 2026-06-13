"""Temporary stages that reserve the MVP pipeline boundaries."""

from travelmovieai.application.context import ProjectContext
from travelmovieai.domain.enums import PipelineStage
from travelmovieai.domain.models import StageResult
from travelmovieai.pipeline.base import Stage


class PlaceholderStage(Stage):
    def __init__(self, name: PipelineStage) -> None:
        self.name = name

    def run(self, context: ProjectContext) -> StageResult:
        return StageResult(
            stage=self.name,
            skipped=True,
            message=(
                f"Pipeline scaffold reached '{self.name.value}'. "
                "The processing implementation is not part of this scaffold."
            ),
        )
