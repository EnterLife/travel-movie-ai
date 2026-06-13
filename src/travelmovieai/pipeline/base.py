"""Base contract implemented by every processing stage."""

from abc import ABC, abstractmethod

from travelmovieai.application.context import ProjectContext
from travelmovieai.domain.enums import PipelineStage
from travelmovieai.domain.models import StageResult


class Stage(ABC):
    name: PipelineStage

    @abstractmethod
    def run(self, context: ProjectContext) -> StageResult:
        """Run a stage and return the artifacts it produced."""
