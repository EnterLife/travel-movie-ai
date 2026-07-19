"""Sequential pipeline runner with a stable stage order."""

from collections.abc import Callable, Sequence
from dataclasses import replace

from travelmovieai.application.cache import cleanup_context_cache
from travelmovieai.application.context import ProjectContext
from travelmovieai.core.exceptions import PipelineStageError
from travelmovieai.domain.enums import PipelineStage
from travelmovieai.domain.models import StageResult
from travelmovieai.pipeline.base import Stage


class PipelineRunner:
    def __init__(self, stages: Sequence[Stage]) -> None:
        self.stages = tuple(stages)
        self._last_trace: tuple[StageResult, ...] = ()

    @property
    def last_trace(self) -> tuple[StageResult, ...]:
        """Results observed during the most recent run, excluding nested traces."""
        return self._last_trace

    def run_until(
        self,
        context: ProjectContext,
        target: PipelineStage,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> StageResult:
        self._last_trace = ()
        if target not in {stage.name for stage in self.stages}:
            raise ValueError(f"Pipeline target is not registered: {target}")

        context.prepare()
        cleanup_context_cache(context)
        target_index = next(
            index for index, stage in enumerate(self.stages) if stage.name is target
        )
        total_stages = target_index + 1
        trace: list[StageResult] = []
        for index, stage in enumerate(self.stages[:total_stages]):
            if progress is not None:
                progress(
                    index * 1000,
                    total_stages * 1000,
                    f"Starting {stage.name.value.replace('_', ' ')}",
                )

            def report_stage_progress(
                current: int,
                total: int,
                message: str,
                *,
                stage_index: int = index,
            ) -> None:
                if progress is None:
                    return
                fraction = current / total if total > 0 else 0.0
                bounded = max(0.0, min(1.0, fraction))
                progress(
                    stage_index * 1000 + round(bounded * 1000),
                    total_stages * 1000,
                    message,
                )

            stage_context = replace(
                context,
                progress=report_stage_progress if progress is not None else None,
            )
            result = stage.run(stage_context)
            if result.stage is not stage.name:
                raise PipelineStageError(
                    f"Pipeline stage {stage.name} returned a result for {result.stage}."
                )
            trace.append(result.model_copy(update={"trace": []}))
            self._last_trace = tuple(trace)
            if progress is not None:
                progress((index + 1) * 1000, total_stages * 1000, result.message)
            if stage.name == target:
                return result.model_copy(update={"trace": list(trace)})
        raise AssertionError("Validated pipeline target was not reached.")
