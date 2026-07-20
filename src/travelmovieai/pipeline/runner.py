"""Sequential pipeline runner with typed, weighted progress and run records."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import suppress
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from travelmovieai.application.cache import cleanup_context_cache
from travelmovieai.application.context import ProjectContext
from travelmovieai.application.workspace_lease import WorkspaceLease
from travelmovieai.core.exceptions import PipelineStageError
from travelmovieai.core.security import redact_sensitive_text
from travelmovieai.domain.enums import PipelineStage
from travelmovieai.domain.models import StageExecutionMetadata, StageResult
from travelmovieai.infrastructure.artifacts import write_json_atomic
from travelmovieai.pipeline.base import Stage
from travelmovieai.pipeline.progress import (
    STAGE_UNITS,
    LegacyProgressCallback,
    PipelineRunFailure,
    PipelineRunManifest,
    PipelineStageRun,
    ProgressEvent,
    ProgressEventCallback,
    StageProgress,
    stage_weight,
)

_OVERALL_PROGRESS_TOTAL = 1000


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
        progress: LegacyProgressCallback | None = None,
        progress_events: ProgressEventCallback | None = None,
    ) -> StageResult:
        self._last_trace = ()
        if target not in {stage.name for stage in self.stages}:
            raise ValueError(f"Pipeline target is not registered: {target}")

        with WorkspaceLease(context.workspace, operation=f"pipeline:{target.value}"):
            return self._run_locked(context, target, progress, progress_events)

    def _run_locked(
        self,
        context: ProjectContext,
        target: PipelineStage,
        progress: LegacyProgressCallback | None,
        progress_events: ProgressEventCallback | None,
    ) -> StageResult:
        context.prepare()
        cleanup_context_cache(context)
        target_index = next(
            index for index, stage in enumerate(self.stages) if stage.name is target
        )
        selected_stages = self.stages[: target_index + 1]
        total_weight = sum(stage_weight(stage.name) for stage in selected_stages)
        run_started_at = datetime.now(UTC)
        run_started_clock = perf_counter()
        run_path = context.artifacts_dir / "pipeline_run.json"
        manifest = PipelineRunManifest(
            run_id=uuid4(),
            target=target,
            status="running",
            started_at=run_started_at,
            stage_count=len(selected_stages),
            total_weight=total_weight,
        )
        write_json_atomic(run_path, manifest)

        completed_weight = 0.0
        last_overall_current = 0
        trace: list[StageResult] = []
        active_stage: Stage | None = None
        active_started_at: datetime | None = None
        active_started_clock: float | None = None

        def emit(
            stage: PipelineStage,
            stage_progress: StageProgress,
            *,
            completed: float,
            weight: float,
        ) -> None:
            nonlocal last_overall_current
            mapped = round(
                (completed + weight * stage_progress.fraction)
                / total_weight
                * _OVERALL_PROGRESS_TOTAL
            )
            overall_current = max(last_overall_current, min(_OVERALL_PROGRESS_TOTAL, mapped))
            last_overall_current = overall_current
            event = ProgressEvent(
                stage=stage,
                current=stage_progress.current,
                total=stage_progress.total,
                unit=stage_progress.unit,
                message=stage_progress.message,
                overall_current=overall_current,
                overall_total=_OVERALL_PROGRESS_TOTAL,
            )
            if progress_events is not None:
                progress_events(event)
            if progress is not None:
                progress(event.overall_current, event.overall_total, event.message)

        try:
            for stage in selected_stages:
                weight = stage_weight(stage.name)
                unit = STAGE_UNITS.get(stage.name, "items")
                active_stage = stage
                active_started_at = datetime.now(UTC)
                active_started_clock = perf_counter()
                emit(
                    stage.name,
                    StageProgress(
                        current=0,
                        total=1,
                        unit=unit,
                        message=f"Starting {stage.name.value.replace('_', ' ')}",
                    ),
                    completed=completed_weight,
                    weight=weight,
                )

                def report_stage_progress(
                    current: int,
                    total: int,
                    message: str,
                    *,
                    current_stage: PipelineStage = stage.name,
                    current_unit: str = unit,
                    base_weight: float = completed_weight,
                    current_weight: float = weight,
                ) -> None:
                    emit(
                        current_stage,
                        StageProgress(
                            current=max(0, current),
                            total=max(0, total),
                            unit=current_unit,
                            message=message,
                        ),
                        completed=base_weight,
                        weight=current_weight,
                    )

                stage_context = replace(
                    context,
                    progress=(
                        report_stage_progress
                        if progress is not None or progress_events is not None
                        else None
                    ),
                )
                result = stage.run(stage_context)
                if result.stage is not stage.name:
                    raise PipelineStageError(
                        f"Pipeline stage {stage.name} returned a result for {result.stage}."
                    )
                finished_at = datetime.now(UTC)
                duration_seconds = max(0.0, perf_counter() - active_started_clock)
                manifest.stages.append(
                    PipelineStageRun(
                        stage=stage.name,
                        weight=weight,
                        started_at=active_started_at,
                        finished_at=finished_at,
                        duration_seconds=duration_seconds,
                        status=result.status,
                        cache_hit=result.cache_hit,
                        artifact_count=len(result.artifacts),
                        execution=_safe_execution_metadata(result.execution),
                    )
                )
                manifest.completed_stage_count = len(manifest.stages)
                write_json_atomic(run_path, manifest)
                trace.append(result.model_copy(update={"trace": []}))
                self._last_trace = tuple(trace)
                active_stage = None
                active_started_at = None
                active_started_clock = None
                completed_weight += weight
                emit(
                    stage.name,
                    StageProgress(current=1, total=1, unit=unit, message=result.message),
                    completed=completed_weight - weight,
                    weight=weight,
                )
                if (
                    stage.name is PipelineStage.VISION_ANALYSIS
                    and context.settings.vision_model_pool_size == 0
                ):
                    _clear_idle_vision_models()
                if stage.name == target:
                    manifest.status = "completed"
                    manifest.finished_at = datetime.now(UTC)
                    manifest.duration_seconds = max(0.0, perf_counter() - run_started_clock)
                    write_json_atomic(run_path, manifest)
                    return result.model_copy(update={"trace": list(trace)})
        except BaseException as error:
            if (
                active_stage is not None
                and active_started_at is not None
                and active_started_clock is not None
            ):
                manifest.stages.append(
                    PipelineStageRun(
                        stage=active_stage.name,
                        weight=stage_weight(active_stage.name),
                        started_at=active_started_at,
                        finished_at=datetime.now(UTC),
                        duration_seconds=max(0.0, perf_counter() - active_started_clock),
                        status="failed",
                    )
                )
            manifest.completed_stage_count = len(trace)
            manifest.status = "failed"
            manifest.finished_at = datetime.now(UTC)
            manifest.duration_seconds = max(0.0, perf_counter() - run_started_clock)
            private_paths = _private_path_variants(context)
            message = redact_sensitive_text(str(error), private_paths=private_paths).strip()
            manifest.failure = PipelineRunFailure(
                error_type=type(error).__name__,
                message=message or "Pipeline execution failed.",
            )
            with suppress(OSError):
                write_json_atomic(run_path, manifest)
            raise
        raise AssertionError("Validated pipeline target was not reached.")


def _safe_execution_metadata(metadata: StageExecutionMetadata) -> StageExecutionMetadata:
    model = metadata.model
    if model and (Path(model).is_absolute() or "\\" in model):
        model = "<local-model>"
    return metadata.model_copy(update={"model": model})


def _private_path_variants(context: ProjectContext) -> list[Path]:
    candidates: list[Path | None] = [
        context.input_path,
        context.workspace,
        context.output_path,
        context.settings.model_cache,
        context.settings.music_library,
        context.settings.piper_model,
        _absolute_path(context.settings.ffmpeg_binary),
        _absolute_path(context.settings.ffprobe_binary),
        _absolute_path(context.settings.piper_binary),
        _absolute_path(context.settings.vision_model),
        _absolute_path(context.settings.embedding_model),
        _absolute_path(context.settings.story_model),
        _absolute_path(context.settings.music_model),
    ]
    if context.montage_settings is not None:
        candidates.extend(
            [
                context.montage_settings.music_path,
                context.montage_settings.overlay_font_path,
                _absolute_path(context.montage_settings.vision_model),
                _absolute_path(context.montage_settings.music_model),
            ]
        )

    variants: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate is None:
            continue
        expanded = candidate.expanduser()
        resolved = _resolve_path(expanded)
        for variant in (resolved, expanded):
            key = str(variant).casefold()
            if key not in seen:
                seen.add(key)
                variants.append(variant)
    return variants


def _absolute_path(value: str | None) -> Path | None:
    if not value:
        return None
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else None


def _resolve_path(path: Path) -> Path:
    try:
        return path.resolve()
    except (OSError, RuntimeError):
        return path.absolute()


def _clear_idle_vision_models() -> None:
    from travelmovieai.infrastructure.vision import clear_idle_vision_models

    clear_idle_vision_models()
