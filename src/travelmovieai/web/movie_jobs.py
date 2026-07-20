"""Background quick montage jobs."""

import inspect
import logging
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from threading import Condition, RLock
from time import monotonic
from typing import Any, Protocol
from uuid import UUID, uuid4

from pydantic import ValidationError

from travelmovieai.application.disk_space import ensure_render_disk_space
from travelmovieai.application.validation import ProjectPaths
from travelmovieai.application.variants import (
    safe_variant_slug,
    validate_variant_name,
    variant_output_path,
)
from travelmovieai.core.exceptions import (
    InvalidProjectPathError,
    JobPersistenceError,
    TravelMovieError,
    WorkspaceBusyError,
)
from travelmovieai.core.logging import correlation_context, register_private_log_paths
from travelmovieai.domain.enums import PipelineStage
from travelmovieai.domain.models import QuickMontageResult, QuickMontageSettings
from travelmovieai.infrastructure.artifacts import write_json_atomic
from travelmovieai.infrastructure.system import ResourceProfile
from travelmovieai.pipeline.progress import ProgressEvent, weighted_stage_ranges
from travelmovieai.web.schemas import (
    JobLogEntry,
    JobStatus,
    JobSubtaskProgress,
    MovieJobResponse,
    MovieJobState,
    MovieJobStateHistory,
    ResourceProfileResponse,
)
from travelmovieai.web.state import redact_sensitive_text

LOGGER = logging.getLogger(__name__)

_SUBTASK_LABELS = {
    PipelineStage.MEDIA_SCAN.value: "Media Scan",
    PipelineStage.SCENE_DETECTION.value: "Scene Detection",
    PipelineStage.FRAME_SAMPLING.value: "Frame Extraction",
    PipelineStage.QUALITY_ANALYSIS.value: "OpenCV Analysis",
    PipelineStage.VISION_ANALYSIS.value: "Vision AI",
    PipelineStage.SPEECH_ANALYSIS.value: "Speech Recognition",
    PipelineStage.AUDIO_ANALYSIS.value: "Audio Analysis",
    PipelineStage.EMBEDDINGS.value: "Embeddings",
    PipelineStage.DUPLICATE_DETECTION.value: "Duplicate Detection",
    PipelineStage.SCENE_CAPTIONING.value: "Scene Captioning",
    PipelineStage.EVENT_DETECTION.value: "Event Detection",
    PipelineStage.STORY_BUILDER.value: "Story Builder",
    PipelineStage.SCENE_RANKING.value: "Scene Ranking",
    PipelineStage.MUSIC_SELECTION.value: "Music",
    PipelineStage.NARRATION.value: "Narration",
    PipelineStage.VOICE_SYNTHESIS.value: "Voice Synthesis",
    PipelineStage.TIMELINE_BUILDER.value: "Timeline",
    PipelineStage.RENDERING.value: "Rendering and Validation",
}
_SEMANTIC_SUBTASK_RANGES = {
    stage.value: (
        1.0 + bounds[0] * 0.99,
        1.0 + bounds[1] * 0.99,
        _SUBTASK_LABELS[stage.value],
    )
    for stage, bounds in weighted_stage_ranges().items()
}
_QUICK_SUBTASK_RANGES = {
    stage.value: (0.0, 0.0, _SUBTASK_LABELS[stage.value]) for stage in PipelineStage
}
_QUICK_SUBTASK_RANGES.update(
    {
        PipelineStage.MEDIA_SCAN.value: (0.0, 5.0, _SUBTASK_LABELS["media_scan"]),
        PipelineStage.SCENE_RANKING.value: (5.0, 78.0, "Chronological Clip Selection"),
        PipelineStage.MUSIC_SELECTION.value: (78.0, 80.0, _SUBTASK_LABELS["music_selection"]),
        PipelineStage.TIMELINE_BUILDER.value: (
            80.0,
            85.0,
            _SUBTASK_LABELS["timeline_builder"],
        ),
        PipelineStage.RENDERING.value: (85.0, 100.0, _SUBTASK_LABELS["rendering"]),
    }
)


class _MovieCancelled(Exception):
    pass


class _MovieInterrupted(Exception):
    """Stop this worker while keeping its persisted job recoverable."""


class MovieService(Protocol):
    def resolve_project_paths(self, input_path: Path, workspace: Path | None) -> ProjectPaths: ...

    def create_quick_montage(
        self,
        *,
        input_path: Path,
        workspace: Path | None,
        settings: QuickMontageSettings,
        variant_name: str = "Default",
        output_path: Path | None = None,
        progress: Callable[[int, int, str], None] | None = None,
        progress_events: Callable[[ProgressEvent], None] | None = None,
    ) -> QuickMontageResult: ...


@dataclass(slots=True)
class _MovieJob:
    id: UUID
    status: JobStatus
    input_path: Path
    workspace: Path
    settings: QuickMontageSettings
    variant_name: str
    variant_slug: str
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    phase_started_at: datetime | None = None
    phase_last_progress_at: datetime | None = None
    phase_last_progress_percent: float | None = None
    message: str = ""
    error: str | None = None
    progress_current: int = 0
    progress_total: int = 0
    phase: str = "queued"
    pipeline_stage: PipelineStage | None = None
    resources: ResourceProfile | None = None
    subtasks: list[JobSubtaskProgress] = field(default_factory=list)
    logs: list[JobLogEntry] = field(default_factory=list)
    result: QuickMontageResult | None = None
    pause_requested: bool = False
    cancel_requested: bool = False
    worker_finished: bool = False
    paused_at: datetime | None = None
    paused_seconds: float = 0
    persistence_degraded: bool = False


class MovieJobManager:
    def __init__(
        self,
        service: MovieService,
        *,
        state_path: Path | None = None,
        history_limit: int = 100,
        render_disk_reserve_mb: int = 1024,
        render_disk_safety_factor: float = 3.0,
    ) -> None:
        self._service = service
        self._state_path = state_path
        self._history_limit = history_limit
        self._render_disk_reserve_mb = render_disk_reserve_mb
        self._render_disk_safety_factor = render_disk_safety_factor
        self._last_persist_at = 0.0
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="travelmovieai-movie",
        )
        self._jobs: dict[UUID, _MovieJob] = {}
        self._lock = RLock()
        self._condition = Condition(self._lock)
        self._shutting_down = False
        self._load()

    def submit(
        self,
        input_path: Path,
        workspace: Path | None,
        settings: QuickMontageSettings,
        variant_name: str = "Default",
    ) -> MovieJobResponse:
        paths = self._service.resolve_project_paths(input_path, workspace)
        register_private_log_paths((paths.input_path, paths.workspace))
        try:
            normalized_variant_name = validate_variant_name(variant_name)
        except ValueError as error:
            raise InvalidProjectPathError(str(error)) from error
        with self._lock:
            if self._workspace_is_active(paths.workspace):
                raise WorkspaceBusyError("A movie edit is already running for this workspace.")
            job = _MovieJob(
                id=uuid4(),
                status=JobStatus.QUEUED,
                input_path=paths.input_path,
                workspace=paths.workspace,
                settings=settings,
                variant_name=normalized_variant_name,
                variant_slug=safe_variant_slug(normalized_variant_name),
                created_at=datetime.now(UTC),
                message="The edit is waiting to start.",
                subtasks=_build_subtasks(settings),
            )
            register_private_log_paths(_job_private_paths(job, self._service))
            _append_log(job, "Job added to the queue.")
            self._jobs[job.id] = job
            self._trim_history()
            try:
                self._persist(force=True, required=True)
            except JobPersistenceError:
                self._jobs.pop(job.id, None)
                raise
        self._executor.submit(self._run, job.id)
        return _to_response(job)

    def get(self, job_id: UUID) -> MovieJobResponse | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return _to_response(job) if job else None

    def list(self, limit: int = 20) -> list[MovieJobResponse]:
        with self._lock:
            jobs = sorted(
                self._jobs.values(),
                key=lambda job: job.created_at,
                reverse=True,
            )
            return [_to_response(job) for job in jobs[:limit]]

    def pause(self, job_id: UUID) -> MovieJobResponse | None:
        with self._condition:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job.status in {JobStatus.QUEUED, JobStatus.RUNNING}:
                job.pause_requested = True
                job.status = JobStatus.PAUSED
                job.paused_at = datetime.now(UTC)
                job.message = (
                    "Pause requested. The current operation will finish before processing stops."
                )
                _append_log(job, job.message)
                self._persist(force=True)
            return _to_response(job)

    def resume(self, job_id: UUID) -> MovieJobResponse | None:
        with self._condition:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job.status == JobStatus.PAUSED and not job.cancel_requested:
                resumed_at = datetime.now(UTC)
                if job.paused_at is not None:
                    paused_for = resumed_at - job.paused_at
                    job.paused_seconds += paused_for.total_seconds()
                    if job.phase_started_at is not None:
                        job.phase_started_at += paused_for
                job.paused_at = None
                job.pause_requested = False
                job.status = JobStatus.RUNNING
                job.message = "Edit resumed."
                _append_log(job, job.message)
                self._persist(force=True)
                self._condition.notify_all()
            return _to_response(job)

    def cancel(self, job_id: UUID) -> MovieJobResponse | None:
        with self._condition:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job.status not in {
                JobStatus.COMPLETED,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
            }:
                job.cancel_requested = True
                job.pause_requested = False
                job.status = JobStatus.CANCELLED
                job.finished_at = datetime.now(UTC)
                job.message = (
                    "Stop requested. The current operation is finishing; no new work will start."
                )
                _fail_active_subtask(job, job.message)
                _append_log(job, job.message, level="warning")
                self._persist(force=True)
                self._condition.notify_all()
            return _to_response(job)

    def resolve_project_paths(self, input_path: Path, workspace: Path | None) -> ProjectPaths:
        return self._service.resolve_project_paths(input_path, workspace)

    def output_path(self, job_id: UUID) -> Path | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.result.output_path if job and job.result else None

    def is_workspace_active(self, workspace: Path) -> bool:
        with self._lock:
            return self._workspace_is_active(workspace)

    def shutdown(self) -> None:
        with self._condition:
            self._shutting_down = True
            self._persist(force=True)
            self._condition.notify_all()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _run(self, job_id: UUID) -> None:
        with correlation_context(str(job_id)):
            self._run_correlated(job_id)

    def _run_correlated(self, job_id: UUID) -> None:
        with self._condition:
            job = self._jobs[job_id]
            while job.pause_requested and not job.cancel_requested and not self._shutting_down:
                self._condition.wait()
            if self._shutting_down:
                job.worker_finished = True
                self._persist(force=True)
                return
            if job.cancel_requested:
                job.worker_finished = True
                self._persist(force=True)
                return
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now(UTC)
            job.phase_started_at = job.started_at
            job.message = "Preparing edit..."
            job.phase = "preparing"
            profile_getter = getattr(self._service, "get_resource_profile", None)
            if callable(profile_getter):
                try:
                    job.resources = profile_getter()
                except Exception:
                    LOGGER.exception("Could not detect hardware resources")
                    job.resources = None
            _append_log(job, job.message)
            try:
                self._persist(force=True, required=True)
            except JobPersistenceError as error:
                job.status = JobStatus.FAILED
                job.finished_at = datetime.now(UTC)
                job.message = "The edit could not start because job state is not writable."
                job.error = str(error)
                job.persistence_degraded = True
                job.worker_finished = True
                return

        def progress(current: int, total: int, message: str) -> None:
            with self._condition:
                while job.pause_requested and not job.cancel_requested and not self._shutting_down:
                    self._condition.wait()
                if self._shutting_down:
                    raise _MovieInterrupted
                if job.cancel_requested:
                    raise _MovieCancelled
                safe_message = redact_sensitive_text(
                    message,
                    private_paths=_job_private_paths(job, self._service),
                )
                now = datetime.now(UTC)
                progress_percent = _progress_percent(current, total)
                phase = (
                    job.pipeline_stage.value
                    if job.settings.semantic_analysis and job.pipeline_stage is not None
                    else _phase_from_message(message)
                )
                normalized_message = message.casefold()
                reset_phase = phase != job.phase or "starting scene analysis" in normalized_message
                if reset_phase:
                    job.phase_started_at = now
                    job.phase_last_progress_at = None
                    job.phase_last_progress_percent = None
                job.progress_current = current
                job.progress_total = total
                job.message = safe_message
                job.phase = phase
                if (
                    job.phase_last_progress_percent is None
                    or progress_percent > job.phase_last_progress_percent
                ):
                    job.phase_last_progress_at = now
                    job.phase_last_progress_percent = progress_percent
                _update_subtasks(job, job.phase, progress_percent, safe_message)
                _append_log(job, safe_message)
                self._persist()

        def progress_event(event: ProgressEvent) -> None:
            with self._condition:
                job.pipeline_stage = event.stage

        try:
            movie_output_path = variant_output_path(job.workspace, job.variant_name, job.id)
            ensure_render_disk_space(
                workspace=job.workspace,
                output_path=movie_output_path,
                settings=job.settings,
                reserve_mb=self._render_disk_reserve_mb,
                safety_factor=self._render_disk_safety_factor,
            )
            create_options: dict[str, Any] = {
                "input_path": job.input_path,
                "workspace": job.workspace,
                "settings": job.settings,
                "variant_name": job.variant_name,
                "output_path": movie_output_path,
                "progress": progress,
            }
            if _accepts_keyword(self._service.create_quick_montage, "progress_events"):
                create_options["progress_events"] = progress_event
            result = self._service.create_quick_montage(**create_options)
        except _MovieCancelled:
            self._mark_worker_finished(job)
            return
        except _MovieInterrupted:
            self._mark_worker_finished(job)
            return
        except TravelMovieError as error:
            self._fail(job, str(error))
            self._mark_worker_finished(job)
            return
        except Exception:
            LOGGER.exception(
                "Movie edit job failed unexpectedly",
                extra={"job_id": str(job.id)},
            )
            self._fail(job, "Internal edit error.")
            self._mark_worker_finished(job)
            return

        with self._condition:
            if job.cancel_requested:
                job.worker_finished = True
                return
            job.status = JobStatus.COMPLETED
            job.finished_at = datetime.now(UTC)
            job.message = "Film ready."
            job.phase = "completed"
            job.result = result
            register_private_log_paths(_job_private_paths(job, self._service))
            job.progress_current = 1000
            job.progress_total = 1000
            _complete_subtasks(job)
            _append_log(job, job.message)
            job.worker_finished = True
            self._trim_history()
            self._persist(force=True)

    def _fail(self, job: _MovieJob, error: str) -> None:
        with self._lock:
            if job.cancel_requested:
                return
            safe_error = redact_sensitive_text(
                error,
                private_paths=_job_private_paths(job, self._service),
            )
            job.status = JobStatus.FAILED
            job.finished_at = datetime.now(UTC)
            job.message = "The edit failed."
            job.phase = "failed"
            job.error = safe_error
            _fail_active_subtask(job, safe_error)
            _append_log(job, safe_error, level="error")
            self._trim_history()
            self._persist(force=True)

    def _mark_worker_finished(self, job: _MovieJob) -> None:
        with self._condition:
            job.worker_finished = True
            self._persist(force=True)
            self._condition.notify_all()

    def _workspace_is_active(self, workspace: Path) -> bool:
        key = os.path.normcase(str(workspace.resolve()))
        return any(
            os.path.normcase(str(job.workspace.resolve())) == key
            and (
                job.status in {JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.PAUSED}
                or job.status == JobStatus.CANCELLED
                and not job.worker_finished
            )
            for job in self._jobs.values()
        )

    def _trim_history(self) -> None:
        if len(self._jobs) <= self._history_limit:
            return
        removable = sorted(
            (
                job
                for job in self._jobs.values()
                if job.status not in {JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.PAUSED}
            ),
            key=lambda job: job.created_at,
        )
        while len(self._jobs) > self._history_limit and removable:
            self._jobs.pop(removable.pop(0).id, None)

    def _load(self) -> None:
        if self._state_path is None or not self._state_path.exists():
            return
        try:
            history = MovieJobStateHistory.model_validate_json(
                self._state_path.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError):
            LOGGER.exception("Could not load movie job history")
            return

        now = datetime.now(UTC)
        recovered_ids: list[UUID] = []
        for saved in history.jobs:
            job = _from_state(saved)
            if job.status in {JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.PAUSED}:
                was_paused = job.status is JobStatus.PAUSED
                job.status = JobStatus.PAUSED if was_paused else JobStatus.QUEUED
                job.started_at = None
                job.finished_at = None
                job.phase_started_at = None
                job.phase_last_progress_at = None
                job.phase_last_progress_percent = None
                job.progress_current = 0
                job.progress_total = 0
                job.phase = "queued"
                job.error = None
                job.result = None
                job.pause_requested = was_paused
                job.cancel_requested = False
                job.paused_at = now if was_paused else None
                job.worker_finished = False
                job.subtasks = _build_subtasks(job.settings)
                job.message = (
                    "Recovered paused edit. Continue when ready; valid stage caches will be reused."
                    if was_paused
                    else "Recovered interrupted edit and queued it to resume from validated caches."
                )
                _append_log(job, job.message, level="warning")
                recovered_ids.append(job.id)
            else:
                job.worker_finished = True
            self._jobs[job.id] = job
        self._trim_history()
        self._persist(force=True)
        for job_id in recovered_ids:
            self._executor.submit(self._run, job_id)

    def _persist(self, *, force: bool = False, required: bool = False) -> bool:
        if self._state_path is None:
            return True
        now = monotonic()
        if not force and now - self._last_persist_at < 1.0:
            return True
        jobs = sorted(
            (
                _to_state(
                    job,
                    private_paths=_job_private_paths(job, self._service),
                )
                for job in self._jobs.values()
            ),
            key=lambda job: job.created_at,
            reverse=True,
        )
        try:
            write_json_atomic(self._state_path, MovieJobStateHistory(jobs=jobs))
        except OSError as error:
            LOGGER.exception("Could not persist movie job history")
            for job in self._jobs.values():
                if job.status in {JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.PAUSED}:
                    job.persistence_degraded = True
            if required:
                raise JobPersistenceError(
                    "Restart-safe edit state could not be written; check workspace "
                    "permissions and disk space."
                ) from error
            return False
        self._last_persist_at = now
        return True


def _to_state(
    job: _MovieJob,
    *,
    private_paths: tuple[Path, ...] = (),
) -> MovieJobState:
    safe_message = redact_sensitive_text(job.message, private_paths=private_paths)
    safe_error = (
        redact_sensitive_text(job.error, private_paths=private_paths)
        if job.error is not None
        else None
    )
    safe_subtasks = [
        task.model_copy(
            update={
                "message": redact_sensitive_text(task.message, private_paths=private_paths),
            }
        )
        for task in job.subtasks
    ]
    safe_logs = [
        entry.model_copy(
            update={
                "message": redact_sensitive_text(entry.message, private_paths=private_paths),
            }
        )
        for entry in job.logs
    ]
    safe_result = job.result
    if safe_result is not None and safe_result.music_model is not None:
        safe_result = safe_result.model_copy(
            update={
                "music_model": redact_sensitive_text(
                    safe_result.music_model,
                    private_paths=private_paths,
                )
            }
        )
    return MovieJobState(
        id=job.id,
        status=job.status,
        input_path=job.input_path,
        workspace=job.workspace,
        variant_name=job.variant_name,
        variant_slug=job.variant_slug,
        settings=job.settings,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        phase_started_at=job.phase_started_at,
        phase_last_progress_at=job.phase_last_progress_at,
        phase_last_progress_percent=job.phase_last_progress_percent,
        message=safe_message,
        error=safe_error,
        progress_current=job.progress_current,
        progress_total=job.progress_total,
        phase=job.phase,
        pipeline_stage=job.pipeline_stage,
        resources=(
            ResourceProfileResponse.model_validate(asdict(job.resources)) if job.resources else None
        ),
        subtasks=safe_subtasks,
        logs=safe_logs,
        result=safe_result,
        paused_seconds=job.paused_seconds,
        persistence_degraded=job.persistence_degraded,
    )


def _from_state(saved: MovieJobState) -> _MovieJob:
    return _MovieJob(
        id=saved.id,
        status=saved.status,
        input_path=saved.input_path,
        workspace=saved.workspace,
        variant_name=saved.variant_name,
        variant_slug=saved.variant_slug,
        settings=saved.settings,
        created_at=saved.created_at,
        started_at=saved.started_at,
        finished_at=saved.finished_at,
        phase_started_at=saved.phase_started_at,
        phase_last_progress_at=saved.phase_last_progress_at,
        phase_last_progress_percent=saved.phase_last_progress_percent,
        message=saved.message,
        error=saved.error,
        progress_current=saved.progress_current,
        progress_total=saved.progress_total,
        phase=saved.phase,
        pipeline_stage=saved.pipeline_stage,
        resources=(ResourceProfile(**saved.resources.model_dump()) if saved.resources else None),
        subtasks=saved.subtasks,
        logs=saved.logs,
        result=saved.result,
        paused_seconds=saved.paused_seconds,
        persistence_degraded=saved.persistence_degraded,
    )


def _to_response(job: _MovieJob) -> MovieJobResponse:
    result = job.result
    now = job.finished_at or datetime.now(UTC)
    started_at = job.started_at or job.created_at
    active_pause_seconds = (
        max(0.0, (now - job.paused_at).total_seconds()) if job.paused_at is not None else 0.0
    )
    elapsed_seconds = max(
        0.0,
        (now - started_at).total_seconds() - job.paused_seconds - active_pause_seconds,
    )
    progress_percent = (
        min(100.0, job.progress_current / job.progress_total * 100)
        if job.progress_total > 0
        else 0.0
    )
    eta_seconds = None
    phase_range = _subtask_ranges(job.settings).get(job.phase)
    if (
        job.status == JobStatus.RUNNING
        and phase_range is not None
        and job.phase_started_at is not None
        and job.phase_last_progress_at is not None
        and job.phase_last_progress_percent is not None
    ):
        eta_seconds = _estimate_phase_eta(
            phase_started_at=job.phase_started_at,
            last_progress_at=job.phase_last_progress_at,
            now=now,
            phase_start_percent=phase_range[0],
            phase_end_percent=phase_range[1],
            last_progress_percent=job.phase_last_progress_percent,
        )
    return MovieJobResponse(
        id=job.id,
        status=job.status,
        input_path=job.input_path,
        workspace=job.workspace,
        variant_name=job.variant_name,
        variant_slug=job.variant_slug,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        message=job.message,
        error=job.error,
        phase=job.phase,
        pipeline_stage=job.pipeline_stage,
        progress_current=job.progress_current,
        progress_total=job.progress_total,
        progress_percent=progress_percent,
        elapsed_seconds=elapsed_seconds,
        eta_seconds=eta_seconds,
        resources=(
            ResourceProfileResponse.model_validate(asdict(job.resources)) if job.resources else None
        ),
        subtasks=job.subtasks,
        logs=job.logs,
        output_path=result.output_path if result else None,
        clip_count=result.clip_count if result else None,
        duration_seconds=result.duration_seconds if result else None,
        selection_mode=result.selection_mode if result else None,
        render_encoder=result.render_encoder if result else None,
        music_mode=result.music_mode if result else None,
        music_profile=result.music_profile if result else None,
        music_generator=result.music_generator if result else None,
        music_model=result.music_model if result else None,
        quality_score=result.quality_score if result else None,
        quality_issue_count=result.quality_issue_count if result else 0,
        quality_gate_status=result.quality_gate_status if result else None,
        semantic_score_p10=result.semantic_score_p10 if result else None,
        dominant_event_ratio=result.dominant_event_ratio if result else None,
        adjacent_source_repeat_ratio=(result.adjacent_source_repeat_ratio if result else None),
        center_cut_ratio=result.center_cut_ratio if result else None,
        full_media_qa_completed=result.full_media_qa_completed if result else False,
        persistence_degraded=job.persistence_degraded,
    )


def _append_log(job: _MovieJob, message: str, *, level: str = "info") -> None:
    progress_percent = _progress_percent(job.progress_current, job.progress_total)
    if job.logs and job.logs[-1].message == message and job.logs[-1].level == level:
        return
    job.logs.append(
        JobLogEntry(
            timestamp=datetime.now(UTC),
            level=level,
            phase=job.phase,
            message=message,
            progress_percent=progress_percent,
        )
    )
    del job.logs[:-250]


def _job_private_paths(job: _MovieJob, service: object) -> tuple[Path, ...]:
    paths: list[Path] = []

    def add_path(value: object, *, strings_must_be_absolute: bool = False) -> None:
        if isinstance(value, Path):
            candidate = value.expanduser()
        elif isinstance(value, str) and value:
            candidate = Path(value).expanduser()
            if strings_must_be_absolute and not candidate.is_absolute():
                return
        else:
            return
        if candidate.is_absolute():
            paths.append(candidate)
        with suppress(OSError):
            paths.append(candidate.resolve())

    add_path(job.input_path)
    add_path(job.workspace)
    add_path(job.settings.music_path)
    add_path(job.settings.overlay_font_path)
    add_path(job.settings.vision_model, strings_must_be_absolute=True)
    add_path(job.settings.music_model, strings_must_be_absolute=True)
    if job.result is not None:
        add_path(job.result.output_path)
        add_path(job.result.timeline_path)
        add_path(job.result.music_model, strings_must_be_absolute=True)

    application_settings = getattr(service, "settings", None)
    if application_settings is not None:
        for field_name in ("model_cache", "music_library", "piper_model"):
            add_path(getattr(application_settings, field_name, None))
        for field_name in (
            "vision_model",
            "embedding_model",
            "story_model",
            "music_model",
        ):
            add_path(
                getattr(application_settings, field_name, None),
                strings_must_be_absolute=True,
            )
    return tuple(dict.fromkeys(paths))


def _build_subtasks(settings: QuickMontageSettings) -> list[JobSubtaskProgress]:
    quick_stages = {
        PipelineStage.MEDIA_SCAN.value,
        PipelineStage.SCENE_RANKING.value,
        PipelineStage.MUSIC_SELECTION.value,
        PipelineStage.TIMELINE_BUILDER.value,
        PipelineStage.RENDERING.value,
    }
    subtasks = []
    for task_id, (_, _, label) in _subtask_ranges(settings).items():
        skipped = (
            not settings.semantic_analysis
            and task_id not in quick_stages
            or task_id == "quality_analysis"
            and not settings.quality_analysis
            or task_id == "speech_analysis"
            and not settings.speech_analysis
            or task_id == "audio_analysis"
            and not settings.audio_analysis
            or task_id == "music_selection"
            and not settings.music_enabled
            or task_id in {"narration", "voice_synthesis"}
            and not settings.narration_enabled
        )
        subtasks.append(
            JobSubtaskProgress(
                id=task_id,
                label=label,
                status="skipped" if skipped else "pending",
                progress_percent=100 if skipped else 0,
                message="Disabled in settings" if skipped else "Waiting",
            )
        )
    return subtasks


def _update_subtasks(
    job: _MovieJob,
    phase: str,
    global_percent: float,
    message: str,
) -> None:
    ranges = _subtask_ranges(job.settings)
    if phase not in ranges:
        return
    semantic_stage_order = (
        {stage.value: index for index, stage in enumerate(PipelineStage)}
        if job.settings.semantic_analysis
        else {}
    )
    active_stage_index = semantic_stage_order.get(phase)
    updated = []
    for task in job.subtasks:
        if task.status == "skipped":
            updated.append(task)
            continue
        start, end, _ = ranges[task.id]
        if task.id == phase:
            local_percent = min(100.0, max(0.0, (global_percent - start) / (end - start) * 100))
            if global_percent > 0 and local_percent == 0:
                local_percent = 0.1
            updated.append(
                task.model_copy(
                    update={
                        "status": "completed" if local_percent >= 100 else "running",
                        "progress_percent": local_percent,
                        "message": message,
                    }
                )
            )
        elif (
            active_stage_index is not None
            and semantic_stage_order.get(task.id, len(PipelineStage)) < active_stage_index
        ) or end <= global_percent:
            updated.append(
                task.model_copy(
                    update={
                        "status": "completed",
                        "progress_percent": 100,
                        "message": (task.message if task.message != "Waiting" else "Complete"),
                    }
                )
            )
        else:
            updated.append(task)
    job.subtasks = updated


def _complete_subtasks(job: _MovieJob) -> None:
    job.subtasks = [
        task
        if task.status == "skipped"
        else task.model_copy(
            update={
                "status": "completed",
                "progress_percent": 100,
                "message": task.message if task.message != "Waiting" else "Complete",
            }
        )
        for task in job.subtasks
    ]


def _fail_active_subtask(job: _MovieJob, error: str) -> None:
    job.subtasks = [
        task.model_copy(update={"status": "failed", "message": error})
        if task.status == "running"
        else task
        for task in job.subtasks
    ]


def _progress_percent(current: int, total: int) -> float:
    return min(100.0, current / total * 100) if total > 0 else 0.0


def _estimate_phase_eta(
    *,
    phase_started_at: datetime,
    last_progress_at: datetime,
    now: datetime,
    phase_start_percent: float,
    phase_end_percent: float,
    last_progress_percent: float,
) -> float | None:
    phase_size = phase_end_percent - phase_start_percent
    if phase_size <= 0:
        return None
    completed_fraction = (last_progress_percent - phase_start_percent) / phase_size
    if not 0 < completed_fraction < 1:
        return None
    measured_seconds = (last_progress_at - phase_started_at).total_seconds()
    if measured_seconds <= 0:
        return None
    eta_at_last_progress = measured_seconds / completed_fraction * (1 - completed_fraction)
    seconds_since_progress = max(0.0, (now - last_progress_at).total_seconds())
    if seconds_since_progress > eta_at_last_progress:
        return None
    return eta_at_last_progress - seconds_since_progress


def _phase_from_message(message: str) -> str:
    normalized = message.casefold()
    phases = (
        (
            ("resource profile", "checking media", "media scan"),
            "media_scan",
        ),
        (("scene detection", "scenes found"), "scene_detection"),
        (("frames", "frame extraction"), "frame_sampling"),
        (("opencv", "cuda quality", "quality analysis"), "quality_analysis"),
        (("ai analysis", "ai cache", "vision"), "vision_analysis"),
        (("whisper", "speech"), "speech_analysis"),
        (("audio analysis", "silence", "noise"), "audio_analysis"),
        (("embedding", "faiss"), "embeddings"),
        (("duplicate",), "duplicate_detection"),
        (("caption", "description"), "scene_captioning"),
        (("event detection", "events"), "event_detection"),
        (("story builder", "storyboard", "story"), "story_builder"),
        (("scene ranking", "selection", "selecting quick clips"), "scene_ranking"),
        (("music selection", "music", "soundtrack", "ace-step"), "music_selection"),
        (("voice synthesis", "piper"), "voice_synthesis"),
        (("narration",), "narration"),
        (("timeline", "quick edit plan"), "timeline_builder"),
        (("render", "clip", "transition", "assembly"), "rendering"),
        (("ffprobe", "film ready"), "rendering"),
    )
    for markers, phase in phases:
        if any(marker in normalized for marker in markers):
            return phase
    return "processing"


def _accepts_keyword(callback: Callable[..., object], name: str) -> bool:
    try:
        parameters = inspect.signature(callback).parameters
    except (TypeError, ValueError):
        return False
    return name in parameters


def _subtask_ranges(
    settings: QuickMontageSettings,
) -> dict[str, tuple[float, float, str]]:
    return _SEMANTIC_SUBTASK_RANGES if settings.semantic_analysis else _QUICK_SUBTASK_RANGES
