"""Background quick montage jobs."""

import logging
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from threading import Condition, RLock
from typing import Protocol
from uuid import UUID, uuid4

from travelmovieai.application.validation import ProjectPaths
from travelmovieai.core.exceptions import TravelMovieError, WorkspaceBusyError
from travelmovieai.domain.models import QuickMontageResult, QuickMontageSettings
from travelmovieai.infrastructure.system import ResourceProfile
from travelmovieai.web.schemas import (
    JobLogEntry,
    JobStatus,
    JobSubtaskProgress,
    MovieJobResponse,
    ResourceProfileResponse,
)

LOGGER = logging.getLogger(__name__)

_SUBTASK_RANGES = {
    "media_scan": (0.0, 6.0, "Media Scan"),
    "scene_detection": (6.0, 12.0, "Scene Detection"),
    "frame_sampling": (12.0, 32.0, "Frame Extraction"),
    "quality_analysis": (32.0, 45.0, "OpenCV Analysis"),
    "vision_analysis": (45.0, 70.0, "Vision AI"),
    "speech_analysis": (70.0, 74.0, "Speech Recognition"),
    "audio_analysis": (74.0, 76.0, "Audio Analysis"),
    "story_builder": (76.0, 82.0, "Story and Selection"),
    "music": (82.0, 84.0, "Music"),
    "timeline": (84.0, 85.0, "Timeline"),
    "rendering": (85.0, 99.0, "Rendering"),
    "validation": (99.0, 100.0, "Validation"),
}


class _MovieCancelled(Exception):
    pass


class MovieService(Protocol):
    def resolve_project_paths(self, input_path: Path, workspace: Path | None) -> ProjectPaths: ...

    def create_quick_montage(
        self,
        *,
        input_path: Path,
        workspace: Path | None,
        settings: QuickMontageSettings,
        output_path: Path | None = None,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> QuickMontageResult: ...


@dataclass(slots=True)
class _MovieJob:
    id: UUID
    status: JobStatus
    input_path: Path
    workspace: Path
    settings: QuickMontageSettings
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
    resources: ResourceProfile | None = None
    subtasks: list[JobSubtaskProgress] = field(default_factory=list)
    logs: list[JobLogEntry] = field(default_factory=list)
    result: QuickMontageResult | None = None
    pause_requested: bool = False
    cancel_requested: bool = False
    worker_finished: bool = False
    paused_at: datetime | None = None
    paused_seconds: float = 0


class MovieJobManager:
    def __init__(self, service: MovieService) -> None:
        self._service = service
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="travelmovieai-movie",
        )
        self._jobs: dict[UUID, _MovieJob] = {}
        self._lock = RLock()
        self._condition = Condition(self._lock)

    def submit(
        self,
        input_path: Path,
        workspace: Path | None,
        settings: QuickMontageSettings,
    ) -> MovieJobResponse:
        paths = self._service.resolve_project_paths(input_path, workspace)
        with self._lock:
            if self._workspace_is_active(paths.workspace):
                raise WorkspaceBusyError("A movie edit is already running for this workspace.")
            job = _MovieJob(
                id=uuid4(),
                status=JobStatus.QUEUED,
                input_path=paths.input_path,
                workspace=paths.workspace,
                settings=settings,
                created_at=datetime.now(UTC),
                message="The edit is waiting to start.",
                subtasks=_build_subtasks(settings),
            )
            _append_log(job, "Job added to the queue.")
            self._jobs[job.id] = job
        self._executor.submit(self._run, job.id)
        return _to_response(job)

    def get(self, job_id: UUID) -> MovieJobResponse | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return _to_response(job) if job else None

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
        self._executor.shutdown(wait=False, cancel_futures=False)

    def _run(self, job_id: UUID) -> None:
        with self._condition:
            job = self._jobs[job_id]
            while job.pause_requested and not job.cancel_requested:
                self._condition.wait()
            if job.cancel_requested:
                job.worker_finished = True
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

        def progress(current: int, total: int, message: str) -> None:
            with self._condition:
                while job.pause_requested and not job.cancel_requested:
                    self._condition.wait()
                if job.cancel_requested:
                    raise _MovieCancelled
                now = datetime.now(UTC)
                progress_percent = _progress_percent(current, total)
                phase = _phase_from_message(message)
                normalized_message = message.casefold()
                reset_phase = (
                    phase != job.phase
                    or "starting scene analysis" in normalized_message
                )
                if reset_phase:
                    job.phase_started_at = now
                    job.phase_last_progress_at = None
                    job.phase_last_progress_percent = None
                job.progress_current = current
                job.progress_total = total
                job.message = message
                job.phase = phase
                if (
                    job.phase_last_progress_percent is None
                    or progress_percent > job.phase_last_progress_percent
                ):
                    job.phase_last_progress_at = now
                    job.phase_last_progress_percent = progress_percent
                _update_subtasks(job, job.phase, progress_percent, message)
                _append_log(job, message)

        try:
            result = self._service.create_quick_montage(
                input_path=job.input_path,
                workspace=job.workspace,
                settings=job.settings,
                progress=progress,
            )
        except _MovieCancelled:
            self._mark_worker_finished(job)
            return
        except TravelMovieError as error:
            self._fail(job, str(error))
            self._mark_worker_finished(job)
            return
        except Exception:
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
            job.progress_current = 1000
            job.progress_total = 1000
            _complete_subtasks(job)
            _append_log(job, job.message)
            job.worker_finished = True

    def _fail(self, job: _MovieJob, error: str) -> None:
        with self._lock:
            if job.cancel_requested:
                return
            job.status = JobStatus.FAILED
            job.finished_at = datetime.now(UTC)
            job.message = "The edit failed."
            job.phase = "failed"
            job.error = error
            _fail_active_subtask(job, error)
            _append_log(job, error, level="error")

    def _mark_worker_finished(self, job: _MovieJob) -> None:
        with self._condition:
            job.worker_finished = True
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
    phase_range = _SUBTASK_RANGES.get(job.phase)
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
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        message=job.message,
        error=job.error,
        phase=job.phase,
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


def _build_subtasks(settings: QuickMontageSettings) -> list[JobSubtaskProgress]:
    semantic_only = {
        "scene_detection",
        "frame_sampling",
        "quality_analysis",
        "vision_analysis",
        "speech_analysis",
        "audio_analysis",
        "story_builder",
    }
    subtasks = []
    for task_id, (_, _, label) in _SUBTASK_RANGES.items():
        skipped = (
            task_id in semantic_only
            and not settings.semantic_analysis
            or task_id == "quality_analysis"
            and not settings.quality_analysis
            or task_id == "speech_analysis"
            and not settings.speech_analysis
            or task_id == "audio_analysis"
            and not settings.audio_analysis
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
    if phase not in _SUBTASK_RANGES:
        return
    updated = []
    for task in job.subtasks:
        if task.status == "skipped":
            updated.append(task)
            continue
        start, end, _ = _SUBTASK_RANGES[task.id]
        if task.id == phase:
            local_percent = min(100.0, max(0.0, (global_percent - start) / (end - start) * 100))
            updated.append(
                task.model_copy(
                    update={
                        "status": "completed" if local_percent >= 100 else "running",
                        "progress_percent": local_percent,
                        "message": message,
                    }
                )
            )
        elif end <= global_percent:
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
        (
            (
                "duplicate",
                "description",
                "events",
                "story",
                "selection",
            ),
            "story_builder",
        ),
        (("music", "soundtrack", "ace-step"), "music"),
        (("timeline",), "timeline"),
        (("render", "clip", "transition", "assembly"), "rendering"),
        (("ffprobe", "film ready"), "validation"),
    )
    for markers, phase in phases:
        if any(marker in normalized for marker in markers):
            return phase
    return "processing"
