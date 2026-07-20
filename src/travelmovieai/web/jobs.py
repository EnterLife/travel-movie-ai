"""Background scan job management for the local web UI."""

import logging
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Protocol
from uuid import UUID, uuid4

from pydantic import ValidationError

from travelmovieai.application.validation import ProjectPaths
from travelmovieai.core.exceptions import JobPersistenceError, TravelMovieError, WorkspaceBusyError
from travelmovieai.core.logging import register_private_log_paths
from travelmovieai.domain.models import MediaScanReport, StageResult
from travelmovieai.infrastructure.artifacts import write_json_atomic
from travelmovieai.web.schemas import JobStatus, ScanJobHistory, ScanJobResponse
from travelmovieai.web.state import redact_sensitive_text

LOGGER = logging.getLogger(__name__)


class ScanService(Protocol):
    def analyze(
        self,
        *,
        input_path: Path,
        workspace: Path | None,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> StageResult: ...

    def resolve_project_paths(self, input_path: Path, workspace: Path | None) -> ProjectPaths: ...


@dataclass(slots=True)
class _ScanJob:
    id: UUID
    status: JobStatus
    input_path: Path
    workspace: Path
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    message: str = ""
    error: str | None = None
    report: MediaScanReport | None = None
    progress_current: int = 0
    progress_total: int = 0
    persistence_degraded: bool = False


class _ScanInterrupted(Exception):
    """Stop this worker while keeping its persisted job recoverable."""


class ScanJobManager:
    def __init__(
        self,
        service: ScanService,
        *,
        state_path: Path | None = None,
        history_limit: int = 100,
    ) -> None:
        self._service = service
        self._state_path = state_path
        self._history_limit = history_limit
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="travelmovieai-scan",
        )
        self._jobs: dict[UUID, _ScanJob] = {}
        self._lock = RLock()
        self._shutting_down = False
        self._load()

    def submit(self, input_path: Path, workspace: Path | None) -> ScanJobResponse:
        project_paths = self.resolve_project_paths(input_path, workspace)
        register_private_log_paths((project_paths.input_path, project_paths.workspace))
        job = _ScanJob(
            id=uuid4(),
            status=JobStatus.QUEUED,
            input_path=project_paths.input_path,
            workspace=project_paths.workspace,
            created_at=datetime.now(UTC),
            message="The job is waiting to start.",
        )
        with self._lock:
            if self._workspace_is_active(project_paths.workspace):
                raise WorkspaceBusyError(
                    "Another job is already queued or running for this workspace."
                )
            self._jobs[job.id] = job
            self._trim_history()
            try:
                self._persist(required=True)
            except JobPersistenceError:
                self._jobs.pop(job.id, None)
                raise
        self._executor.submit(self._run, job.id)
        return _to_response(job)

    def resolve_project_paths(self, input_path: Path, workspace: Path | None) -> ProjectPaths:
        return self._service.resolve_project_paths(input_path, workspace)

    def get(self, job_id: UUID) -> ScanJobResponse | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return _to_response(job) if job else None

    def get_report(self, job_id: UUID) -> MediaScanReport | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.report if job else None

    def list(self, limit: int = 20) -> list[ScanJobResponse]:
        with self._lock:
            jobs = sorted(
                self._jobs.values(),
                key=lambda job: job.created_at,
                reverse=True,
            )
            return [_to_response(job) for job in jobs[:limit]]

    def is_workspace_active(self, workspace: Path) -> bool:
        with self._lock:
            return self._workspace_is_active(workspace)

    def shutdown(self) -> None:
        with self._lock:
            self._shutting_down = True
            self._persist()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _run(self, job_id: UUID) -> None:
        with self._lock:
            job = self._jobs[job_id]
            if self._shutting_down:
                return
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now(UTC)
            job.finished_at = None
            job.error = None
            job.message = "Scanning media..."
            try:
                self._persist(required=True)
            except JobPersistenceError as error:
                job.status = JobStatus.FAILED
                job.finished_at = datetime.now(UTC)
                job.message = "The scan could not start because job state is not writable."
                job.error = str(error)
                job.persistence_degraded = True
                return

        def progress(current: int, total: int, message: str) -> None:
            with self._lock:
                if self._shutting_down:
                    raise _ScanInterrupted
                job.progress_current = max(0, current)
                job.progress_total = max(0, total)
                job.message = message
                self._persist()

        try:
            result = self._service.analyze(
                input_path=job.input_path,
                workspace=job.workspace,
                progress=progress,
            )
            report_path = job.workspace / "artifacts" / "analysis.json"
            report = MediaScanReport.model_validate_json(report_path.read_text(encoding="utf-8"))
        except _ScanInterrupted:
            return
        except (TravelMovieError, OSError, ValidationError) as error:
            self._fail(job, str(error))
            return
        except Exception:
            LOGGER.exception("Unexpected scan job failure", extra={"job_id": str(job.id)})
            self._fail(job, "Internal server error. Details were written to the log.")
            return

        with self._lock:
            job.status = JobStatus.COMPLETED
            job.finished_at = datetime.now(UTC)
            job.message = result.message
            job.report = report
            job.progress_current = 1
            job.progress_total = 1
            self._persist()

    def _fail(self, job: _ScanJob, error: str) -> None:
        with self._lock:
            safe_error = redact_sensitive_text(
                error,
                private_paths=[job.input_path, job.workspace],
            )
            job.status = JobStatus.FAILED
            job.finished_at = datetime.now(UTC)
            job.message = "The scan failed."
            job.error = safe_error
            self._persist()

    def _workspace_is_active(self, workspace: Path) -> bool:
        key = _path_key(workspace)
        return any(
            _path_key(job.workspace) == key and job.status in {JobStatus.QUEUED, JobStatus.RUNNING}
            for job in self._jobs.values()
        )

    def _trim_history(self) -> None:
        if len(self._jobs) <= self._history_limit:
            return
        removable = sorted(
            (
                job
                for job in self._jobs.values()
                if job.status not in {JobStatus.QUEUED, JobStatus.RUNNING}
            ),
            key=lambda job: job.created_at,
        )
        while len(self._jobs) > self._history_limit and removable:
            self._jobs.pop(removable.pop(0).id, None)

    def _load(self) -> None:
        if self._state_path is None or not self._state_path.exists():
            return
        try:
            history = ScanJobHistory.model_validate_json(
                self._state_path.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError):
            LOGGER.exception("Could not load web job history")
            return

        recovered_ids: list[UUID] = []
        for saved in history.jobs:
            status = saved.status
            message = saved.message
            error = saved.error
            finished_at = saved.finished_at
            if status in {JobStatus.QUEUED, JobStatus.RUNNING}:
                status = JobStatus.QUEUED
                message = "Recovered interrupted scan; waiting to resume."
                error = None
                finished_at = None
                recovered_ids.append(saved.id)

            report = _load_report(saved.workspace) if status == JobStatus.COMPLETED else None
            self._jobs[saved.id] = _ScanJob(
                id=saved.id,
                status=status,
                input_path=saved.input_path,
                workspace=saved.workspace,
                created_at=saved.created_at,
                started_at=saved.started_at,
                finished_at=finished_at,
                message=message,
                error=error,
                report=report,
                progress_current=saved.progress_current,
                progress_total=saved.progress_total,
                persistence_degraded=saved.persistence_degraded,
            )
        self._trim_history()
        self._persist()
        for job_id in recovered_ids:
            self._executor.submit(self._run, job_id)

    def _persist(self, *, required: bool = False) -> bool:
        if self._state_path is None:
            return True
        jobs = sorted(
            (_to_response(job) for job in self._jobs.values()),
            key=lambda job: job.created_at,
            reverse=True,
        )
        try:
            write_json_atomic(self._state_path, ScanJobHistory(jobs=jobs))
        except OSError as error:
            LOGGER.exception("Could not persist web job history")
            for job in self._jobs.values():
                if job.status in {JobStatus.QUEUED, JobStatus.RUNNING}:
                    job.persistence_degraded = True
            if required:
                raise JobPersistenceError(
                    "Restart-safe scan state could not be written; check workspace "
                    "permissions and disk space."
                ) from error
            return False
        return True


def _load_report(workspace: Path) -> MediaScanReport | None:
    report_path = workspace / "artifacts" / "analysis.json"
    if not report_path.exists():
        return None
    try:
        return MediaScanReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    except (OSError, ValidationError):
        LOGGER.warning("Could not restore scan report from %s", report_path)
        return None


def _path_key(path: Path) -> str:
    return os.path.normcase(str(path.resolve()))


def _to_response(job: _ScanJob) -> ScanJobResponse:
    return ScanJobResponse(
        id=job.id,
        status=job.status,
        input_path=job.input_path,
        workspace=job.workspace,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        message=job.message,
        error=job.error,
        progress_current=job.progress_current,
        progress_total=job.progress_total,
        progress_percent=(
            min(100.0, job.progress_current / job.progress_total * 100)
            if job.progress_total > 0
            else 0.0
        ),
        persistence_degraded=job.persistence_degraded,
    )
