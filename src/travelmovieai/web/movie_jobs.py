"""Background quick montage jobs."""

import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Protocol
from uuid import UUID, uuid4

from travelmovieai.application.validation import ProjectPaths
from travelmovieai.core.exceptions import TravelMovieError, WorkspaceBusyError
from travelmovieai.domain.models import QuickMontageResult, QuickMontageSettings
from travelmovieai.web.schemas import JobStatus, MovieJobResponse


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
    message: str = ""
    error: str | None = None
    progress_current: int = 0
    progress_total: int = 0
    result: QuickMontageResult | None = None


class MovieJobManager:
    def __init__(self, service: MovieService) -> None:
        self._service = service
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="travelmovieai-movie",
        )
        self._jobs: dict[UUID, _MovieJob] = {}
        self._lock = RLock()

    def submit(
        self,
        input_path: Path,
        workspace: Path | None,
        settings: QuickMontageSettings,
    ) -> MovieJobResponse:
        paths = self._service.resolve_project_paths(input_path, workspace)
        with self._lock:
            if self._workspace_is_active(paths.workspace):
                raise WorkspaceBusyError("Для этого workspace уже выполняется монтаж фильма.")
            job = _MovieJob(
                id=uuid4(),
                status=JobStatus.QUEUED,
                input_path=paths.input_path,
                workspace=paths.workspace,
                settings=settings,
                created_at=datetime.now(UTC),
                message="Монтаж ожидает запуска.",
            )
            self._jobs[job.id] = job
        self._executor.submit(self._run, job.id)
        return _to_response(job)

    def get(self, job_id: UUID) -> MovieJobResponse | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return _to_response(job) if job else None

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
        with self._lock:
            job = self._jobs[job_id]
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now(UTC)
            job.message = "Подготовка монтажа..."

        def progress(current: int, total: int, message: str) -> None:
            with self._lock:
                job.progress_current = current
                job.progress_total = total
                job.message = message

        try:
            result = self._service.create_quick_montage(
                input_path=job.input_path,
                workspace=job.workspace,
                settings=job.settings,
                progress=progress,
            )
        except TravelMovieError as error:
            self._fail(job, str(error))
            return
        except Exception:
            self._fail(job, "Внутренняя ошибка монтажа.")
            return

        with self._lock:
            job.status = JobStatus.COMPLETED
            job.finished_at = datetime.now(UTC)
            job.message = "Фильм готов."
            job.result = result
            job.progress_current = job.progress_total

    def _fail(self, job: _MovieJob, error: str) -> None:
        with self._lock:
            job.status = JobStatus.FAILED
            job.finished_at = datetime.now(UTC)
            job.message = "Монтаж завершился с ошибкой."
            job.error = error

    def _workspace_is_active(self, workspace: Path) -> bool:
        key = os.path.normcase(str(workspace.resolve()))
        return any(
            os.path.normcase(str(job.workspace.resolve())) == key
            and job.status in {JobStatus.QUEUED, JobStatus.RUNNING}
            for job in self._jobs.values()
        )


def _to_response(job: _MovieJob) -> MovieJobResponse:
    result = job.result
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
        progress_current=job.progress_current,
        progress_total=job.progress_total,
        output_path=result.output_path if result else None,
        clip_count=result.clip_count if result else None,
        duration_seconds=result.duration_seconds if result else None,
        selection_mode=result.selection_mode if result else None,
        render_encoder=result.render_encoder if result else None,
        music_mode=result.music_mode if result else None,
        music_profile=result.music_profile if result else None,
    )
