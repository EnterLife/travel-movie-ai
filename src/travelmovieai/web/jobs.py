"""Background scan job management for the local web UI."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Protocol
from uuid import UUID, uuid4

from travelmovieai.domain.models import MediaScanReport, StageResult
from travelmovieai.web.schemas import JobStatus, ScanJobResponse


class ScanService(Protocol):
    def analyze(self, *, input_path: Path, workspace: Path | None) -> StageResult: ...

    def resolve_workspace(self, input_path: Path, workspace: Path | None) -> Path: ...


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


class ScanJobManager:
    def __init__(self, service: ScanService) -> None:
        self._service = service
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="travelmovieai-scan",
        )
        self._jobs: dict[UUID, _ScanJob] = {}
        self._lock = RLock()

    def submit(self, input_path: Path, workspace: Path | None) -> ScanJobResponse:
        resolved_input = input_path.resolve()
        resolved_workspace = self.resolve_workspace(resolved_input, workspace)
        job = _ScanJob(
            id=uuid4(),
            status=JobStatus.QUEUED,
            input_path=resolved_input,
            workspace=resolved_workspace,
            created_at=datetime.now(UTC),
            message="Задание ожидает запуска.",
        )
        with self._lock:
            self._jobs[job.id] = job
        self._executor.submit(self._run, job.id)
        return _to_response(job)

    def resolve_workspace(self, input_path: Path, workspace: Path | None) -> Path:
        return self._service.resolve_workspace(input_path, workspace)

    def get(self, job_id: UUID) -> ScanJobResponse | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return _to_response(job) if job else None

    def get_report(self, job_id: UUID) -> MediaScanReport | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.report if job else None

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=False)

    def _run(self, job_id: UUID) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now(UTC)
            job.message = "Сканирование медиатеки..."

        try:
            result = self._service.analyze(
                input_path=job.input_path,
                workspace=job.workspace,
            )
            report_path = job.workspace / "artifacts" / "analysis.json"
            report = MediaScanReport.model_validate_json(report_path.read_text(encoding="utf-8"))
        except Exception as error:
            with self._lock:
                job.status = JobStatus.FAILED
                job.finished_at = datetime.now(UTC)
                job.message = "Сканирование завершилось с ошибкой."
                job.error = str(error)
            return

        with self._lock:
            job.status = JobStatus.COMPLETED
            job.finished_at = datetime.now(UTC)
            job.message = result.message
            job.report = report


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
    )
