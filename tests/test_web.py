import time
from datetime import UTC, datetime
from pathlib import Path
from threading import Event
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from travelmovieai.application.validation import (
    ProjectPaths,
    validate_project_paths,
)
from travelmovieai.core.exceptions import WorkspaceBusyError
from travelmovieai.domain.enums import PipelineStage
from travelmovieai.domain.models import MediaScanReport, StageResult
from travelmovieai.infrastructure.artifacts import write_json_atomic
from travelmovieai.infrastructure.system import ExecutableStatus
from travelmovieai.web.app import create_app
from travelmovieai.web.jobs import ScanJobManager
from travelmovieai.web.schemas import JobStatus, ScanJobHistory, ScanJobResponse


class FakeScanService:
    def analyze(self, *, input_path: Path, workspace: Path | None) -> StageResult:
        assert workspace is not None
        report = MediaScanReport(
            input_path=input_path,
            scanned_at=datetime.now(UTC),
            discovered_count=2,
            probed_count=2,
        )
        analysis_path = workspace / "artifacts" / "analysis.json"
        write_json_atomic(analysis_path, report)
        return StageResult(
            stage=PipelineStage.MEDIA_SCAN,
            artifacts=[analysis_path],
            message="Media scan found 2 file(s).",
        )

    def resolve_project_paths(self, input_path: Path, workspace: Path | None) -> ProjectPaths:
        resolved_workspace = workspace or Path("workspace") / input_path.name
        return validate_project_paths(input_path, resolved_workspace)


class BlockingScanService(FakeScanService):
    def __init__(self) -> None:
        self.started = Event()
        self.release = Event()

    def analyze(self, *, input_path: Path, workspace: Path | None) -> StageResult:
        self.started.set()
        self.release.wait(timeout=5)
        return super().analyze(input_path=input_path, workspace=workspace)


def test_web_interface_serves_page_and_health() -> None:
    with TestClient(
        create_app(
            job_manager=ScanJobManager(FakeScanService()),
            executable_checker=_available_executable,
        )
    ) as client:
        page = client.get("/")
        health = client.get("/api/health")
        styles = client.get("/static/styles.css")

    assert page.status_code == 200
    assert "TravelMovieAI" in page.text
    assert health.json()["status"] == "ok"
    assert health.json()["ready"] is True
    assert health.json()["ffprobe"]["available"] is True
    assert styles.status_code == 200
    assert "--accent" in styles.text


def test_web_health_is_not_ready_without_ffprobe() -> None:
    def unavailable(name: str) -> ExecutableStatus:
        return ExecutableStatus(
            name=name,
            configured_value=name,
            available=name != "ffprobe",
            error="Исполняемый файл не найден." if name == "ffprobe" else None,
        )

    with TestClient(
        create_app(
            job_manager=ScanJobManager(FakeScanService()),
            executable_checker=unavailable,
        )
    ) as client:
        health = client.get("/api/health")

    assert health.json()["status"] == "degraded"
    assert health.json()["ready"] is False


def test_web_scan_job_reaches_completed_result(tmp_path: Path) -> None:
    media = tmp_path / "Моя поездка"
    media.mkdir()
    workspace = tmp_path / "workspace"

    with TestClient(create_app(job_manager=ScanJobManager(FakeScanService()))) as client:
        response = client.post(
            "/api/scans",
            json={"input_path": str(media), "workspace": str(workspace)},
        )
        assert response.status_code == 202
        job_id = response.json()["id"]

        job = _wait_for_http_job(client, job_id)
        result = client.get(f"/api/scans/{job_id}/result")
        history = client.get("/api/scans")

    assert job["status"] == "completed"
    assert result.status_code == 200
    assert result.json()["discovered_count"] == 2
    assert history.json()["jobs"][0]["id"] == job_id


def test_web_scan_rejects_invalid_paths(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    media = tmp_path / "media"
    media.mkdir()

    with TestClient(create_app(job_manager=ScanJobManager(FakeScanService()))) as client:
        missing_response = client.post(
            "/api/scans",
            json={"input_path": str(missing)},
        )
        unsafe_workspace = client.post(
            "/api/scans",
            json={"input_path": str(media), "workspace": str(tmp_path)},
        )

    assert missing_response.status_code == 400
    assert unsafe_workspace.status_code == 400


def test_job_manager_rejects_active_workspace(tmp_path: Path) -> None:
    media = tmp_path / "media"
    media.mkdir()
    workspace = tmp_path / "workspace"
    service = BlockingScanService()
    manager = ScanJobManager(service)

    first = manager.submit(media, workspace)
    assert service.started.wait(timeout=1)
    with pytest.raises(WorkspaceBusyError):
        manager.submit(media, workspace)

    service.release.set()
    _wait_for_manager_job(manager, first.id)
    manager.shutdown()


def test_web_returns_conflict_for_active_workspace(tmp_path: Path) -> None:
    media = tmp_path / "media"
    media.mkdir()
    workspace = tmp_path / "workspace"
    service = BlockingScanService()
    manager = ScanJobManager(service)

    with TestClient(create_app(job_manager=manager)) as client:
        first = client.post(
            "/api/scans",
            json={"input_path": str(media), "workspace": str(workspace)},
        )
        assert first.status_code == 202
        assert service.started.wait(timeout=1)

        conflict = client.post(
            "/api/scans",
            json={"input_path": str(media), "workspace": str(workspace)},
        )
        service.release.set()
        _wait_for_http_job(client, first.json()["id"])

    assert conflict.status_code == 409
    assert "уже выполняется" in conflict.json()["detail"]


def test_job_history_survives_manager_restart(tmp_path: Path) -> None:
    media = tmp_path / "media"
    media.mkdir()
    workspace = tmp_path / "workspace"
    state_path = tmp_path / "state" / "jobs.json"
    manager = ScanJobManager(FakeScanService(), state_path=state_path)

    submitted = manager.submit(media, workspace)
    completed = _wait_for_manager_job(manager, submitted.id)
    manager.shutdown()

    restored_manager = ScanJobManager(FakeScanService(), state_path=state_path)
    restored = restored_manager.get(submitted.id)

    assert completed.status is JobStatus.COMPLETED
    assert restored is not None
    assert restored.status is JobStatus.COMPLETED
    assert restored_manager.get_report(submitted.id) is not None
    restored_manager.shutdown()


def test_interrupted_job_is_marked_failed_on_restart(tmp_path: Path) -> None:
    media = tmp_path / "media"
    media.mkdir()
    workspace = tmp_path / "workspace"
    state_path = tmp_path / "state" / "jobs.json"
    queued = ScanJobResponse(
        id=UUID("9926dabb-3616-4cec-b87a-b5709e58bd13"),
        status=JobStatus.RUNNING,
        input_path=media,
        workspace=workspace,
        created_at=datetime.now(UTC),
        message="Сканирование медиатеки...",
    )
    write_json_atomic(state_path, ScanJobHistory(jobs=[queued]))

    manager = ScanJobManager(FakeScanService(), state_path=state_path)
    restored = manager.get(queued.id)

    assert restored is not None
    assert restored.status is JobStatus.FAILED
    assert "перезапуском сервера" in restored.message
    manager.shutdown()


def _available_executable(name: str) -> ExecutableStatus:
    return ExecutableStatus(
        name=name,
        configured_value=name,
        available=True,
        resolved_path=Path(f"C:/{name}.exe"),
        version=f"{name} version test",
    )


def _wait_for_http_job(client: TestClient, job_id: str) -> dict[str, object]:
    for _ in range(50):
        response = client.get(f"/api/scans/{job_id}")
        payload: dict[str, object] = response.json()
        if payload["status"] in {"completed", "failed"}:
            return payload
        time.sleep(0.01)
    raise AssertionError("Web scan job did not finish")


def _wait_for_manager_job(manager: ScanJobManager, job_id: UUID) -> ScanJobResponse:
    for _ in range(100):
        job = manager.get(job_id)
        if job and job.status in {JobStatus.COMPLETED, JobStatus.FAILED}:
            return job
        time.sleep(0.01)
    raise AssertionError("Scan job did not finish")
