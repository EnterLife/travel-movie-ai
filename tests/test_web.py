import time
from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient

from travelmovieai.domain.enums import PipelineStage
from travelmovieai.domain.models import MediaScanReport, StageResult
from travelmovieai.infrastructure.artifacts import write_json_atomic
from travelmovieai.web.app import create_app
from travelmovieai.web.jobs import ScanJobManager


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

    def resolve_workspace(self, input_path: Path, workspace: Path | None) -> Path:
        return (workspace or Path("workspace") / input_path.name).resolve()


def test_web_interface_serves_page_and_health() -> None:
    with TestClient(create_app(job_manager=ScanJobManager(FakeScanService()))) as client:
        page = client.get("/")
        health = client.get("/api/health")
        styles = client.get("/static/styles.css")

    assert page.status_code == 200
    assert "TravelMovieAI" in page.text
    assert health.json() == {"status": "ok", "service": "travelmovieai"}
    assert styles.status_code == 200
    assert "--accent" in styles.text


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

        job = _wait_for_job(client, job_id)
        result = client.get(f"/api/scans/{job_id}/result")

    assert job["status"] == "completed"
    assert result.status_code == 200
    assert result.json()["discovered_count"] == 2


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


def _wait_for_job(client: TestClient, job_id: str) -> dict[str, object]:
    for _ in range(50):
        response = client.get(f"/api/scans/{job_id}")
        payload: dict[str, object] = response.json()
        if payload["status"] in {"completed", "failed"}:
            return payload
        time.sleep(0.01)
    raise AssertionError("Web scan job did not finish")
