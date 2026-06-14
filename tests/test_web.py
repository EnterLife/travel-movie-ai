import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from travelmovieai.application.validation import (
    ProjectPaths,
    validate_project_paths,
)
from travelmovieai.core.config import Settings
from travelmovieai.core.exceptions import WorkspaceBusyError
from travelmovieai.domain.enums import MediaType, PipelineStage
from travelmovieai.domain.models import (
    MediaAsset,
    MediaScanReport,
    QuickMontageResult,
    QuickMontageSettings,
    Scene,
    StageResult,
)
from travelmovieai.infrastructure.artifacts import write_json_atomic
from travelmovieai.infrastructure.database import MediaAssetRepository
from travelmovieai.infrastructure.lm_studio import LMStudioModels
from travelmovieai.infrastructure.system import (
    CudaStatus,
    ExecutableStatus,
    ResourceProfile,
)
from travelmovieai.web.app import create_app
from travelmovieai.web.jobs import ScanJobManager
from travelmovieai.web.movie_jobs import MovieJobManager, _estimate_phase_eta
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


class FakeMovieService(FakeScanService):
    def get_resource_profile(self) -> ResourceProfile:
        return ResourceProfile(
            logical_cores=16,
            memory_mb=32768,
            gpu_name="RTX Test",
            gpu_memory_mb=12288,
            nvenc=True,
            frame_workers=10,
            analysis_workers=13,
            render_workers=4,
            ffmpeg_threads=4,
            model_batch_size=8,
            summary="16 CPU threads, 32 GB RAM, RTX Test, NVENC",
        )

    def create_quick_montage(
        self,
        *,
        input_path: Path,
        workspace: Path | None,
        settings: QuickMontageSettings,
        output_path: Path | None = None,
        progress: object | None = None,
    ) -> QuickMontageResult:
        assert workspace is not None
        movie_path = workspace / "artifacts" / "final.mp4"
        movie_path.parent.mkdir(parents=True, exist_ok=True)
        movie_path.write_bytes(b"fake mp4")
        timeline_path = workspace / "artifacts" / "quick_timeline.json"
        timeline_path.write_text("{}", encoding="utf-8")
        if callable(progress):
            progress(1, 1, "Фильм готов")
        return QuickMontageResult(
            output_path=movie_path,
            timeline_path=timeline_path,
            clip_count=3,
            duration_seconds=settings.target_duration_seconds,
            selection_mode="semantic" if settings.semantic_analysis else "chronological",
        )


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
    assert "D:\\Vacation\\Japan2026" not in page.text
    assert "Выбрать папку" in page.text
    assert 'class="section-number"' not in page.text
    assert "STAGE 01" not in page.text
    assert "Соберите клип путешествия" in page.text
    assert health.json()["status"] == "ok"
    assert health.json()["ready"] is True
    assert health.json()["ffprobe"]["available"] is True
    assert styles.status_code == 200
    assert "--accent" in styles.text


def test_phase_eta_counts_down_between_progress_updates() -> None:
    started = datetime(2026, 6, 14, 12, 0, tzinfo=UTC)
    last_progress = started + timedelta(seconds=20)

    first = _estimate_phase_eta(
        phase_started_at=started,
        last_progress_at=last_progress,
        now=last_progress,
        phase_start_percent=45,
        phase_end_percent=70,
        last_progress_percent=50,
    )
    later = _estimate_phase_eta(
        phase_started_at=started,
        last_progress_at=last_progress,
        now=last_progress + timedelta(seconds=10),
        phase_start_percent=45,
        phase_end_percent=70,
        last_progress_percent=50,
    )

    assert first == 80
    assert later == 70


def test_web_directory_dialog_returns_selected_path(tmp_path: Path) -> None:
    calls: list[tuple[Path | None, str, bool]] = []

    def select(initial: Path | None, title: str, must_exist: bool) -> Path:
        calls.append((initial, title, must_exist))
        return tmp_path

    with TestClient(
        create_app(
            job_manager=ScanJobManager(FakeScanService()),
            directory_selector=select,
        )
    ) as client:
        response = client.post(
            "/api/dialogs/directory",
            json={"purpose": "input", "initial_path": str(tmp_path)},
        )

    assert response.status_code == 200
    assert response.json()["selected_path"] == str(tmp_path)
    assert calls == [(tmp_path, "Выберите папку с видео и фотографиями", True)]


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


def test_web_capabilities_lists_models_and_cuda() -> None:
    with TestClient(
        create_app(
            job_manager=ScanJobManager(FakeScanService()),
            model_lister=lambda url, key, timeout: LMStudioModels(
                available=True,
                models=("text-model", "vision-omni"),
            ),
            cuda_checker=lambda ffmpeg: CudaStatus(
                available=True,
                gpu_name="RTX Test",
                memory_mb=6144,
                ffmpeg_nvenc=True,
            ),
        )
    ) as client:
        response = client.get("/api/capabilities?include_lm_studio=true")

    payload = response.json()
    assert response.status_code == 200
    assert payload["local_ai"]["resolved_model"] == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert payload["ai"]["available"] is True
    assert payload["ai"]["models"][1]["likely_vision"] is True
    assert payload["ai"]["models"][1]["recommended"] is True
    assert payload["cuda"]["ffmpeg_nvenc"] is True
    assert payload["resources"]["render_workers"] >= 1
    assert payload["resources"]["model_batch_size"] == 4


def test_web_capabilities_skip_lm_studio_by_default() -> None:
    def unexpected_lm_studio_call(
        url: str,
        key: str | None,
        timeout: float,
    ) -> LMStudioModels:
        raise AssertionError("LM Studio should not be contacted")

    with TestClient(
        create_app(
            job_manager=ScanJobManager(FakeScanService()),
            model_lister=unexpected_lm_studio_call,
        )
    ) as client:
        response = client.get("/api/capabilities")

    assert response.status_code == 200
    assert response.json()["ai"]["available"] is False


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


def test_web_movie_job_can_be_downloaded(tmp_path: Path) -> None:
    media = tmp_path / "media"
    media.mkdir()
    workspace = tmp_path / "workspace"

    with TestClient(
        create_app(
            job_manager=ScanJobManager(FakeScanService()),
            movie_job_manager=MovieJobManager(FakeMovieService()),
        )
    ) as client:
        response = client.post(
            "/api/movies",
            json={
                "input_path": str(media),
                "workspace": str(workspace),
                "settings": {
                    "target_duration_seconds": 12,
                    "semantic_analysis": True,
                    "transition": "dissolve",
                },
            },
        )
        assert response.status_code == 202
        job_id = response.json()["id"]
        job = _wait_for_movie_job(client, job_id)
        download = client.get(f"/api/movies/{job_id}/download")

    assert job["status"] == "completed"
    assert job["clip_count"] == 3
    assert job["selection_mode"] == "semantic"
    assert job["progress_percent"] == 100
    assert job["phase"] == "completed"
    assert job["resources"]["render_workers"] == 4
    assert len(job["subtasks"]) == 11
    assert all(
        task["status"] == "completed" for task in job["subtasks"] if task["id"] != "speech_analysis"
    )
    assert (
        next(task for task in job["subtasks"] if task["id"] == "speech_analysis")["status"]
        == "skipped"
    )
    assert len(job["logs"]) >= 3
    assert job["logs"][-1]["message"] == "Фильм готов."
    assert download.status_code == 200
    assert download.content == b"fake mp4"


def test_web_scene_override_is_persisted(tmp_path: Path) -> None:
    media = tmp_path / "media"
    media.mkdir()
    workspace = tmp_path / "workspace"
    repository = MediaAssetRepository(workspace / "project.db")
    repository.initialize()
    asset = MediaAsset(
        path=media / "clip.mp4",
        relative_path=Path("clip.mp4"),
        media_type=MediaType.VIDEO,
        extension=".mp4",
        size_bytes=1,
        modified_at=datetime.now(UTC),
        modified_ns=1,
        duration_seconds=3,
    )
    repository.synchronize([asset], datetime.now(UTC))
    scene = Scene(asset_id=asset.id, start_seconds=0, end_seconds=3)
    repository.synchronize_scenes([scene])

    with TestClient(
        create_app(
            settings=Settings(),
            job_manager=ScanJobManager(FakeScanService()),
        )
    ) as client:
        query = {"input_path": str(media), "workspace": str(workspace)}
        listed = client.get("/api/scenes", params=query)
        updated = client.patch(
            f"/api/scenes/{scene.id}",
            json={**query, "decision": "include"},
        )

    assert listed.status_code == 200
    assert listed.json()["scenes"][0]["id"] == str(scene.id)
    assert updated.status_code == 200
    assert updated.json()["scenes"][0]["metadata"]["selection_override"] == "include"


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


def _wait_for_movie_job(client: TestClient, job_id: str) -> dict[str, object]:
    for _ in range(100):
        response = client.get(f"/api/movies/{job_id}")
        payload: dict[str, object] = response.json()
        if payload["status"] in {"completed", "failed"}:
            return payload
        time.sleep(0.01)
    raise AssertionError("Movie job did not finish")
