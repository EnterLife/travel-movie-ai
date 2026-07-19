import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Barrier, BrokenBarrierError, Event
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from travelmovieai.application.service import TravelMovieService
from travelmovieai.application.validation import (
    ProjectPaths,
    validate_project_paths,
)
from travelmovieai.application.workspace_identity import ensure_workspace_identity
from travelmovieai.core.config import Settings
from travelmovieai.core.exceptions import MontageError, TravelMovieError, WorkspaceBusyError
from travelmovieai.domain.enums import MediaType, PipelineStage
from travelmovieai.domain.models import (
    Event as StoryEvent,
)
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
from travelmovieai.infrastructure.system import (
    CudaStatus,
    ExecutableStatus,
    ResourceProfile,
)
from travelmovieai.web.app import create_app
from travelmovieai.web.jobs import ScanJobManager
from travelmovieai.web.movie_jobs import MovieJobManager, _estimate_phase_eta
from travelmovieai.web.schemas import (
    JobStatus,
    MovieJobResponse,
    MovieJobState,
    MovieJobStateHistory,
    ScanJobHistory,
    ScanJobResponse,
)


class FakeScanService:
    def analyze(
        self,
        *,
        input_path: Path,
        workspace: Path | None,
        progress: object | None = None,
    ) -> StageResult:
        assert workspace is not None
        if callable(progress):
            progress(1, 1, "Media scan complete")
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

    def analyze(
        self,
        *,
        input_path: Path,
        workspace: Path | None,
        progress: object | None = None,
    ) -> StageResult:
        self.started.set()
        self.release.wait(timeout=5)
        return super().analyze(input_path=input_path, workspace=workspace, progress=progress)


class InterruptibleScanService(FakeScanService):
    def __init__(self) -> None:
        self.started = Event()
        self.release = Event()
        self.finished = Event()

    def analyze(
        self,
        *,
        input_path: Path,
        workspace: Path | None,
        progress: object | None = None,
    ) -> StageResult:
        self.started.set()
        self.release.wait(timeout=5)
        try:
            return super().analyze(
                input_path=input_path,
                workspace=workspace,
                progress=progress,
            )
        finally:
            self.finished.set()


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
        variant_name: str = "Default",
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
            progress(1, 1, "Film ready")
        return QuickMontageResult(
            output_path=movie_path,
            timeline_path=timeline_path,
            clip_count=3,
            duration_seconds=settings.target_duration_seconds,
            selection_mode="semantic" if settings.semantic_analysis else "chronological",
        )


class ControlledMovieService(FakeMovieService):
    def __init__(self) -> None:
        self.started = Event()
        self.advance = Event()

    def create_quick_montage(
        self,
        *,
        input_path: Path,
        workspace: Path | None,
        settings: QuickMontageSettings,
        variant_name: str = "Default",
        output_path: Path | None = None,
        progress: object | None = None,
    ) -> QuickMontageResult:
        assert workspace is not None
        if callable(progress):
            progress(100, 1000, "Frames: 1/10")
        self.started.set()
        self.advance.wait(timeout=5)
        if callable(progress):
            progress(200, 1000, "Frames: 2/10")
        return super().create_quick_montage(
            input_path=input_path,
            workspace=workspace,
            settings=settings,
            variant_name=variant_name,
            output_path=output_path,
            progress=progress,
        )


class ExternalOutputMovieService(FakeMovieService):
    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path

    def create_quick_montage(
        self,
        *,
        input_path: Path,
        workspace: Path | None,
        settings: QuickMontageSettings,
        variant_name: str = "Default",
        output_path: Path | None = None,
        progress: object | None = None,
    ) -> QuickMontageResult:
        assert workspace is not None
        self.output_path.write_bytes(b"private file")
        timeline_path = workspace / "artifacts" / "quick_timeline.json"
        timeline_path.parent.mkdir(parents=True, exist_ok=True)
        timeline_path.write_text("{}", encoding="utf-8")
        if callable(progress):
            progress(1, 1, "Film ready")
        return QuickMontageResult(
            output_path=self.output_path,
            timeline_path=timeline_path,
            clip_count=1,
            duration_seconds=settings.target_duration_seconds,
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
        script = client.get("/static/app.js")

    assert page.status_code == 200
    assert "TravelMovieAI" in page.text
    assert "D:\\Vacation\\Japan2026" not in page.text
    assert "Browse" in page.text
    assert 'class="section-number"' not in page.text
    assert "STAGE 01" not in page.text
    assert "Create a travel film" in page.text
    assert 'id="music-engine"' in page.text
    assert 'id="music-model"' in page.text
    assert 'id="transition"' in page.text
    assert (
        '<option value="cinematic" selected>Cuts + fade to black · event-aware</option>'
        in page.text
    )
    assert '<option value="fade">Fade through black</option>' in page.text
    assert "Soft dissolve" not in page.text
    assert 'value="soft"' not in page.text
    assert 'id="preserve-chronology"' in page.text
    assert 'id="movie-variant"' in page.text
    assert 'id="version-before"' in page.text
    assert 'id="event-list"' in page.text
    assert 'id="semantic-analysis" type="checkbox">' in page.text
    assert 'id="speech-analysis" type="checkbox">' in page.text
    assert 'id="narration-enabled" type="checkbox">' in page.text
    assert 'id="framing-mode"' in page.text
    assert 'id="vertical-video-layout"' in page.text
    assert 'id="photo-motion"' in page.text
    assert 'id="color-normalization" type="checkbox">' in page.text
    assert 'id="hdr-to-sdr" type="checkbox">' in page.text
    assert 'id="event-titles-enabled" type="checkbox">' in page.text
    assert 'id="scene-subtitles-enabled" type="checkbox">' in page.text
    assert 'id="music-bpm-analysis" type="checkbox">' in page.text
    assert 'id="music-volume-envelope" type="checkbox">' in page.text
    assert 'id="load-more-scenes"' in page.text
    assert 'id="music-volume" type="range" min="0" max="100" value="100"' in page.text
    assert '<span id="music-volume-value">100%</span>' in page.text
    assert "ACE-Step only" in page.text
    assert health.json()["status"] == "ok"
    assert health.json()["ready"] is True
    assert health.json()["ffprobe"]["available"] is True
    assert styles.status_code == 200
    assert "--accent" in styles.text
    assert script.status_code == 200
    assert "transition: transition.value" in script.text
    assert "preserve_chronology: preserveChronology.checked" in script.text
    assert "narration_enabled: narrationEnabled.checked" in script.text
    assert 'workspace.value = ""' in script.text
    assert "framing_mode: framingMode.value" in script.text
    assert "color_normalization: colorNormalization.checked" in script.text
    assert "music_bpm_analysis: musicBpmAnalysis.checked" in script.text
    assert "music_volume_envelope: musicVolumeEnvelope.checked" in script.text
    assert 'query.set("offset", String(scenes.length))' in script.text
    assert 'variant_name: movieVariant.value.trim() || "Default"' in script.text
    assert 'requestJson("/api/events/order"' in script.text
    assert "requestJson(`/api/timeline-versions/compare?${query}`)" in script.text
    assert "FFmpeg not found" in script.text
    assert "Scans ready" in script.text
    assert "The scanner is not ready. Check FFprobe." in script.text


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
    assert calls == [(tmp_path, "Choose a folder with videos and photos", True)]


def test_web_health_is_not_ready_without_ffprobe() -> None:
    def unavailable(name: str) -> ExecutableStatus:
        return ExecutableStatus(
            name=name,
            configured_value=name,
            available=name != "ffprobe",
            error="Executable not found." if name == "ffprobe" else None,
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


def test_web_health_is_not_ready_without_ffmpeg() -> None:
    def unavailable(name: str) -> ExecutableStatus:
        return ExecutableStatus(
            name=name,
            configured_value=name,
            available=name != "ffmpeg",
            error="Executable not found." if name == "ffmpeg" else None,
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
            package_checker=lambda _: True,
            cuda_checker=lambda ffmpeg: CudaStatus(
                available=True,
                gpu_name="RTX Test",
                memory_mb=6144,
                ffmpeg_nvenc=True,
            ),
        )
    ) as client:
        response = client.get("/api/capabilities")

    payload = response.json()
    assert response.status_code == 200
    assert payload["local_ai"]["resolved_model"] == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert payload["music_ai"]["resolved_model"] == "ACE-Step/acestep-v15-turbo"
    assert payload["music_ai"]["available"] is True
    assert payload["local_ai"]["available"] is True
    assert payload["speech"]["available"] is True
    assert payload["narration"]["available"] is False
    assert payload["default_workspace_root"].endswith("workspace")
    assert payload["cuda"]["ffmpeg_nvenc"] is True
    assert payload["resources"]["nvenc"] is True
    assert payload["resources"]["render_workers"] >= 1
    assert payload["resources"]["model_batch_size"] == 2
    assert payload["resources"]["resource_mode"] == "performance"
    assert payload["recommended_render_device"] == "cuda"
    assert payload["recommended_resource_mode"] == "performance"


def test_web_capabilities_recommends_cpu_when_nvenc_is_unavailable() -> None:
    with TestClient(
        create_app(
            job_manager=ScanJobManager(FakeScanService()),
            cuda_checker=lambda ffmpeg: CudaStatus(available=False),
        )
    ) as client:
        payload = client.get("/api/capabilities").json()

    assert payload["recommended_render_device"] == "cpu"
    assert payload["resources"]["device"] == "cpu"


@pytest.mark.parametrize(
    ("settings_payload", "package_checker", "expected_detail"),
    [
        (
            {"semantic_analysis": True},
            lambda _: False,
            "Semantic scene selection is unavailable",
        ),
        (
            {"semantic_analysis": True, "speech_analysis": True},
            lambda package: package != "faster_whisper",
            "Speech recognition is unavailable",
        ),
        (
            {"semantic_analysis": True, "narration_enabled": True},
            lambda _: True,
            "Local narration is unavailable",
        ),
    ],
)
def test_web_movie_rejects_explicit_unavailable_local_capability(
    tmp_path: Path,
    settings_payload: dict[str, bool],
    package_checker: Callable[[str], bool],
    expected_detail: str,
) -> None:
    media = tmp_path / "media"
    media.mkdir()

    with TestClient(
        create_app(
            settings=Settings(voice_provider="disabled"),
            job_manager=ScanJobManager(FakeScanService()),
            movie_job_manager=MovieJobManager(FakeMovieService()),
            package_checker=package_checker,
        )
    ) as client:
        response = client.post(
            "/api/movies",
            json={"input_path": str(media), "settings": settings_payload},
        )

    assert response.status_code == 422
    assert expected_detail in response.json()["detail"]


def test_web_movie_rejects_speech_without_semantic_pipeline(tmp_path: Path) -> None:
    media = tmp_path / "media"
    media.mkdir()

    with TestClient(
        create_app(
            job_manager=ScanJobManager(FakeScanService()),
            movie_job_manager=MovieJobManager(FakeMovieService()),
            package_checker=lambda _: True,
        )
    ) as client:
        response = client.post(
            "/api/movies",
            json={
                "input_path": str(media),
                "settings": {"semantic_analysis": False, "speech_analysis": True},
            },
        )

    assert response.status_code == 422
    assert "requires semantic scene selection" in response.json()["detail"]


@pytest.mark.parametrize(
    "settings_payload",
    [
        {"framing_mode": "smart"},
        {"color_normalization": True},
        {"event_titles_enabled": True},
        {"scene_subtitles_enabled": True},
    ],
)
def test_web_movie_rejects_semantic_visual_features_in_quick_mode(
    tmp_path: Path,
    settings_payload: dict[str, object],
) -> None:
    media = tmp_path / "media"
    media.mkdir()
    with TestClient(
        create_app(
            job_manager=ScanJobManager(FakeScanService()),
            movie_job_manager=MovieJobManager(FakeMovieService()),
            package_checker=lambda _: True,
        )
    ) as client:
        response = client.post(
            "/api/movies",
            json={"input_path": str(media), "settings": settings_payload},
        )

    assert response.status_code == 422
    assert "require semantic scene analysis" in response.json()["detail"]


def test_web_rejects_non_loopback_host_and_mutating_origin() -> None:
    with TestClient(create_app(job_manager=ScanJobManager(FakeScanService()))) as client:
        hostile_host = client.get("/api/health", headers={"Host": "attacker.example"})
        hostile_origin = client.post(
            "/api/scans",
            json={},
            headers={"Origin": "https://attacker.example"},
        )
        loopback_ipv6 = client.get("/api/health", headers={"Host": "[::1]:8000"})

    assert hostile_host.status_code == 400
    assert hostile_origin.status_code == 403
    assert loopback_ipv6.status_code == 200


def test_web_movie_rejects_explicit_cuda_without_nvenc(tmp_path: Path) -> None:
    media = tmp_path / "media"
    media.mkdir()

    with TestClient(
        create_app(
            job_manager=ScanJobManager(FakeScanService()),
            movie_job_manager=MovieJobManager(FakeMovieService()),
            cuda_checker=lambda _: CudaStatus(available=False),
        )
    ) as client:
        response = client.post(
            "/api/movies",
            json={
                "input_path": str(media),
                "settings": {"render_device": "cuda"},
            },
        )

    assert response.status_code == 422
    assert "CUDA rendering is unavailable" in response.json()["detail"]
    assert "Choose Auto or CPU" in response.json()["detail"]


def test_web_accepts_configured_local_narration_capability(tmp_path: Path) -> None:
    media = tmp_path / "media"
    media.mkdir()
    voice_model = tmp_path / "voice.onnx"
    voice_model.write_bytes(b"model")

    with TestClient(
        create_app(
            settings=Settings(
                voice_provider="piper",
                piper_binary="piper",
                piper_model=voice_model,
            ),
            job_manager=ScanJobManager(FakeScanService()),
            movie_job_manager=MovieJobManager(FakeMovieService()),
            package_checker=lambda _: True,
            executable_checker=_available_executable,
        )
    ) as client:
        capabilities = client.get("/api/capabilities")
        created = client.post(
            "/api/movies",
            json={
                "input_path": str(media),
                "workspace": str(tmp_path / "workspace"),
                "settings": {
                    "semantic_analysis": True,
                    "narration_enabled": True,
                },
            },
        )

    assert capabilities.status_code == 200
    assert capabilities.json()["narration"]["available"] is True
    assert created.status_code == 202


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
        nested_workspace = client.post(
            "/api/scans",
            json={"input_path": str(media), "workspace": str(media / "workspace")},
        )

    assert missing_response.status_code == 400
    assert unsafe_workspace.status_code == 400
    assert nested_workspace.status_code == 400


def test_web_movie_job_can_be_downloaded(tmp_path: Path) -> None:
    media = tmp_path / "media"
    media.mkdir()
    workspace = tmp_path / "workspace"

    with TestClient(
        create_app(
            job_manager=ScanJobManager(FakeScanService()),
            movie_job_manager=MovieJobManager(FakeMovieService()),
            package_checker=lambda _: True,
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
                    "transition": "cinematic",
                },
            },
        )
        assert response.status_code == 202
        job_id = response.json()["id"]
        job = _wait_for_movie_job(client, job_id)
        history = client.get("/api/movies?limit=1")
        download = client.get(f"/api/movies/{job_id}/download")

    assert job["status"] == "completed"
    assert job["clip_count"] == 3
    assert job["selection_mode"] == "semantic"
    assert job["quality_score"] is None
    assert job["quality_issue_count"] == 0
    assert job["progress_percent"] == 100
    assert job["phase"] == "completed"
    assert job["resources"]["render_workers"] == 4
    assert len(job["subtasks"]) == len(PipelineStage)
    assert all(
        task["status"] == "completed" for task in job["subtasks"] if task["status"] != "skipped"
    )
    assert (
        next(task for task in job["subtasks"] if task["id"] == "speech_analysis")["status"]
        == "skipped"
    )
    assert {task["id"] for task in job["subtasks"] if task["status"] == "skipped"} == {
        "speech_analysis",
        "narration",
        "voice_synthesis",
    }
    assert len(job["logs"]) >= 3
    assert job["logs"][-1]["message"] == "Film ready."
    assert history.status_code == 200
    assert history.json()["jobs"][0]["id"] == job_id
    assert download.status_code == 200
    assert download.content == b"fake mp4"


@pytest.mark.parametrize("transition", ["dissolve", "soft"])
def test_web_movie_request_rejects_prohibited_transition(
    tmp_path: Path,
    transition: str,
) -> None:
    media = tmp_path / "media"
    media.mkdir()

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
                "settings": {"transition": transition},
            },
        )

    assert response.status_code == 422
    assert any(
        error["loc"] == ["body", "settings", "transition"] for error in response.json()["detail"]
    )


def test_web_movie_download_rejects_output_outside_workspace(tmp_path: Path) -> None:
    media = tmp_path / "media"
    media.mkdir()
    workspace = tmp_path / "workspace"
    secret = tmp_path / "secret.mp4"

    with TestClient(
        create_app(
            job_manager=ScanJobManager(FakeScanService()),
            movie_job_manager=MovieJobManager(ExternalOutputMovieService(secret)),
        )
    ) as client:
        response = client.post(
            "/api/movies",
            json={
                "input_path": str(media),
                "workspace": str(workspace),
                "settings": {"target_duration_seconds": 12},
            },
        )
        assert response.status_code == 202
        job_id = response.json()["id"]
        job = _wait_for_movie_job(client, job_id)
        download = client.get(f"/api/movies/{job_id}/download")

    assert job["status"] == "completed"
    assert secret.read_bytes() == b"private file"
    assert download.status_code == 403
    assert download.json()["detail"] == "Invalid movie output path."


def test_web_movie_download_reports_missing_rendered_file(tmp_path: Path) -> None:
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
                "settings": {"target_duration_seconds": 12},
            },
        )
        assert response.status_code == 202
        job_id = response.json()["id"]
        job = _wait_for_movie_job(client, job_id)
        assert isinstance(job["output_path"], str)
        Path(job["output_path"]).unlink()
        download = client.get(f"/api/movies/{job_id}/download")

    assert job["status"] == "completed"
    assert download.status_code == 404
    assert download.json()["detail"] == "Rendered movie file not found."


def test_movie_job_can_pause_resume_and_cancel(tmp_path: Path) -> None:
    media = tmp_path / "media"
    media.mkdir()
    service = ControlledMovieService()
    manager = MovieJobManager(service)
    submitted = manager.submit(
        media,
        tmp_path / "workspace",
        QuickMontageSettings(),
    )
    assert service.started.wait(timeout=2)

    paused = manager.pause(submitted.id)
    assert paused is not None
    assert paused.status is JobStatus.PAUSED
    service.advance.set()
    time.sleep(0.05)
    assert manager.get(submitted.id).status is JobStatus.PAUSED  # type: ignore[union-attr]

    resumed = manager.resume(submitted.id)
    assert resumed is not None
    assert resumed.status is JobStatus.RUNNING
    completed = _wait_for_manager_movie_job(manager, submitted.id)
    assert completed.status is JobStatus.COMPLETED
    manager.shutdown()

    cancel_service = ControlledMovieService()
    cancel_manager = MovieJobManager(cancel_service)
    cancel_job = cancel_manager.submit(
        media,
        tmp_path / "cancel-workspace",
        QuickMontageSettings(),
    )
    assert cancel_service.started.wait(timeout=2)
    cancelled = cancel_manager.cancel(cancel_job.id)
    assert cancelled is not None
    assert cancelled.status is JobStatus.CANCELLED
    cancel_service.advance.set()
    time.sleep(0.1)
    assert cancel_manager.get(cancel_job.id).status is JobStatus.CANCELLED  # type: ignore[union-attr]
    cancel_manager.shutdown()


def test_quick_movie_cancel_is_observed_inside_media_scan(tmp_path: Path) -> None:
    class ScanBoundaryService(TravelMovieService):
        def __init__(self) -> None:
            super().__init__(Settings())
            self.started = Event()
            self.release = Event()
            self.received_progress = False

        def analyze(
            self,
            *,
            input_path: Path,
            workspace: Path | None,
            progress: Callable[[int, int, str], None] | None = None,
        ) -> StageResult:
            del input_path, workspace
            self.received_progress = callable(progress)
            assert progress is not None
            progress(0, 2, "Media scan: 0/2")
            self.started.set()
            self.release.wait(timeout=5)
            progress(1, 2, "Media scan: 1/2")
            raise AssertionError("cancel callback should interrupt the quick scan")

    media = tmp_path / "media"
    media.mkdir()
    service = ScanBoundaryService()
    manager = MovieJobManager(
        service,
        render_disk_reserve_mb=0,
        render_disk_safety_factor=1,
    )
    submitted = manager.submit(media, tmp_path / "workspace", QuickMontageSettings())
    assert service.started.wait(timeout=2)

    cancelled = manager.cancel(submitted.id)
    service.release.set()
    time.sleep(0.1)

    assert cancelled is not None and cancelled.status is JobStatus.CANCELLED
    assert service.received_progress is True
    assert manager.get(submitted.id).status is JobStatus.CANCELLED  # type: ignore[union-attr]
    manager.shutdown()


def test_semantic_movie_reports_canonical_mid_pipeline_subtask(tmp_path: Path) -> None:
    class MidPipelineService(FakeMovieService):
        def __init__(self) -> None:
            self.reached = Event()
            self.release = Event()

        def create_quick_montage(
            self,
            *,
            input_path: Path,
            workspace: Path | None,
            settings: QuickMontageSettings,
            variant_name: str = "Default",
            output_path: Path | None = None,
            progress: object | None = None,
        ) -> QuickMontageResult:
            assert callable(progress)
            progress(8500, 18000, "Starting duplicate detection")
            self.reached.set()
            self.release.wait(timeout=5)
            return super().create_quick_montage(
                input_path=input_path,
                workspace=workspace,
                settings=settings,
                variant_name=variant_name,
                output_path=output_path,
                progress=progress,
            )

    media = tmp_path / "media"
    media.mkdir()
    service = MidPipelineService()
    manager = MovieJobManager(
        service,
        render_disk_reserve_mb=0,
        render_disk_safety_factor=1,
    )
    submitted = manager.submit(
        media,
        tmp_path / "workspace",
        QuickMontageSettings(semantic_analysis=True),
    )
    assert service.reached.wait(timeout=2)
    running = manager.get(submitted.id)
    assert running is not None
    subtasks = {task.id: task for task in running.subtasks}

    assert running.phase == "duplicate_detection"
    assert subtasks["embeddings"].status == "completed"
    assert subtasks["duplicate_detection"].status == "running"
    assert 0 < subtasks["duplicate_detection"].progress_percent < 100
    assert subtasks["event_detection"].status == "pending"

    service.release.set()
    assert _wait_for_manager_movie_job(manager, submitted.id).status is JobStatus.COMPLETED
    manager.shutdown()


def test_web_scene_override_is_persisted(tmp_path: Path) -> None:
    media = tmp_path / "media"
    media.mkdir()
    workspace = tmp_path / "workspace"
    ensure_workspace_identity(media, workspace)
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


def test_web_scene_pagination_reaches_and_edits_scenes_after_first_120(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media = tmp_path / "media"
    media.mkdir()
    workspace = tmp_path / "workspace"
    ensure_workspace_identity(media, workspace)
    repository = MediaAssetRepository(workspace / "project.db")
    repository.initialize()
    asset = MediaAsset(
        path=media / "long-clip.mp4",
        relative_path=Path("long-clip.mp4"),
        media_type=MediaType.VIDEO,
        extension=".mp4",
        size_bytes=1,
        modified_at=datetime.now(UTC),
        modified_ns=1,
        duration_seconds=130,
    )
    repository.synchronize([asset], datetime.now(UTC))
    thumbnail = workspace / "frames" / "scene-125.jpg"
    thumbnail.parent.mkdir(parents=True)
    thumbnail.write_bytes(b"jpeg")
    scenes = [
        Scene(
            asset_id=asset.id,
            start_seconds=float(index),
            end_seconds=float(index + 1),
            keyframe_path=thumbnail if index == 125 else None,
        )
        for index in range(130)
    ]
    repository.synchronize_scenes(scenes)
    event = StoryEvent(title="Late event", scene_ids=[scene.id for scene in scenes[120:]])
    repository.synchronize_events([event])

    def reject_full_scene_load(_: MediaAssetRepository) -> list[Scene]:
        raise AssertionError("thumbnail endpoint must not load every scene")

    monkeypatch.setattr(MediaAssetRepository, "list_scenes", reject_full_scene_load)
    query = {"input_path": str(media), "workspace": str(workspace)}
    with TestClient(
        create_app(settings=Settings(), job_manager=ScanJobManager(FakeScanService()))
    ) as client:
        first = client.get("/api/scenes", params={**query, "limit": 120})
        late = client.get(
            "/api/scenes",
            params={**query, "offset": 120, "limit": 10},
        )
        filtered = client.get(
            "/api/scenes",
            params={**query, "event_id": str(event.id), "limit": 5},
        )
        updated = client.patch(
            f"/api/scenes/{scenes[125].id}",
            json={**query, "decision": "include"},
        )
        thumbnail_response = client.get(
            f"/api/scenes/{scenes[125].id}/thumbnail",
            params=query,
        )
        invalid_offset = client.get("/api/scenes", params={**query, "offset": -1})
        invalid_limit = client.get("/api/scenes", params={**query, "limit": 501})
        missing_event = client.get(
            "/api/scenes",
            params={
                **query,
                "event_id": "00000000-0000-0000-0000-000000000001",
            },
        )

    assert first.status_code == 200
    assert first.json()["total"] == 130
    assert len(first.json()["scenes"]) == 120
    assert late.status_code == 200
    assert late.json()["offset"] == 120
    assert late.json()["limit"] == 10
    assert late.json()["scenes"][5]["id"] == str(scenes[125].id)
    assert filtered.status_code == 200
    assert filtered.json()["total"] == 10
    assert filtered.json()["scenes"][0]["id"] == str(scenes[120].id)
    assert updated.status_code == 200
    assert updated.json()["scenes"][0]["metadata"]["selection_override"] == "include"
    assert thumbnail_response.status_code == 200
    assert thumbnail_response.content == b"jpeg"
    assert invalid_offset.status_code == 422
    assert invalid_limit.status_code == 422
    assert missing_event.status_code == 404


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
    assert "already queued or running" in conflict.json()["detail"]


def test_scan_and_movie_submissions_atomically_reserve_a_workspace(tmp_path: Path) -> None:
    media = tmp_path / "media"
    media.mkdir()
    workspace = tmp_path / "workspace"
    cross_check_gate = Barrier(2)
    active_snapshot_gate = Barrier(2)
    request_gate = Barrier(3)
    scan_service = BlockingScanService()
    movie_service = ControlledMovieService()

    class GatedScanJobManager(ScanJobManager):
        def is_workspace_active(self, candidate: Path) -> bool:
            with suppress(BrokenBarrierError):
                cross_check_gate.wait(timeout=0.25)
            active = super().is_workspace_active(candidate)
            with suppress(BrokenBarrierError):
                active_snapshot_gate.wait(timeout=0.25)
            return active

    class GatedMovieJobManager(MovieJobManager):
        def is_workspace_active(self, candidate: Path) -> bool:
            with suppress(BrokenBarrierError):
                cross_check_gate.wait(timeout=0.25)
            active = super().is_workspace_active(candidate)
            with suppress(BrokenBarrierError):
                active_snapshot_gate.wait(timeout=0.25)
            return active

    scan_manager = GatedScanJobManager(scan_service)
    movie_manager = GatedMovieJobManager(
        movie_service,
        render_disk_reserve_mb=0,
        render_disk_safety_factor=1,
    )
    app = create_app(
        job_manager=scan_manager,
        movie_job_manager=movie_manager,
        package_checker=lambda _: True,
    )

    with TestClient(app) as client:

        def post_scan():
            request_gate.wait(timeout=2)
            return client.post(
                "/api/scans",
                json={"input_path": str(media), "workspace": str(workspace)},
            )

        def post_movie():
            request_gate.wait(timeout=2)
            return client.post(
                "/api/movies",
                json={
                    "input_path": str(media),
                    "workspace": str(workspace),
                    "settings": {
                        "target_duration_seconds": 5,
                        "music_enabled": False,
                    },
                },
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            scan_future = executor.submit(post_scan)
            movie_future = executor.submit(post_movie)
            request_gate.wait(timeout=2)
            scan_response = scan_future.result(timeout=3)
            movie_response = movie_future.result(timeout=3)

        scan_service.release.set()
        movie_service.advance.set()
        if scan_response.status_code == 202:
            _wait_for_manager_job(scan_manager, UUID(scan_response.json()["id"]))
        if movie_response.status_code == 202:
            _wait_for_manager_movie_job(movie_manager, UUID(movie_response.json()["id"]))

    assert sorted((scan_response.status_code, movie_response.status_code)) == [202, 409]


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


def test_interrupted_scan_job_resumes_same_job_after_restart(tmp_path: Path) -> None:
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
        message="Scanning media...",
    )
    write_json_atomic(state_path, ScanJobHistory(jobs=[queued]))

    manager = ScanJobManager(FakeScanService(), state_path=state_path)
    restored = _wait_for_manager_job(manager, queued.id)

    assert restored is not None
    assert restored.status is JobStatus.COMPLETED
    assert restored.id == queued.id
    assert manager.get_report(queued.id) is not None
    manager.shutdown()


def test_scan_shutdown_keeps_active_job_recoverable(tmp_path: Path) -> None:
    media = tmp_path / "media"
    media.mkdir()
    workspace = tmp_path / "workspace"
    state_path = tmp_path / "state" / "jobs.json"
    service = InterruptibleScanService()
    manager = ScanJobManager(service, state_path=state_path)

    submitted = manager.submit(media, workspace)
    assert service.started.wait(timeout=2)
    manager.shutdown()
    service.release.set()
    assert service.finished.wait(timeout=2)

    restored_manager = ScanJobManager(FakeScanService(), state_path=state_path)
    restored = _wait_for_manager_job(restored_manager, submitted.id)

    assert restored.status is JobStatus.COMPLETED
    assert restored.id == submitted.id
    restored_manager.shutdown()


def test_scan_job_redacts_credentials_from_persisted_failure(tmp_path: Path) -> None:
    class FailingScanService(FakeScanService):
        def analyze(
            self,
            *,
            input_path: Path,
            workspace: Path | None,
            progress: object | None = None,
        ) -> StageResult:
            del input_path, workspace, progress
            raise TravelMovieError("probe failed api_key=private-value")

    media = tmp_path / "media"
    media.mkdir()
    state_path = tmp_path / "state" / "jobs.json"
    manager = ScanJobManager(FailingScanService(), state_path=state_path)

    submitted = manager.submit(media, tmp_path / "workspace")
    failed = _wait_for_manager_job(manager, submitted.id)
    persisted = state_path.read_text(encoding="utf-8")

    assert failed.status is JobStatus.FAILED
    assert "private-value" not in persisted
    assert "<redacted>" in persisted
    manager.shutdown()


def test_movie_job_history_survives_manager_restart(tmp_path: Path) -> None:
    media = tmp_path / "media"
    media.mkdir()
    workspace = tmp_path / "workspace"
    state_path = tmp_path / "state" / "movie_jobs.json"
    manager = MovieJobManager(
        FakeMovieService(),
        state_path=state_path,
        render_disk_reserve_mb=0,
        render_disk_safety_factor=1.0,
    )

    submitted = manager.submit(media, workspace, QuickMontageSettings())
    completed = _wait_for_manager_movie_job(manager, submitted.id)
    manager.shutdown()

    restored_manager = MovieJobManager(
        FakeMovieService(),
        state_path=state_path,
        render_disk_reserve_mb=0,
        render_disk_safety_factor=1.0,
    )
    restored = restored_manager.get(submitted.id)

    assert completed.status is JobStatus.COMPLETED
    assert restored is not None
    assert restored.status is JobStatus.COMPLETED
    assert restored.output_path == workspace / "artifacts" / "final.mp4"
    assert restored_manager.output_path(submitted.id) == restored.output_path
    assert [job.id for job in restored_manager.list()] == [submitted.id]
    restored_manager.shutdown()


def test_interrupted_movie_job_resumes_same_job_after_restart(tmp_path: Path) -> None:
    media = tmp_path / "media"
    media.mkdir()
    workspace = tmp_path / "workspace"
    state_path = tmp_path / "state" / "movie_jobs.json"
    running = MovieJobState(
        id=UUID("b3ff9226-a51c-407d-b5ec-df98c7040b25"),
        status=JobStatus.RUNNING,
        input_path=media,
        workspace=workspace,
        settings=QuickMontageSettings(),
        created_at=datetime.now(UTC),
        started_at=datetime.now(UTC),
        phase="rendering",
        message="Rendering",
    )
    write_json_atomic(state_path, MovieJobStateHistory(jobs=[running]))

    class CountingMovieService(FakeMovieService):
        def __init__(self) -> None:
            self.calls = 0

        def create_quick_montage(self, **kwargs: object) -> QuickMontageResult:
            self.calls += 1
            return super().create_quick_montage(**kwargs)

    service = CountingMovieService()
    manager = MovieJobManager(
        service,
        state_path=state_path,
        render_disk_reserve_mb=0,
        render_disk_safety_factor=1.0,
    )
    restored = _wait_for_manager_movie_job(manager, running.id)

    assert restored.status is JobStatus.COMPLETED
    assert restored.id == running.id
    assert restored.error is None
    assert service.calls == 1
    assert any("Recovered interrupted edit" in entry.message for entry in restored.logs)
    assert manager.is_workspace_active(workspace) is False
    manager.shutdown()


def test_paused_movie_job_stays_paused_after_restart_then_resumes(tmp_path: Path) -> None:
    media = tmp_path / "media"
    media.mkdir()
    workspace = tmp_path / "workspace"
    state_path = tmp_path / "state" / "movie_jobs.json"
    paused = MovieJobState(
        id=UUID("a9866155-3336-4029-976c-1361ae5f214e"),
        status=JobStatus.PAUSED,
        input_path=media,
        workspace=workspace,
        settings=QuickMontageSettings(),
        created_at=datetime.now(UTC),
        started_at=datetime.now(UTC),
        phase="vision_analysis",
        message="Paused",
    )
    write_json_atomic(state_path, MovieJobStateHistory(jobs=[paused]))

    manager = MovieJobManager(
        FakeMovieService(),
        state_path=state_path,
        render_disk_reserve_mb=0,
        render_disk_safety_factor=1.0,
    )
    restored = manager.get(paused.id)

    assert restored is not None
    assert restored.status is JobStatus.PAUSED
    assert manager.is_workspace_active(workspace) is True
    resumed = manager.resume(paused.id)
    completed = _wait_for_manager_movie_job(manager, paused.id)

    assert resumed is not None
    assert completed.status is JobStatus.COMPLETED
    assert completed.id == paused.id
    manager.shutdown()


def test_movie_job_history_is_bounded_to_configured_limit(tmp_path: Path) -> None:
    media = tmp_path / "media"
    workspace = tmp_path / "workspace"
    state_path = tmp_path / "state" / "movie_jobs.json"
    now = datetime.now(UTC)
    states = [
        MovieJobState(
            id=UUID(f"00000000-0000-0000-0000-{index:012d}"),
            status=JobStatus.FAILED,
            input_path=media,
            workspace=workspace,
            settings=QuickMontageSettings(),
            created_at=now - timedelta(minutes=3 - index),
            finished_at=now,
        )
        for index in range(1, 4)
    ]
    write_json_atomic(state_path, MovieJobStateHistory(jobs=states))

    manager = MovieJobManager(FakeMovieService(), state_path=state_path, history_limit=2)
    restored_ids = [job.id for job in manager.list(limit=10)]

    assert restored_ids == [states[2].id, states[1].id]
    persisted = MovieJobStateHistory.model_validate_json(state_path.read_text(encoding="utf-8"))
    assert len(persisted.jobs) == 2
    manager.shutdown()


def test_movie_job_redacts_credentials_from_persisted_failure(tmp_path: Path) -> None:
    class FailingMovieService(FakeMovieService):
        def create_quick_montage(
            self,
            *,
            input_path: Path,
            workspace: Path | None,
            settings: QuickMontageSettings,
            variant_name: str = "Default",
            output_path: Path | None = None,
            progress: object | None = None,
        ) -> QuickMontageResult:
            del input_path, workspace, settings, variant_name, output_path, progress
            raise TravelMovieError("provider failed token=top-secret Authorization: Bearer abc123")

    media = tmp_path / "media"
    media.mkdir()
    state_path = tmp_path / "state" / "movie_jobs.json"
    manager = MovieJobManager(
        FailingMovieService(),
        state_path=state_path,
        render_disk_reserve_mb=0,
        render_disk_safety_factor=1.0,
    )

    submitted = manager.submit(media, tmp_path / "workspace", QuickMontageSettings())
    failed = _wait_for_manager_movie_job(manager, submitted.id)
    persisted = state_path.read_text(encoding="utf-8")

    assert failed.status is JobStatus.FAILED
    assert "top-secret" not in persisted
    assert "abc123" not in persisted
    assert persisted.count("<redacted>") >= 2
    manager.shutdown()


def test_movie_job_fails_before_service_when_disk_preflight_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media = tmp_path / "media"
    media.mkdir()
    workspace = tmp_path / "workspace"

    def reject_capacity(**_: object) -> None:
        raise MontageError("Not enough free disk space for rendering.")

    monkeypatch.setattr(
        "travelmovieai.web.movie_jobs.ensure_render_disk_space",
        reject_capacity,
    )
    manager = MovieJobManager(FakeMovieService())

    submitted = manager.submit(media, workspace, QuickMontageSettings())
    failed = _wait_for_manager_movie_job(manager, submitted.id)

    assert failed.status is JobStatus.FAILED
    assert "Not enough free disk space" in (failed.error or "")
    assert not (workspace / "artifacts" / "final.mp4").exists()
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


def _wait_for_manager_movie_job(
    manager: MovieJobManager,
    job_id: UUID,
) -> MovieJobResponse:
    for _ in range(100):
        job = manager.get(job_id)
        if job and job.status in {
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        }:
            return job
        time.sleep(0.01)
    raise AssertionError("Movie job did not finish")
