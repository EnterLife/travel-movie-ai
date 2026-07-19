from pathlib import Path

import pytest

from travelmovieai.application import diagnostics
from travelmovieai.application.diagnostics import run_system_diagnostics
from travelmovieai.core.config import Settings
from travelmovieai.infrastructure.system import CudaStatus, ExecutableStatus


@pytest.fixture(autouse=True)
def available_ffmpeg_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        diagnostics,
        "_ffmpeg_filters",
        lambda _: {"drawtext", "zscale", "tonemap"},
    )


def test_diagnostics_are_ready_with_required_binaries_and_cpu_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(diagnostics, "check_executable", _available_executable)
    monkeypatch.setattr(diagnostics, "check_cuda", lambda _: CudaStatus(available=False))
    monkeypatch.setattr(diagnostics, "find_spec", lambda _: object())

    report = run_system_diagnostics(Settings(device="auto"))

    assert report.ready is True
    assert any(check.name == "CUDA" and check.level == "warning" for check in report.checks)


def test_diagnostics_fail_when_required_ffmpeg_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def executable(binary: str) -> ExecutableStatus:
        return ExecutableStatus(
            name=binary,
            configured_value=binary,
            available=binary != "ffmpeg",
            error="Executable not found." if binary == "ffmpeg" else None,
        )

    monkeypatch.setattr(diagnostics, "check_executable", executable)
    monkeypatch.setattr(diagnostics, "check_cuda", lambda _: CudaStatus(available=False))
    monkeypatch.setattr(diagnostics, "find_spec", lambda _: object())

    report = run_system_diagnostics(Settings())

    assert report.ready is False
    assert any(check.name == "ffmpeg" and check.level == "error" for check in report.checks)


def test_diagnostics_fail_for_configured_missing_embedding_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(diagnostics, "check_executable", _available_executable)
    monkeypatch.setattr(diagnostics, "check_cuda", lambda _: CudaStatus(available=False))
    monkeypatch.setattr(diagnostics, "find_spec", lambda _: None)

    report = run_system_diagnostics(
        Settings(embedding_backend="sentence-transformers", embedding_index="faiss")
    )

    assert report.ready is False
    assert any(check.name == "Embeddings" and check.level == "error" for check in report.checks)
    assert any(check.name == "FAISS" and check.level == "error" for check in report.checks)


def test_diagnostics_validate_enabled_piper_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(diagnostics, "check_executable", _available_executable)
    monkeypatch.setattr(diagnostics, "check_cuda", lambda _: CudaStatus(available=False))
    monkeypatch.setattr(diagnostics, "find_spec", lambda _: object())
    monkeypatch.setattr(diagnostics.shutil, "which", lambda _: None)

    report = run_system_diagnostics(
        Settings(voice_provider="piper", piper_model=tmp_path / "missing.onnx")
    )

    assert report.ready is False
    assert any(check.name == "Piper" and check.level == "error" for check in report.checks)


def test_diagnostics_report_missing_optional_ffmpeg_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(diagnostics, "check_executable", _available_executable)
    monkeypatch.setattr(diagnostics, "check_cuda", lambda _: CudaStatus(available=False))
    monkeypatch.setattr(diagnostics, "find_spec", lambda _: object())
    monkeypatch.setattr(diagnostics, "_ffmpeg_filters", lambda _: {"drawtext"})

    report = run_system_diagnostics(Settings())

    filter_check = next(check for check in report.checks if check.name == "FFmpeg filters")
    assert filter_check.level == "warning"
    assert "tonemap" in filter_check.message
    assert "zscale" in filter_check.message


def test_diagnostics_validate_exact_offline_model_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(diagnostics, "check_executable", _available_executable)
    monkeypatch.setattr(diagnostics, "check_cuda", lambda _: CudaStatus(available=False))
    monkeypatch.setattr(diagnostics, "find_spec", lambda _: object())
    (tmp_path / "unrelated-model").mkdir()

    report = run_system_diagnostics(
        Settings(
            allow_model_download=False,
            model_cache=tmp_path,
            vision_model="private/local-vision",
        )
    )

    model_check = next(check for check in report.checks if check.name == "Model cache")
    assert report.ready is False
    assert model_check.level == "error"
    assert "private/local-vision" in model_check.message


def _available_executable(binary: str) -> ExecutableStatus:
    return ExecutableStatus(
        name=binary,
        configured_value=binary,
        available=True,
        resolved_path=Path(binary),
        version=f"{binary} version 1",
    )
