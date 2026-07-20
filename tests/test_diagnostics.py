from pathlib import Path

import pytest

from travelmovieai.application import diagnostics
from travelmovieai.application.diagnostics import run_system_diagnostics
from travelmovieai.core.config import Settings
from travelmovieai.infrastructure.system import CudaStatus, ExecutableStatus, ResourceProfile


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


def test_diagnostics_reject_auto_vision_when_offline_snapshot_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(diagnostics, "check_executable", _available_executable)
    monkeypatch.setattr(diagnostics, "check_cuda", lambda _: CudaStatus(available=False))
    monkeypatch.setattr(diagnostics, "find_spec", lambda _: object())
    monkeypatch.setattr(
        diagnostics,
        "detect_resource_profile",
        lambda *_args, **_kwargs: _resource_profile(memory_mb=16 * 1024),
    )

    report = run_system_diagnostics(
        Settings(allow_model_download=False, model_cache=tmp_path, vision_model="auto")
    )

    model_check = next(check for check in report.checks if check.name == "Model cache")
    assert report.ready is False
    assert model_check.level == "error"
    assert "Qwen2.5-VL-3B" in model_check.message


def test_offline_model_preflight_does_not_require_optional_whisper_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(diagnostics, "check_executable", _available_executable)
    monkeypatch.setattr(diagnostics, "check_cuda", lambda _: CudaStatus(available=False))
    monkeypatch.setattr(diagnostics, "find_spec", lambda _: object())
    monkeypatch.setattr(
        diagnostics,
        "detect_resource_profile",
        lambda *_args, **_kwargs: _resource_profile(memory_mb=16 * 1024),
    )
    vision_snapshot = tmp_path / "Qwen" / "Qwen2.5-VL-3B-Instruct"
    vision_snapshot.mkdir(parents=True)
    (vision_snapshot / "config.json").write_text("{}", encoding="utf-8")
    (vision_snapshot / "model.safetensors").write_bytes(b"weights")

    report = run_system_diagnostics(Settings(allow_model_download=False, model_cache=tmp_path))

    model_check = next(check for check in report.checks if check.name == "Model cache")
    assert model_check.level == "ok"
    assert "whisper" not in model_check.message.casefold()


def test_diagnostics_resolve_auto_vision_for_detected_hardware_tier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(diagnostics, "check_executable", _available_executable)
    monkeypatch.setattr(diagnostics, "check_cuda", lambda _: CudaStatus(available=False))
    monkeypatch.setattr(diagnostics, "find_spec", lambda _: object())
    monkeypatch.setattr(
        diagnostics,
        "detect_resource_profile",
        lambda *_args, **_kwargs: _resource_profile(
            memory_mb=64 * 1024,
            gpu_memory_mb=12 * 1024,
        ),
    )

    report = run_system_diagnostics(
        Settings(allow_model_download=False, model_cache=tmp_path, vision_model="auto")
    )

    model_check = next(check for check in report.checks if check.name == "Model cache")
    assert model_check.level == "error"
    assert "Qwen2.5-VL-7B" in model_check.message


def test_model_snapshot_requires_config_and_weights_in_nested_cache(tmp_path: Path) -> None:
    snapshot = (
        tmp_path / "sentence-transformers" / "models--example--embedding" / "snapshots" / "revision"
    )
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    assert diagnostics.model_snapshot_present(tmp_path, "example/embedding") is False

    (snapshot / "model.safetensors").write_bytes(b"weights")
    assert diagnostics.model_snapshot_present(tmp_path, "example/embedding") is True


@pytest.mark.parametrize("namespace", ["faster-whisper", "story"])
def test_model_snapshot_recognizes_pipeline_cache_namespaces(
    tmp_path: Path,
    namespace: str,
) -> None:
    snapshot = tmp_path / namespace / "models--example--model" / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "model.bin").write_bytes(b"weights")

    assert diagnostics.model_snapshot_present(tmp_path, "example/model") is True


def test_diagnostics_resolve_florence_auto_model_for_offline_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(diagnostics, "check_executable", _available_executable)
    monkeypatch.setattr(diagnostics, "check_cuda", lambda _: CudaStatus(available=False))
    monkeypatch.setattr(diagnostics, "find_spec", lambda _: object())

    report = run_system_diagnostics(
        Settings(
            allow_model_download=False,
            model_cache=tmp_path,
            vision_provider="florence",
            vision_model="auto",
        )
    )

    model_check = next(check for check in report.checks if check.name == "Model cache")
    assert model_check.level == "error"
    assert "microsoft/Florence-2-large" in model_check.message


def _available_executable(binary: str) -> ExecutableStatus:
    return ExecutableStatus(
        name=binary,
        configured_value=binary,
        available=True,
        resolved_path=Path(binary),
        version=f"{binary} version 1",
    )


def _resource_profile(
    *,
    memory_mb: int,
    gpu_memory_mb: int | None = None,
) -> ResourceProfile:
    return ResourceProfile(
        logical_cores=8,
        memory_mb=memory_mb,
        gpu_name="test-gpu" if gpu_memory_mb is not None else None,
        gpu_memory_mb=gpu_memory_mb,
        nvenc=gpu_memory_mb is not None,
        frame_workers=4,
        analysis_workers=4,
        render_workers=1,
        ffmpeg_threads=8,
        model_batch_size=2,
        summary="test resources",
        device="cuda" if gpu_memory_mb is not None else "cpu",
        resource_mode="balanced",
    )
