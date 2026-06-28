from travelmovieai.infrastructure import system
from travelmovieai.infrastructure.system import CudaStatus, detect_resource_profile


def test_resource_profile_uses_available_cpu_memory_and_gpu(monkeypatch) -> None:
    monkeypatch.setattr(system.os, "cpu_count", lambda: 16)
    monkeypatch.setattr(system, "_system_memory_mb", lambda: 32 * 1024)

    profile = detect_resource_profile(
        cuda=CudaStatus(
            available=True,
            gpu_name="RTX Test",
            memory_mb=12 * 1024,
            ffmpeg_nvenc=True,
        )
    )

    assert profile.frame_workers == 10
    assert profile.analysis_workers == 10
    assert profile.render_workers == 1
    assert profile.ffmpeg_threads == 4
    assert profile.model_batch_size == 8
    assert "NVENC" in profile.summary


def test_resource_profile_honors_manual_overrides(monkeypatch) -> None:
    monkeypatch.setattr(system.os, "cpu_count", lambda: 12)
    monkeypatch.setattr(system, "_system_memory_mb", lambda: 16 * 1024)

    profile = detect_resource_profile(
        cuda=CudaStatus(available=False),
        worker_override=6,
        batch_override=3,
    )

    assert profile.frame_workers == 6
    assert profile.analysis_workers == 6
    assert profile.render_workers == 2
    assert profile.ffmpeg_threads == 4
    assert profile.model_batch_size == 3


def test_resource_profile_caps_high_manual_render_override(monkeypatch) -> None:
    monkeypatch.setattr(system.os, "cpu_count", lambda: 24)
    monkeypatch.setattr(system, "_system_memory_mb", lambda: 64 * 1024)

    profile = detect_resource_profile(
        cuda=CudaStatus(available=True, ffmpeg_nvenc=True),
        worker_override=20,
    )

    assert profile.frame_workers == 12
    assert profile.analysis_workers == 12
    assert profile.render_workers == 2
    assert profile.ffmpeg_threads == 4


def test_six_gb_gpu_uses_safe_two_scene_batch(monkeypatch) -> None:
    monkeypatch.setattr(system.os, "cpu_count", lambda: 16)
    monkeypatch.setattr(system, "_system_memory_mb", lambda: 32 * 1024)

    profile = detect_resource_profile(
        cuda=CudaStatus(
            available=True,
            gpu_name="RTX 3060",
            memory_mb=6 * 1024,
            ffmpeg_nvenc=True,
        )
    )

    assert profile.model_batch_size == 2
    assert "vision batch 2" in profile.summary


def test_high_memory_workstation_uses_more_analysis_workers(monkeypatch) -> None:
    monkeypatch.setattr(system.os, "cpu_count", lambda: 32)
    monkeypatch.setattr(system, "_system_memory_mb", lambda: 64 * 1024)

    profile = detect_resource_profile(
        cuda=CudaStatus(
            available=True,
            gpu_name="RTX Workstation",
            memory_mb=16 * 1024,
            ffmpeg_nvenc=True,
        )
    )

    assert profile.frame_workers == 12
    assert profile.analysis_workers == 12
    assert profile.model_batch_size == 16
