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
    assert profile.analysis_workers == 13
    assert profile.render_workers == 4
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
    assert profile.render_workers == 6
    assert profile.ffmpeg_threads == 2
    assert profile.model_batch_size == 3
