from collections import namedtuple
from pathlib import Path

import pytest

from travelmovieai.application import disk_space
from travelmovieai.application.disk_space import (
    ensure_render_disk_space,
    estimate_rendered_movie_bytes,
)
from travelmovieai.core.exceptions import MontageError
from travelmovieai.domain.models import QuickMontageSettings

_DiskUsage = namedtuple("_DiskUsage", "total used free")
_GIBIBYTE = 1024**3


def test_render_disk_estimate_grows_with_duration_and_resolution() -> None:
    short_hd = estimate_rendered_movie_bytes(
        QuickMontageSettings(target_duration_seconds=30, width=1280, height=720)
    )
    long_uhd = estimate_rendered_movie_bytes(
        QuickMontageSettings(target_duration_seconds=120, width=3840, height=2160)
    )

    assert long_uhd > short_hd * 20


def test_render_disk_estimate_uses_preview_resolution_cap() -> None:
    settings = QuickMontageSettings(
        target_duration_seconds=120,
        width=3840,
        height=2160,
        fps=60,
    )

    final_size = estimate_rendered_movie_bytes(settings)
    preview_size = estimate_rendered_movie_bytes(settings.model_copy(update={"preview_mode": True}))

    assert preview_size < final_size / 20


def test_render_disk_preflight_accepts_shared_volume_with_capacity(tmp_path: Path) -> None:
    estimate = ensure_render_disk_space(
        workspace=tmp_path / "workspace",
        output_path=tmp_path / "output" / "movie.mp4",
        settings=QuickMontageSettings(target_duration_seconds=120),
        reserve_mb=1024,
        safety_factor=3.0,
        disk_usage=lambda _: _DiskUsage(100 * _GIBIBYTE, 20 * _GIBIBYTE, 80 * _GIBIBYTE),
    )

    assert estimate.shared_volume is True
    assert estimate.workspace_required_bytes > estimate.estimated_movie_bytes
    assert estimate.output_required_bytes == estimate.workspace_required_bytes


def test_render_disk_preflight_rejects_insufficient_space(tmp_path: Path) -> None:
    with pytest.raises(MontageError, match="Not enough free disk space"):
        ensure_render_disk_space(
            workspace=tmp_path / "workspace",
            output_path=tmp_path / "output" / "movie.mp4",
            settings=QuickMontageSettings(target_duration_seconds=3600, width=3840, height=2160),
            reserve_mb=1024,
            safety_factor=3.0,
            disk_usage=lambda _: _DiskUsage(2 * _GIBIBYTE, _GIBIBYTE, _GIBIBYTE),
        )


def test_render_disk_preflight_checks_workspace_and_output_volumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root = tmp_path / "workspace"
    output_root = tmp_path / "output"
    workspace_root.mkdir()
    output_root.mkdir()
    monkeypatch.setattr(
        disk_space,
        "_volume_id",
        lambda path: 1 if path == workspace_root else 2,
    )

    def usage(path: Path) -> _DiskUsage:
        free = 50 * _GIBIBYTE if path == workspace_root else 1
        return _DiskUsage(100 * _GIBIBYTE, 50 * _GIBIBYTE, free)

    with pytest.raises(MontageError, match="0 MiB available"):
        ensure_render_disk_space(
            workspace=workspace_root,
            output_path=output_root / "movie.mp4",
            settings=QuickMontageSettings(target_duration_seconds=60),
            reserve_mb=512,
            safety_factor=3.0,
            disk_usage=usage,
        )


def test_render_disk_preflight_wraps_probe_failure(tmp_path: Path) -> None:
    def failing_usage(_: Path) -> _DiskUsage:
        raise OSError("drive unavailable")

    with pytest.raises(MontageError, match="Could not check free disk space"):
        ensure_render_disk_space(
            workspace=tmp_path / "workspace",
            output_path=tmp_path / "output" / "movie.mp4",
            settings=QuickMontageSettings(),
            reserve_mb=1024,
            safety_factor=3.0,
            disk_usage=failing_usage,
        )
