from collections import namedtuple
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from travelmovieai.application import disk_space
from travelmovieai.application.disk_space import (
    ensure_render_disk_space,
    estimate_render_working_set,
    estimate_rendered_movie_bytes,
)
from travelmovieai.core.exceptions import MontageError
from travelmovieai.domain.enums import MediaType
from travelmovieai.domain.models import (
    MontageClip,
    QuickMontagePlan,
    QuickMontageSettings,
)

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


def test_transition_working_set_accounts_for_lossless_segments_and_final_temp(
    tmp_path: Path,
) -> None:
    plan = _transition_plan(tmp_path, transition="fade")

    working_set = estimate_render_working_set(plan.settings, plan=plan)

    assert working_set.uses_lossless_mezzanine is True
    assert working_set.active_transition_count == 1
    assert working_set.segment_duration_seconds == pytest.approx(60.5)
    assert working_set.estimated_mezzanine_bytes > working_set.estimated_movie_bytes * 100
    assert working_set.estimated_peak_working_set_bytes == (
        working_set.estimated_mezzanine_bytes
        + working_set.estimated_movie_bytes
        + working_set.estimated_final_temporary_bytes
    )


def test_lossless_mezzanine_estimate_scales_with_resolution_fps_and_duration() -> None:
    small = estimate_render_working_set(
        QuickMontageSettings(
            target_duration_seconds=30,
            width=640,
            height=360,
            fps=15,
            transition="fade",
        )
    )
    large = estimate_render_working_set(
        QuickMontageSettings(
            target_duration_seconds=60,
            width=1280,
            height=720,
            fps=30,
            transition="fade",
        )
    )

    assert small.uses_lossless_mezzanine is True
    assert large.estimated_mezzanine_bytes > small.estimated_mezzanine_bytes * 12


def test_transition_preflight_rejects_space_that_only_fits_delivery_movie(
    tmp_path: Path,
) -> None:
    plan = _transition_plan(tmp_path, transition="fade")
    working_set = estimate_render_working_set(plan.settings, plan=plan)
    legacy_requirement = working_set.estimated_movie_bytes * 3
    assert working_set.estimated_peak_working_set_bytes > legacy_requirement

    with pytest.raises(MontageError, match="Not enough free disk space"):
        ensure_render_disk_space(
            workspace=tmp_path / "workspace",
            output_path=tmp_path / "output" / "movie.mp4",
            settings=plan.settings,
            plan=plan,
            reserve_mb=0,
            safety_factor=3.0,
            disk_usage=lambda _: _DiskUsage(
                working_set.estimated_peak_working_set_bytes,
                1,
                working_set.estimated_peak_working_set_bytes - 1,
            ),
        )


def test_transition_preflight_accepts_exact_shared_volume_boundary(tmp_path: Path) -> None:
    plan = _transition_plan(tmp_path, transition="fade")
    working_set = estimate_render_working_set(plan.settings, plan=plan)
    available = working_set.estimated_peak_working_set_bytes

    estimate = ensure_render_disk_space(
        workspace=tmp_path / "workspace",
        output_path=tmp_path / "output" / "movie.mp4",
        settings=plan.settings,
        plan=plan,
        reserve_mb=0,
        safety_factor=1.0,
        disk_usage=lambda _: _DiskUsage(available, 0, available),
    )

    assert estimate.uses_lossless_mezzanine is True
    assert estimate.workspace_required_bytes == available
    assert estimate.output_required_bytes == available


def test_cinematic_hard_cut_plan_keeps_legacy_preflight_size(tmp_path: Path) -> None:
    plan = _transition_plan(tmp_path, transition="cinematic", clip_transition="cut")
    working_set = estimate_render_working_set(plan.settings, plan=plan)
    legacy_requirement = working_set.estimated_movie_bytes * 3

    estimate = ensure_render_disk_space(
        workspace=tmp_path / "workspace",
        output_path=tmp_path / "output" / "movie.mp4",
        settings=plan.settings,
        plan=plan,
        reserve_mb=0,
        safety_factor=3.0,
        disk_usage=lambda _: _DiskUsage(legacy_requirement, 0, legacy_requirement),
    )

    assert working_set.uses_lossless_mezzanine is False
    assert working_set.estimated_mezzanine_bytes == 0
    assert estimate.workspace_required_bytes == legacy_requirement


def test_transition_preflight_splits_mezzanine_and_final_temp_across_volumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    output = tmp_path / "output"
    workspace.mkdir()
    output.mkdir()
    plan = _transition_plan(tmp_path, transition="fade")
    working_set = estimate_render_working_set(plan.settings, plan=plan)
    monkeypatch.setattr(disk_space, "_volume_id", lambda path: 1 if path == workspace else 2)

    def usage(path: Path) -> _DiskUsage:
        free = (
            working_set.estimated_mezzanine_bytes
            if path == workspace
            else working_set.estimated_movie_bytes * 2
        )
        return _DiskUsage(free, 0, free)

    estimate = ensure_render_disk_space(
        workspace=workspace,
        output_path=output / "movie.mp4",
        settings=plan.settings,
        plan=plan,
        reserve_mb=0,
        safety_factor=1.0,
        disk_usage=usage,
    )

    assert estimate.shared_volume is False
    assert estimate.workspace_required_bytes == working_set.estimated_mezzanine_bytes
    assert estimate.output_required_bytes == working_set.estimated_movie_bytes * 2


def _transition_plan(
    tmp_path: Path,
    *,
    transition: str,
    clip_transition: str | None = "fade",
) -> QuickMontagePlan:
    settings = QuickMontageSettings(
        target_duration_seconds=60,
        width=1920,
        height=1080,
        fps=30,
        transition=transition,
        transition_duration_seconds=0.5,
    )
    clips = [
        MontageClip(
            asset_id=uuid4(),
            source_path=tmp_path / f"clip-{index}.mp4",
            relative_path=Path(f"clip-{index}.mp4"),
            media_type=MediaType.VIDEO,
            duration_seconds=30.25,
            transition=clip_transition if index else None,
        )
        for index in range(2)
    ]
    return QuickMontagePlan(
        created_at=datetime.now(UTC),
        settings=settings,
        clips=clips,
        total_duration_seconds=60,
    )
