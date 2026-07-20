"""Disk-capacity estimates for local movie rendering."""

import math
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from travelmovieai.core.exceptions import MontageError
from travelmovieai.domain.models import QuickMontagePlan, QuickMontageSettings

_MEBIBYTE = 1024 * 1024
_H264_BITS_PER_PIXEL = 0.12
_AUDIO_BYTES_PER_SECOND = 32_000
_MINIMUM_MOVIE_BYTES = 16 * _MEBIBYTE
# QP 0 can approach raw YUV420 size; retain headroom for entropy/container overhead.
_LOSSLESS_H264_BYTES_PER_PIXEL_FRAME = 2.0
_ALAC_BYTES_PER_SECOND = 256_000
_MAX_TRANSITION_SEGMENT_OVERHEAD = 1 / 0.55


class DiskUsage(Protocol):
    @property
    def free(self) -> int: ...


@dataclass(frozen=True, slots=True)
class RenderWorkingSetEstimate:
    output_duration_seconds: float
    segment_duration_seconds: float
    active_transition_count: int
    uses_lossless_mezzanine: bool
    estimated_movie_bytes: int
    estimated_final_temporary_bytes: int
    estimated_mezzanine_bytes: int
    estimated_peak_working_set_bytes: int


@dataclass(frozen=True, slots=True)
class RenderDiskSpaceEstimate:
    estimated_movie_bytes: int
    estimated_final_temporary_bytes: int
    estimated_mezzanine_bytes: int
    estimated_peak_working_set_bytes: int
    uses_lossless_mezzanine: bool
    workspace_required_bytes: int
    workspace_available_bytes: int
    output_required_bytes: int
    output_available_bytes: int
    shared_volume: bool


def ensure_render_disk_space(
    *,
    workspace: Path,
    output_path: Path,
    settings: QuickMontageSettings,
    reserve_mb: int,
    safety_factor: float,
    plan: QuickMontagePlan | None = None,
    disk_usage: Callable[[Path], DiskUsage] | None = None,
) -> RenderDiskSpaceEstimate:
    """Fail early when temporary segments and the final MP4 cannot fit safely."""

    usage_provider = disk_usage or _disk_usage
    working_set = estimate_render_working_set(settings, plan=plan)
    estimated_movie_bytes = working_set.estimated_movie_bytes
    reserve_bytes = reserve_mb * _MEBIBYTE

    try:
        workspace_root = _nearest_existing_parent(workspace)
        output_root = _nearest_existing_parent(output_path.parent)
        shared_volume = _volume_id(workspace_root) == _volume_id(output_root)
        if shared_volume:
            legacy_required = math.ceil(estimated_movie_bytes * safety_factor)
            render_required = (
                max(legacy_required, working_set.estimated_peak_working_set_bytes)
                if working_set.uses_lossless_mezzanine
                else legacy_required
            )
            workspace_required = reserve_bytes + render_required
            output_required = workspace_required
        else:
            temporary_factor = max(1.0, safety_factor - 1.0)
            legacy_workspace_required = math.ceil(estimated_movie_bytes * temporary_factor)
            workspace_render_required = (
                max(legacy_workspace_required, working_set.estimated_mezzanine_bytes)
                if working_set.uses_lossless_mezzanine
                else legacy_workspace_required
            )
            output_render_required = (
                max(
                    estimated_movie_bytes + working_set.estimated_final_temporary_bytes,
                    math.ceil(estimated_movie_bytes * safety_factor),
                )
                if working_set.uses_lossless_mezzanine
                else estimated_movie_bytes
            )
            workspace_required = reserve_bytes + workspace_render_required
            output_required = reserve_bytes + output_render_required
        workspace_available = usage_provider(workspace_root).free
        output_available = (
            workspace_available if shared_volume else usage_provider(output_root).free
        )
    except OSError as error:
        raise MontageError("Could not check free disk space before rendering.") from error

    estimate = RenderDiskSpaceEstimate(
        estimated_movie_bytes=estimated_movie_bytes,
        estimated_final_temporary_bytes=working_set.estimated_final_temporary_bytes,
        estimated_mezzanine_bytes=working_set.estimated_mezzanine_bytes,
        estimated_peak_working_set_bytes=working_set.estimated_peak_working_set_bytes,
        uses_lossless_mezzanine=working_set.uses_lossless_mezzanine,
        workspace_required_bytes=workspace_required,
        workspace_available_bytes=workspace_available,
        output_required_bytes=output_required,
        output_available_bytes=output_available,
        shared_volume=shared_volume,
    )
    _validate_available_space(estimate)
    return estimate


def estimate_rendered_movie_bytes(
    settings: QuickMontageSettings,
    *,
    duration_seconds: float | None = None,
) -> int:
    """Estimate a high-quality H.264/AAC movie from duration and pixel rate."""

    width = min(settings.width, 854) if settings.preview_mode else settings.width
    height = min(settings.height, 480) if settings.preview_mode else settings.height
    frame_rate = min(settings.fps, 24) if settings.preview_mode else settings.fps
    video_bytes_per_second = width * height * frame_rate * _H264_BITS_PER_PIXEL / 8
    duration = max(
        0.0,
        settings.target_duration_seconds if duration_seconds is None else duration_seconds,
    )
    estimated = math.ceil(duration * (video_bytes_per_second + _AUDIO_BYTES_PER_SECOND))
    return max(_MINIMUM_MOVIE_BYTES, estimated)


def estimate_render_working_set(
    settings: QuickMontageSettings,
    *,
    plan: QuickMontagePlan | None = None,
) -> RenderWorkingSetEstimate:
    """Estimate renderer intermediates using the same transition policy as FFmpeg."""

    effective_settings = plan.settings if plan is not None else settings
    output_duration = max(
        0.0,
        plan.total_duration_seconds
        if plan is not None
        else effective_settings.target_duration_seconds,
    )
    estimated_movie_bytes = estimate_rendered_movie_bytes(
        effective_settings,
        duration_seconds=output_duration,
    )
    transition_count = _active_transition_count(effective_settings, plan)
    uses_lossless_mezzanine = transition_count > 0
    if plan is not None:
        segment_duration = sum(max(0.0, clip.duration_seconds) for clip in plan.clips)
    elif uses_lossless_mezzanine:
        segment_duration = output_duration * _MAX_TRANSITION_SEGMENT_OVERHEAD
    else:
        segment_duration = output_duration

    estimated_mezzanine_bytes = (
        _estimate_lossless_mezzanine_bytes(effective_settings, segment_duration)
        if uses_lossless_mezzanine
        else 0
    )
    final_temporary_bytes = estimated_movie_bytes
    estimated_peak = (
        estimated_mezzanine_bytes + estimated_movie_bytes + final_temporary_bytes
        if uses_lossless_mezzanine
        else estimated_movie_bytes
    )
    return RenderWorkingSetEstimate(
        output_duration_seconds=output_duration,
        segment_duration_seconds=segment_duration,
        active_transition_count=transition_count,
        uses_lossless_mezzanine=uses_lossless_mezzanine,
        estimated_movie_bytes=estimated_movie_bytes,
        estimated_final_temporary_bytes=final_temporary_bytes,
        estimated_mezzanine_bytes=estimated_mezzanine_bytes,
        estimated_peak_working_set_bytes=estimated_peak,
    )


def _active_transition_count(
    settings: QuickMontageSettings,
    plan: QuickMontagePlan | None,
) -> int:
    if settings.transition == "none" or settings.transition_duration_seconds <= 0:
        return 0
    if plan is None:
        if settings.transition == "cinematic":
            return 0
        estimated_clip_count = max(
            2,
            math.ceil(settings.target_duration_seconds / max(1.0, settings.max_video_clip_seconds)),
        )
        return estimated_clip_count - 1
    if len(plan.clips) < 2:
        return 0
    if settings.transition == "cinematic":
        return sum(clip.transition == "fade" for clip in plan.clips[1:])
    if settings.transition in {"fade", "wipeleft", "slideright"}:
        return len(plan.clips) - 1
    return 0


def _estimate_lossless_mezzanine_bytes(
    settings: QuickMontageSettings,
    duration_seconds: float,
) -> int:
    width = min(settings.width, 854) if settings.preview_mode else settings.width
    height = min(settings.height, 480) if settings.preview_mode else settings.height
    frame_rate = min(settings.fps, 24) if settings.preview_mode else settings.fps
    bytes_per_second = (
        width * height * frame_rate * _LOSSLESS_H264_BYTES_PER_PIXEL_FRAME + _ALAC_BYTES_PER_SECOND
    )
    return math.ceil(max(0.0, duration_seconds) * bytes_per_second)


def _disk_usage(path: Path) -> DiskUsage:
    return shutil.disk_usage(path)


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _volume_id(path: Path) -> int:
    return path.stat().st_dev


def _validate_available_space(estimate: RenderDiskSpaceEstimate) -> None:
    checks = [(estimate.workspace_required_bytes, estimate.workspace_available_bytes)]
    if not estimate.shared_volume:
        checks.append((estimate.output_required_bytes, estimate.output_available_bytes))
    for required, available in checks:
        if available >= required:
            continue
        required_mb = math.ceil(required / _MEBIBYTE)
        available_mb = max(0, math.floor(available / _MEBIBYTE))
        raise MontageError(
            "Not enough free disk space for rendering: "
            f"approximately {required_mb} MiB required, {available_mb} MiB available. "
            "Free disk space or lower the target duration or resolution."
        )
