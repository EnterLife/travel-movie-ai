"""Disk-capacity estimates for local movie rendering."""

import math
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from travelmovieai.core.exceptions import MontageError
from travelmovieai.domain.models import QuickMontageSettings

_MEBIBYTE = 1024 * 1024
_H264_BITS_PER_PIXEL = 0.12
_AUDIO_BYTES_PER_SECOND = 32_000
_MINIMUM_MOVIE_BYTES = 16 * _MEBIBYTE


class DiskUsage(Protocol):
    @property
    def free(self) -> int: ...


@dataclass(frozen=True, slots=True)
class RenderDiskSpaceEstimate:
    estimated_movie_bytes: int
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
    disk_usage: Callable[[Path], DiskUsage] | None = None,
) -> RenderDiskSpaceEstimate:
    """Fail early when temporary segments and the final MP4 cannot fit safely."""

    usage_provider = disk_usage or _disk_usage
    estimated_movie_bytes = estimate_rendered_movie_bytes(settings)
    reserve_bytes = reserve_mb * _MEBIBYTE

    try:
        workspace_root = _nearest_existing_parent(workspace)
        output_root = _nearest_existing_parent(output_path.parent)
        shared_volume = _volume_id(workspace_root) == _volume_id(output_root)
        if shared_volume:
            workspace_required = reserve_bytes + math.ceil(estimated_movie_bytes * safety_factor)
            output_required = workspace_required
        else:
            temporary_factor = max(1.0, safety_factor - 1.0)
            workspace_required = reserve_bytes + math.ceil(estimated_movie_bytes * temporary_factor)
            output_required = reserve_bytes + estimated_movie_bytes
        workspace_available = usage_provider(workspace_root).free
        output_available = (
            workspace_available if shared_volume else usage_provider(output_root).free
        )
    except OSError as error:
        raise MontageError("Could not check free disk space before rendering.") from error

    estimate = RenderDiskSpaceEstimate(
        estimated_movie_bytes=estimated_movie_bytes,
        workspace_required_bytes=workspace_required,
        workspace_available_bytes=workspace_available,
        output_required_bytes=output_required,
        output_available_bytes=output_available,
        shared_volume=shared_volume,
    )
    _validate_available_space(estimate)
    return estimate


def estimate_rendered_movie_bytes(settings: QuickMontageSettings) -> int:
    """Estimate a high-quality H.264/AAC movie from duration and pixel rate."""

    width = min(settings.width, 854) if settings.preview_mode else settings.width
    height = min(settings.height, 480) if settings.preview_mode else settings.height
    frame_rate = min(settings.fps, 24) if settings.preview_mode else settings.fps
    video_bytes_per_second = width * height * frame_rate * _H264_BITS_PER_PIXEL / 8
    estimated = math.ceil(
        settings.target_duration_seconds * (video_bytes_per_second + _AUDIO_BYTES_PER_SECOND)
    )
    return max(_MINIMUM_MOVIE_BYTES, estimated)


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
