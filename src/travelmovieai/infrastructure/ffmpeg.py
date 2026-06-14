"""FFmpeg and FFprobe process adapters."""

import json
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from travelmovieai.core.exceptions import DependencyUnavailableError, MediaProbeError

_ISO6709_PATTERN = re.compile(r"^(?P<latitude>[+-]\d+(?:\.\d+)?)(?P<longitude>[+-]\d+(?:\.\d+)?)")


@dataclass(frozen=True, slots=True)
class ProbeResult:
    duration_seconds: float | None = None
    video_duration_seconds: float | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    created_at: datetime | None = None
    latitude: float | None = None
    longitude: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class FFprobeClient:
    def __init__(self, binary: str = "ffprobe", timeout_seconds: float = 60) -> None:
        self.binary = binary
        self.timeout_seconds = timeout_seconds

    def probe(self, path: Path) -> ProbeResult:
        command = [
            self.binary,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                check=False,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_seconds,
            )
        except FileNotFoundError as error:
            raise DependencyUnavailableError(
                f"FFprobe executable was not found: {self.binary}"
            ) from error
        except subprocess.TimeoutExpired as error:
            raise MediaProbeError(
                f"FFprobe timed out after {self.timeout_seconds:g}s for {path.name}"
            ) from error

        if completed.returncode != 0:
            detail = completed.stderr.strip() or "unknown FFprobe error"
            raise MediaProbeError(f"Could not inspect {path.name}: {detail}")

        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise MediaProbeError(f"FFprobe returned invalid JSON for {path.name}") from error

        return parse_probe_payload(payload)


def parse_probe_payload(payload: dict[str, Any]) -> ProbeResult:
    format_data = payload.get("format") or {}
    streams = payload.get("streams") or []
    video_stream = next(
        (stream for stream in streams if stream.get("codec_type") == "video"),
        None,
    )
    tags = _collect_tags(format_data, streams)
    latitude, longitude = _parse_location(tags)

    return ProbeResult(
        duration_seconds=_first_float(
            format_data.get("duration"),
            *(stream.get("duration") for stream in streams),
        ),
        video_duration_seconds=(
            _optional_float(video_stream.get("duration")) if video_stream else None
        ),
        width=_optional_int(video_stream.get("width")) if video_stream else None,
        height=_optional_int(video_stream.get("height")) if video_stream else None,
        fps=_parse_rate(video_stream.get("avg_frame_rate")) if video_stream else None,
        created_at=_parse_datetime(tags.get("creation_time")),
        latitude=latitude,
        longitude=longitude,
        metadata={
            "format_name": format_data.get("format_name"),
            "format_long_name": format_data.get("format_long_name"),
            "bit_rate": _optional_int(format_data.get("bit_rate")),
            "video_duration_seconds": (
                _optional_float(video_stream.get("duration")) if video_stream else None
            ),
            "streams": [
                {
                    "codec_type": stream.get("codec_type"),
                    "codec_name": stream.get("codec_name"),
                    "codec_long_name": stream.get("codec_long_name"),
                }
                for stream in streams
            ],
        },
    )


def _collect_tags(format_data: dict[str, Any], streams: list[dict[str, Any]]) -> dict[str, Any]:
    tags: dict[str, Any] = {}
    for source in [format_data, *streams]:
        for key, value in (source.get("tags") or {}).items():
            tags.setdefault(key.lower(), value)
    return tags


def _parse_location(tags: dict[str, Any]) -> tuple[float | None, float | None]:
    value = (
        tags.get("com.apple.quicktime.location.iso6709")
        or tags.get("location")
        or tags.get("location-eng")
    )
    if not isinstance(value, str):
        return None, None
    match = _ISO6709_PATTERN.match(value)
    if not match:
        return None, None
    return float(match.group("latitude")), float(match.group("longitude"))


def _parse_rate(value: Any) -> float | None:
    if not isinstance(value, str) or value in {"", "0/0"}:
        return None
    if "/" not in value:
        return _optional_float(value)
    numerator, denominator = value.split("/", maxsplit=1)
    denominator_value = _optional_float(denominator)
    if not denominator_value:
        return None
    numerator_value = _optional_float(numerator)
    return numerator_value / denominator_value if numerator_value is not None else None


def _parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _first_float(*values: Any) -> float | None:
    for value in values:
        parsed = _optional_float(value)
        if parsed is not None:
            return parsed
    return None


def _optional_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
