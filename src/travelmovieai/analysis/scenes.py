"""Scene detection and representative frame sampling."""

import hashlib
import hmac
import importlib
import json
import math
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from PIL import Image

from travelmovieai.core.exceptions import (
    DependencyUnavailableError,
    MediaProbeError,
    MontageError,
)
from travelmovieai.core.security import sanitize_process_error
from travelmovieai.domain.enums import MediaType
from travelmovieai.domain.models import MediaAsset, QuickMontageSettings, Scene
from travelmovieai.infrastructure.ffmpeg import FFprobeClient

_SCENE_ID_NAMESPACE = uuid5(NAMESPACE_URL, "https://travelmovieai.local/scene/v1")
_SCENE_IDENTITY_VERSION = "deterministic-scene-id-v3-start-in-scene"
CONTACT_SHEET_SCHEMA_VERSION = "contact-sheet-v1-temporal"
_CONTACT_SHEET_PANEL_WIDTH = 480
_CONTACT_SHEET_PANEL_HEIGHT = 270


class SceneDetector:
    """Use PySceneDetect when installed and a deterministic fallback otherwise."""

    def detect(
        self,
        asset: MediaAsset,
        settings: QuickMontageSettings,
    ) -> tuple[list[Scene], bool]:
        if asset.media_type is MediaType.PHOTO:
            return [
                Scene(
                    id=_scene_id(
                        asset,
                        0,
                        settings.photo_duration_seconds,
                    ),
                    asset_id=asset.id,
                    start_seconds=0,
                    end_seconds=settings.photo_duration_seconds,
                    metadata={
                        "detector": "photo",
                        "cache_key": scene_cache_key(asset, settings),
                    },
                )
            ], False

        try:
            scenedetect: Any = importlib.import_module("scenedetect")
        except ImportError:
            return self._uniform_scenes(asset, settings), True

        try:
            boundaries = scenedetect.detect(
                str(asset.path),
                scenedetect.ContentDetector(
                    threshold=settings.scene_threshold,
                    min_scene_len=max(
                        1,
                        round(settings.min_scene_duration_seconds * (asset.fps or 30)),
                    ),
                ),
                show_progress=False,
                start_in_scene=True,
            )
        except Exception:
            return self._uniform_scenes(asset, settings), True

        ranges = [
            (start.seconds, end.seconds) for start, end in boundaries if end.seconds > start.seconds
        ]
        if not ranges:
            return self._uniform_scenes(asset, settings), True
        return self._split_long_ranges(asset, ranges, settings, "pyscenedetect"), False

    def _uniform_scenes(
        self,
        asset: MediaAsset,
        settings: QuickMontageSettings,
    ) -> list[Scene]:
        duration = asset.duration_seconds or 0
        if duration <= 0:
            return []
        ranges: list[tuple[float, float]] = []
        start = 0.0
        while start < duration:
            end = min(duration, start + settings.max_scene_duration_seconds)
            ranges.append((start, end))
            start = end
        return self._split_long_ranges(asset, ranges, settings, "uniform")

    def _split_long_ranges(
        self,
        asset: MediaAsset,
        ranges: Sequence[tuple[float, float]],
        settings: QuickMontageSettings,
        detector: str,
    ) -> list[Scene]:
        scenes: list[Scene] = []
        cache_key = scene_cache_key(asset, settings)
        for range_start, range_end in ranges:
            start = max(0.0, range_start)
            while start < range_end:
                end = min(range_end, start + settings.max_scene_duration_seconds)
                if end - start < settings.min_scene_duration_seconds and scenes:
                    previous = scenes[-1]
                    scenes[-1] = previous.model_copy(
                        update={
                            "id": _scene_id(asset, previous.start_seconds, range_end),
                            "end_seconds": range_end,
                        }
                    )
                    break
                scenes.append(
                    Scene(
                        id=_scene_id(asset, start, end),
                        asset_id=asset.id,
                        start_seconds=start,
                        end_seconds=end,
                        metadata={"detector": detector, "cache_key": cache_key},
                    )
                )
                start = end
        return scenes


class RepresentativeFrameExtractor:
    def __init__(
        self,
        ffmpeg_binary: str = "ffmpeg",
        ffprobe_binary: str = "ffprobe",
        use_cuda_decode: bool = False,
        frame_sample_count: int = 3,
        timeout_seconds: float = 120,
    ) -> None:
        self.ffmpeg_binary = ffmpeg_binary
        self.use_cuda_decode = use_cuda_decode
        self.frame_sample_count = frame_sample_count_for_mode(
            "fast" if frame_sample_count <= 3 else "balanced" if frame_sample_count <= 5 else "deep"
        )
        self.timeout_seconds = timeout_seconds
        self._probe = FFprobeClient(ffprobe_binary)
        self._video_durations: dict[Path, float | None] = {}
        self._duration_lock = Lock()
        self._backend_lock = Lock()
        self._nvdec_count = 0
        self._cpu_count = 0

    @property
    def backend_summary(self) -> str:
        with self._backend_lock:
            return f"NVDEC={self._nvdec_count}, CPU fallback={self._cpu_count}"

    def extract(self, scene: Scene, asset: MediaAsset, frames_dir: Path) -> Path:
        if asset.media_type is MediaType.PHOTO:
            return asset.path

        frames_dir.mkdir(parents=True, exist_ok=True)
        proxy_fingerprint = asset.probe_metadata.get("analysis_proxy_fingerprint")
        proxy_variant = f"-proxy-{str(proxy_fingerprint)[:12]}" if proxy_fingerprint else ""
        frame_path = (
            frames_dir / f"{scene.id}-contact-v4-{self.frame_sample_count}{proxy_variant}.png"
        )
        cached_metadata = scene.metadata.get("contact_sheet")
        if isinstance(cached_metadata, dict) and contact_sheet_file_valid(
            frame_path,
            cached_metadata,
            expected_sample_count=self.frame_sample_count,
        ):
            return frame_path
        frame_path.unlink(missing_ok=True)
        temporary_path = frame_path.with_name(f".{frame_path.stem}.{uuid4().hex}.tmp.png")
        timestamps = _sample_timestamps(
            scene,
            asset,
            self._video_duration(asset),
            self.frame_sample_count,
        )
        command = self._command(asset, timestamps, temporary_path, use_cuda=False)
        commands = (
            [self._command(asset, timestamps, temporary_path, use_cuda=True), command]
            if self.use_cuda_decode
            else [command]
        )
        completed = None
        timed_out_backends: list[str] = []
        try:
            for candidate in commands:
                temporary_path.unlink(missing_ok=True)
                backend = "NVDEC" if self.use_cuda_decode and candidate is commands[0] else "CPU"
                try:
                    completed = subprocess.run(
                        candidate,
                        capture_output=True,
                        check=False,
                        encoding="utf-8",
                        errors="replace",
                        timeout=self.timeout_seconds,
                    )
                except FileNotFoundError as error:
                    raise DependencyUnavailableError(
                        f"FFmpeg executable was not found: {self.ffmpeg_binary}"
                    ) from error
                except subprocess.TimeoutExpired:
                    timed_out_backends.append(backend)
                    continue
                if completed.returncode == 0 and _generated_contact_sheet_valid(
                    temporary_path,
                    self.frame_sample_count,
                ):
                    with self._backend_lock:
                        if self.use_cuda_decode and candidate is commands[0]:
                            self._nvdec_count += 1
                        else:
                            self._cpu_count += 1
                    os.replace(temporary_path, frame_path)
                    return frame_path
            if len(timed_out_backends) == len(commands):
                tried = ", ".join(timed_out_backends)
                raise MontageError(
                    f"FFmpeg timed out after {self.timeout_seconds:g}s while extracting "
                    f"frames from {asset.relative_path} ({tried})."
                )
            detail = (
                sanitize_process_error(
                    completed.stderr,
                    private_paths=[asset.path, temporary_path, frame_path],
                    fallback="unknown FFmpeg error",
                )
                if completed is not None
                else ""
            )
            if not detail:
                return_code = completed.returncode if completed is not None else "unknown"
                detail = f"FFmpeg exited with code {return_code}, but did not create an image."
            raise MontageError(f"Could not extract a frame from {asset.relative_path}: {detail}")
        finally:
            temporary_path.unlink(missing_ok=True)

    def sampling_metadata(
        self,
        scene: Scene,
        asset: MediaAsset,
        image_path: Path,
    ) -> dict[str, object]:
        """Describe the exact chronological samples represented by an extracted sheet."""

        timestamps: tuple[float, ...]
        positions: tuple[float, ...]
        if asset.media_type is MediaType.PHOTO:
            timestamps = (0.0,)
            positions = (0.5,)
            columns = rows = 1
        else:
            timestamps = tuple(
                round(timestamp, 3)
                for timestamp in _sample_timestamps(
                    scene,
                    asset,
                    self._video_duration(asset),
                    self.frame_sample_count,
                )
            )
            duration = max(0.0, scene.end_seconds - scene.start_seconds)
            positions = (
                tuple(
                    max(0.0, min(1.0, (timestamp - scene.start_seconds) / duration))
                    for timestamp in timestamps
                )
                if duration > 0
                else tuple(0.5 for _ in timestamps)
            )
            columns = min(3, len(timestamps))
            rows = math.ceil(len(timestamps) / columns)
        return {
            "schema_version": CONTACT_SHEET_SCHEMA_VERSION,
            "sample_count": len(positions),
            "sample_positions": list(positions),
            "sample_timestamps_seconds": list(timestamps),
            "columns": columns,
            "rows": rows,
            "content_sha256": _file_sha256(image_path),
        }

    def _command(
        self,
        asset: MediaAsset,
        timestamps: tuple[float, ...],
        temporary_path: Path,
        *,
        use_cuda: bool,
    ) -> list[str]:
        command = [
            self.ffmpeg_binary,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
        ]
        for timestamp in timestamps:
            command.extend(["-ss", f"{timestamp:.3f}"])
            if use_cuda:
                command.extend(["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"])
            command.extend(["-i", str(asset.path)])
        download = "hwdownload,format=nv12," if use_cuda else ""
        filters = []
        labels = []
        for index in range(len(timestamps)):
            label = f"f{index}"
            labels.append(f"[{label}]")
            filters.append(
                f"[{index}:v]{download}scale=480:270:force_original_aspect_ratio=decrease,"
                f"pad=480:270:(ow-iw)/2:(oh-ih)/2:black[{label}]"
            )
        layout = "|".join(
            f"{(index % 3) * 480}_{(index // 3) * 270}" for index in range(len(timestamps))
        )
        command.extend(
            [
                "-filter_complex",
                ";".join(
                    [
                        *filters,
                        (
                            f"{''.join(labels)}xstack=inputs={len(labels)}:"
                            f"layout={layout}:fill=black,format=rgb24[v]"
                        ),
                    ]
                ),
                "-map",
                "[v]",
                "-an",
                "-sn",
                "-frames:v",
                "1",
                "-c:v",
                "png",
                "-compression_level",
                "3",
                "-threads",
                "1",
                str(temporary_path),
            ]
        )
        return command

    def _video_duration(self, asset: MediaAsset) -> float | None:
        if asset.path in self._video_durations:
            return self._video_durations[asset.path]
        with self._duration_lock:
            if asset.path in self._video_durations:
                return self._video_durations[asset.path]
            stored = _optional_positive_float(asset.probe_metadata.get("video_duration_seconds"))
            if stored is not None:
                self._video_durations[asset.path] = stored
                return stored
            try:
                duration = self._probe.probe(asset.path).video_duration_seconds
            except (DependencyUnavailableError, MediaProbeError):
                duration = None
            self._video_durations[asset.path] = duration
            return duration


def scene_cache_key(
    asset: MediaAsset,
    settings: QuickMontageSettings,
    *,
    analysis_fingerprint: str | None = None,
) -> str:
    payload = {
        "identity_version": _SCENE_IDENTITY_VERSION,
        "asset_id": str(asset.id),
        "size": asset.size_bytes,
        "modified_ns": asset.modified_ns,
        "threshold": _canonical_cache_number(settings.scene_threshold),
        "min": _canonical_cache_number(settings.min_scene_duration_seconds),
        "max": _canonical_cache_number(settings.max_scene_duration_seconds),
        "analysis_fingerprint": analysis_fingerprint
        or asset.probe_metadata.get("scene_analysis_fingerprint"),
    }
    if asset.media_type is MediaType.PHOTO:
        payload["photo_duration"] = _canonical_cache_number(settings.photo_duration_seconds)
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _canonical_cache_number(value: float) -> int | float:
    numeric = float(value)
    return int(numeric) if numeric.is_integer() else numeric


def _scene_id(asset: MediaAsset, start_seconds: float, end_seconds: float) -> UUID:
    start_microseconds = round(start_seconds * 1_000_000)
    end_microseconds = round(end_seconds * 1_000_000)
    return uuid5(
        _SCENE_ID_NAMESPACE,
        (
            f"{asset.relative_path.as_posix()}:{asset.media_type.value}:"
            f"{asset.size_bytes}:{asset.modified_ns}:"
            f"{start_microseconds}:{end_microseconds}"
        ),
    )


def _sample_timestamps(
    scene: Scene,
    asset: MediaAsset,
    video_duration: float | None,
    count: int = 3,
) -> tuple[float, ...]:
    end = scene.end_seconds
    if video_duration is not None:
        frame_margin = max(0.05, 1 / (asset.fps or 25))
        end = min(end, max(scene.start_seconds, video_duration - frame_margin))
    duration = max(0, end - scene.start_seconds)
    if duration <= 0:
        timestamp = max(0, min(scene.start_seconds, (video_duration or end) - 0.05))
        return tuple(timestamp for _ in range(count))
    return tuple(scene.start_seconds + duration * position for position in _sample_positions(count))


def frame_sample_count_for_mode(mode: str) -> int:
    if mode == "fast":
        return 3
    if mode == "deep":
        return 9
    return 5


def sample_positions_for_count(count: int) -> tuple[float, ...]:
    """Return the canonical sampling positions for supported contact-sheet sizes."""

    if count <= 1:
        return (0.5,)
    return _sample_positions(count)


def _sample_positions(count: int) -> tuple[float, ...]:
    if count <= 3:
        return (0.12, 0.5, 0.88)
    if count <= 5:
        return (0.08, 0.3, 0.5, 0.7, 0.92)
    return (0.06, 0.17, 0.29, 0.4, 0.5, 0.6, 0.71, 0.83, 0.94)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contact_sheet_file_valid(
    path: Path,
    metadata: dict[str, object],
    *,
    expected_sample_count: int | None = None,
) -> bool:
    """Verify that cached contact-sheet metadata still describes the file on disk."""

    sample_count = metadata.get("sample_count")
    columns = metadata.get("columns")
    rows = metadata.get("rows")
    sample_positions = metadata.get("sample_positions")
    sample_timestamps = metadata.get("sample_timestamps_seconds")
    expected_digest = metadata.get("content_sha256")
    if (
        metadata.get("schema_version") != CONTACT_SHEET_SCHEMA_VERSION
        or not isinstance(sample_count, int)
        or isinstance(sample_count, bool)
        or sample_count < 1
        or (expected_sample_count is not None and sample_count != expected_sample_count)
        or not isinstance(columns, int)
        or isinstance(columns, bool)
        or columns < 1
        or not isinstance(rows, int)
        or isinstance(rows, bool)
        or rows < 1
        or not isinstance(sample_positions, list)
        or len(sample_positions) != sample_count
        or not isinstance(sample_timestamps, list)
        or len(sample_timestamps) != sample_count
        or not isinstance(expected_digest, str)
        or len(expected_digest) != 64
        or any(character not in "0123456789abcdef" for character in expected_digest)
    ):
        return False
    if not path.is_file():
        return False

    expected_columns = min(3, sample_count)
    expected_rows = math.ceil(sample_count / expected_columns)
    if columns != expected_columns or rows != expected_rows:
        return False
    try:
        with Image.open(path) as image:
            image_format = image.format if isinstance(image.format, str) else None
            image_size = (int(image.width), int(image.height))
            image.load()
        actual_digest = _file_sha256(path)
    except (OSError, ValueError):
        return False
    if not hmac.compare_digest(actual_digest, expected_digest):
        return False
    if sample_count > 1:
        return image_format == "PNG" and image_size == (
            columns * _CONTACT_SHEET_PANEL_WIDTH,
            rows * _CONTACT_SHEET_PANEL_HEIGHT,
        )
    return image_size[0] > 0 and image_size[1] > 0


def _generated_contact_sheet_valid(path: Path, sample_count: int) -> bool:
    if not path.is_file():
        return False
    columns = min(3, sample_count)
    rows = math.ceil(sample_count / columns)
    try:
        with Image.open(path) as image:
            image_format = image.format if isinstance(image.format, str) else None
            image_size = (int(image.width), int(image.height))
            image.load()
    except (OSError, ValueError):
        return False
    return image_format == "PNG" and image_size == (
        columns * _CONTACT_SHEET_PANEL_WIDTH,
        rows * _CONTACT_SHEET_PANEL_HEIGHT,
    )


def _optional_positive_float(value: object) -> float | None:
    if not isinstance(value, (int, float, str)):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None
