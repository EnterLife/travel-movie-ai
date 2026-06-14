"""Scene detection and representative frame sampling."""

import hashlib
import importlib
import json
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from travelmovieai.core.exceptions import (
    DependencyUnavailableError,
    MediaProbeError,
    MontageError,
)
from travelmovieai.domain.enums import MediaType
from travelmovieai.domain.models import MediaAsset, QuickMontageSettings, Scene
from travelmovieai.infrastructure.ffmpeg import FFprobeClient


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
            )
        except Exception:
            return self._uniform_scenes(asset, settings), True

        ranges = [
            (start.get_seconds(), end.get_seconds())
            for start, end in boundaries
            if end.get_seconds() > start.get_seconds()
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
                    scenes[-1] = previous.model_copy(update={"end_seconds": range_end})
                    break
                scenes.append(
                    Scene(
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
    ) -> None:
        self.ffmpeg_binary = ffmpeg_binary
        self.use_cuda_decode = use_cuda_decode
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
        frame_path = frames_dir / f"{scene.id}-contact-v3.png"
        if frame_path.is_file() and frame_path.stat().st_size > 0:
            return frame_path
        temporary_path = frame_path.with_name(
            f".{frame_path.stem}.{uuid4().hex}.tmp.png"
        )
        timestamps = _sample_timestamps(
            scene,
            asset,
            self._video_duration(asset),
        )
        command = self._command(asset, timestamps, temporary_path, use_cuda=False)
        commands = (
            [self._command(asset, timestamps, temporary_path, use_cuda=True), command]
            if self.use_cuda_decode
            else [command]
        )
        completed = None
        try:
            for candidate in commands:
                temporary_path.unlink(missing_ok=True)
                try:
                    completed = subprocess.run(
                        candidate,
                        capture_output=True,
                        check=False,
                        encoding="utf-8",
                        errors="replace",
                    )
                except FileNotFoundError as error:
                    raise DependencyUnavailableError(
                        f"FFmpeg executable was not found: {self.ffmpeg_binary}"
                    ) from error
                if completed.returncode == 0 and temporary_path.is_file():
                    with self._backend_lock:
                        if self.use_cuda_decode and candidate is commands[0]:
                            self._nvdec_count += 1
                        else:
                            self._cpu_count += 1
                    os.replace(temporary_path, frame_path)
                    return frame_path
            detail = completed.stderr.strip() if completed is not None else ""
            if not detail:
                return_code = completed.returncode if completed is not None else "unknown"
                detail = (
                    f"FFmpeg завершился с кодом {return_code}, "
                    "но не создал изображение."
                )
            raise MontageError(
                f"Не удалось извлечь кадр из {asset.relative_path}: {detail}"
            )
        finally:
            temporary_path.unlink(missing_ok=True)

    def _command(
        self,
        asset: MediaAsset,
        timestamps: tuple[float, float, float],
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
        command.extend(
            [
                "-filter_complex",
                f"[0:v]{download}scale=480:270:force_original_aspect_ratio=decrease,"
                "pad=480:270:(ow-iw)/2:(oh-ih)/2:black[a];"
                f"[1:v]{download}scale=480:270:force_original_aspect_ratio=decrease,"
                "pad=480:270:(ow-iw)/2:(oh-ih)/2:black[b];"
                f"[2:v]{download}scale=480:270:force_original_aspect_ratio=decrease,"
                "pad=480:270:(ow-iw)/2:(oh-ih)/2:black[c];"
                "[a][b][c]hstack=inputs=3,format=rgb24[v]",
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
            stored = _optional_positive_float(
                asset.probe_metadata.get("video_duration_seconds")
            )
            if stored is not None:
                self._video_durations[asset.path] = stored
                return stored
            try:
                duration = self._probe.probe(asset.path).video_duration_seconds
            except (DependencyUnavailableError, MediaProbeError):
                duration = None
            self._video_durations[asset.path] = duration
            return duration


def scene_cache_key(asset: MediaAsset, settings: QuickMontageSettings) -> str:
    payload = {
        "asset_id": str(asset.id),
        "size": asset.size_bytes,
        "modified_ns": asset.modified_ns,
        "threshold": settings.scene_threshold,
        "min": settings.min_scene_duration_seconds,
        "max": settings.max_scene_duration_seconds,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _sample_timestamps(
    scene: Scene,
    asset: MediaAsset,
    video_duration: float | None,
) -> tuple[float, float, float]:
    end = scene.end_seconds
    if video_duration is not None:
        frame_margin = max(0.05, 1 / (asset.fps or 25))
        end = min(end, max(scene.start_seconds, video_duration - frame_margin))
    duration = max(0, end - scene.start_seconds)
    if duration <= 0:
        timestamp = max(0, min(scene.start_seconds, (video_duration or end) - 0.05))
        return timestamp, timestamp, timestamp
    return (
        scene.start_seconds + duration * 0.12,
        scene.start_seconds + duration * 0.5,
        scene.start_seconds + duration * 0.88,
    )


def _optional_positive_float(value: object) -> float | None:
    if not isinstance(value, (int, float, str)):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None
