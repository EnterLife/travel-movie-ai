"""Scene detection and representative frame sampling."""

import hashlib
import importlib
import json
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

from travelmovieai.core.exceptions import DependencyUnavailableError, MontageError
from travelmovieai.domain.enums import MediaType
from travelmovieai.domain.models import MediaAsset, QuickMontageSettings, Scene


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
    def __init__(self, ffmpeg_binary: str = "ffmpeg") -> None:
        self.ffmpeg_binary = ffmpeg_binary

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
        duration = scene.end_seconds - scene.start_seconds
        timestamps = (
            scene.start_seconds + duration * 0.12,
            scene.start_seconds + duration * 0.5,
            scene.start_seconds + duration * 0.88,
        )
        command = [self.ffmpeg_binary, "-hide_banner", "-loglevel", "error", "-y"]
        for timestamp in timestamps:
            command.extend(["-ss", f"{timestamp:.3f}", "-i", str(asset.path)])
        command.extend(
            [
                "-filter_complex",
                "[0:v]scale=480:270:force_original_aspect_ratio=decrease,"
                "pad=480:270:(ow-iw)/2:(oh-ih)/2:black[a];"
                "[1:v]scale=480:270:force_original_aspect_ratio=decrease,"
                "pad=480:270:(ow-iw)/2:(oh-ih)/2:black[b];"
                "[2:v]scale=480:270:force_original_aspect_ratio=decrease,"
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
        try:
            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    check=False,
                    encoding="utf-8",
                    errors="replace",
                )
            except FileNotFoundError as error:
                raise DependencyUnavailableError(
                    f"FFmpeg executable was not found: {self.ffmpeg_binary}"
                ) from error
            if completed.returncode != 0 or not temporary_path.is_file():
                detail = completed.stderr.strip() or "unknown FFmpeg error"
                raise MontageError(
                    f"Не удалось извлечь кадр из {asset.relative_path}: {detail}"
                )
            os.replace(temporary_path, frame_path)
            return frame_path
        finally:
            temporary_path.unlink(missing_ok=True)


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
