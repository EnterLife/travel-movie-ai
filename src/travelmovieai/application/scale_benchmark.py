"""Synthetic metadata benchmark for large projects without large media files."""

from __future__ import annotations

import math
import tracemalloc
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from uuid import NAMESPACE_URL, uuid5

from pydantic import BaseModel, Field

from travelmovieai.application.resource_estimates import (
    ProjectResourceEstimate,
    estimate_project_resources,
)
from travelmovieai.core.config import Settings
from travelmovieai.domain.enums import MediaType
from travelmovieai.domain.models import MediaAsset, MediaScanReport, QuickMontageSettings, Scene
from travelmovieai.infrastructure.artifacts import artifact_fingerprint
from travelmovieai.infrastructure.database import MediaAssetRepository

_SYNTHETIC_ROOT = Path("synthetic-large-project")
_SYNTHETIC_TIME = datetime(2024, 1, 1, tzinfo=UTC)


class SyntheticScaleBenchmarkResult(BaseModel):
    asset_count: int = Field(ge=1)
    source_bytes: int = Field(ge=1)
    metadata_json_bytes: int = Field(ge=1)
    fingerprint: str = Field(min_length=64, max_length=64)
    scene_count: int = Field(ge=1)
    sqlite_bytes: int = Field(ge=1)
    build_seconds: float = Field(ge=0)
    sqlite_write_seconds: float = Field(ge=0)
    sqlite_read_seconds: float = Field(ge=0)
    asset_throughput_per_second: float = Field(ge=0)
    peak_traced_memory_bytes: int = Field(ge=1)
    estimate: ProjectResourceEstimate


def run_synthetic_metadata_benchmark(
    *,
    asset_count: int = 512,
    total_source_bytes: int = 128 * 1024**3,
) -> SyntheticScaleBenchmarkResult:
    """Exercise typed validation, serialization, hashing, and estimation at scale."""

    tracemalloc.start()
    build_started = perf_counter()
    assets = build_synthetic_assets(asset_count=asset_count, total_source_bytes=total_source_bytes)
    scenes = _build_synthetic_scenes(assets)
    report = MediaScanReport(
        input_path=_SYNTHETIC_ROOT,
        scanned_at=_SYNTHETIC_TIME,
        assets=assets,
        discovered_count=len(assets),
        probed_count=len(assets),
    )
    serialized = report.model_dump_json()
    fingerprint = artifact_fingerprint(
        [
            {
                "id": asset.id,
                "relative_path": asset.relative_path,
                "size_bytes": asset.size_bytes,
                "modified_ns": asset.modified_ns,
                "duration_seconds": asset.duration_seconds,
                "width": asset.width,
                "height": asset.height,
            }
            for asset in assets
        ]
    )
    settings = Settings(
        analysis_proxy_mode="auto",
        analysis_proxy_max_dimension=1920,
    )
    montage_settings = QuickMontageSettings(
        semantic_analysis=True,
        analysis_quality_mode="balanced",
    )
    build_seconds = perf_counter() - build_started
    with TemporaryDirectory(prefix="travelmovieai-scale-") as temporary:
        database_path = Path(temporary) / "benchmark.db"
        repository = MediaAssetRepository(database_path)
        write_started = perf_counter()
        repository.initialize()
        repository.synchronize(assets, _SYNTHETIC_TIME)
        repository.synchronize_scenes(scenes)
        sqlite_write_seconds = perf_counter() - write_started
        read_started = perf_counter()
        stored_assets = repository.list_assets()
        stored_scenes = repository.list_scenes()
        sqlite_read_seconds = perf_counter() - read_started
        repository.close()
        sqlite_bytes = sum(
            path.stat().st_size
            for path in database_path.parent.glob(f"{database_path.name}*")
            if path.is_file()
        )
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    if len(stored_assets) != len(assets) or len(stored_scenes) != len(scenes):
        raise RuntimeError("Synthetic SQLite benchmark did not round-trip all metadata.")
    measured_seconds = build_seconds + sqlite_write_seconds + sqlite_read_seconds
    return SyntheticScaleBenchmarkResult(
        asset_count=len(assets),
        source_bytes=sum(asset.size_bytes for asset in assets),
        metadata_json_bytes=len(serialized.encode("utf-8")),
        fingerprint=fingerprint,
        scene_count=len(scenes),
        sqlite_bytes=sqlite_bytes,
        build_seconds=build_seconds,
        sqlite_write_seconds=sqlite_write_seconds,
        sqlite_read_seconds=sqlite_read_seconds,
        asset_throughput_per_second=(len(assets) / measured_seconds if measured_seconds else 0),
        peak_traced_memory_bytes=max(1, peak_memory),
        estimate=estimate_project_resources(
            assets,
            settings=settings,
            montage_settings=montage_settings,
            known_scene_count=len(scenes),
        ),
    )


def build_synthetic_assets(
    *,
    asset_count: int,
    total_source_bytes: int,
) -> list[MediaAsset]:
    if asset_count < 1:
        raise ValueError("asset_count must be at least 1")
    if total_source_bytes < asset_count:
        raise ValueError("total_source_bytes must be at least asset_count")
    quotient, remainder = divmod(total_source_bytes, asset_count)
    assets: list[MediaAsset] = []
    for index in range(asset_count):
        media_type = _media_type(index)
        extension = {
            MediaType.VIDEO: ".mp4",
            MediaType.PHOTO: ".jpg",
            MediaType.AUDIO: ".wav",
        }[media_type]
        relative_path = Path("Синтетика") / f"media {index:05d}{extension}"
        size_bytes = quotient + (1 if index < remainder else 0)
        is_video = media_type is MediaType.VIDEO
        is_photo = media_type is MediaType.PHOTO
        width = (
            7680
            if is_video and index % 3 == 0
            else 3840
            if is_video
            else 6000
            if is_photo
            else None
        )
        height = (
            4320
            if is_video and index % 3 == 0
            else 2160
            if is_video
            else 4000
            if is_photo
            else None
        )
        duration = max(1.0, size_bytes * 8 / 80_000_000) if is_video else None
        assets.append(
            MediaAsset(
                id=uuid5(NAMESPACE_URL, relative_path.as_posix()),
                path=_SYNTHETIC_ROOT / relative_path,
                relative_path=relative_path,
                media_type=media_type,
                extension=extension,
                size_bytes=size_bytes,
                modified_at=_SYNTHETIC_TIME + timedelta(seconds=index),
                modified_ns=1_704_067_200_000_000_000 + index,
                duration_seconds=duration,
                width=width,
                height=height,
                fps=30 if is_video else None,
                probe_metadata=(
                    {
                        "video_duration_seconds": duration,
                        "streams": [{"codec_type": "video", "codec_name": "h264"}],
                    }
                    if is_video
                    else {}
                ),
            )
        )
    return assets


def _media_type(index: int) -> MediaType:
    if index % 20 == 18:
        return MediaType.PHOTO
    if index % 20 == 19:
        return MediaType.AUDIO
    return MediaType.VIDEO


def _build_synthetic_scenes(assets: list[MediaAsset]) -> list[Scene]:
    scenes: list[Scene] = []
    for asset in assets:
        if asset.media_type is MediaType.AUDIO:
            continue
        duration = 3.0 if asset.media_type is MediaType.PHOTO else asset.duration_seconds or 0
        count = 1 if asset.media_type is MediaType.PHOTO else max(1, math.ceil(duration / 12))
        for index in range(count):
            start = min(duration, index * 12.0)
            end = min(duration, max(start + 0.5, (index + 1) * 12.0))
            identity = f"{asset.id}:{index}:{start:.3f}:{end:.3f}"
            scenes.append(
                Scene(
                    id=uuid5(NAMESPACE_URL, identity),
                    asset_id=asset.id,
                    start_seconds=start,
                    end_seconds=end,
                    caption=f"Synthetic scene {index + 1}",
                    metadata={"cache_key": artifact_fingerprint(identity)},
                )
            )
    return scenes
