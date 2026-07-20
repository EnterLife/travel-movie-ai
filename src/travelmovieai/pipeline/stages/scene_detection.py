"""Pipeline stage that detects video scenes and persists their boundaries."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, ValidationError

from travelmovieai.analysis.scenes import SceneDetector, scene_cache_key
from travelmovieai.application.context import ProjectContext
from travelmovieai.domain.enums import MediaType, PipelineStage, StageStatus
from travelmovieai.domain.models import (
    MediaAsset,
    QuickMontageSettings,
    Scene,
    SceneDetectionReport,
    StageExecutionMetadata,
    StageResult,
)
from travelmovieai.infrastructure.artifacts import write_json_atomic
from travelmovieai.infrastructure.database import MediaAssetRepository
from travelmovieai.media.proxy import AnalysisMedia, AnalysisProxyManager
from travelmovieai.pipeline.base import Stage

_FALLBACK_METADATA_KEY = "scene_detection_fallback"

_SHARD_SCHEMA_VERSION: Literal[1] = 1


class _SceneAssetShard(BaseModel):
    schema_version: Literal[1] = _SHARD_SCHEMA_VERSION
    created_at: datetime
    asset_id: UUID
    cache_key: str = Field(min_length=64, max_length=64)
    scenes: list[Scene]
    fallback_used: bool = False
    proxied: bool = False
    proxy_cache_hit: bool = False


@dataclass(frozen=True, slots=True)
class _AssetDetectionResult:
    scenes: list[Scene]
    fallback_used: bool
    proxied: bool
    proxy_cache_hit: bool


class SceneDetectionStage(Stage):
    name = PipelineStage.SCENE_DETECTION

    def __init__(
        self,
        settings: QuickMontageSettings | None = None,
        detector: SceneDetector | None = None,
        proxy_manager: AnalysisProxyManager | None = None,
        workers: int | None = None,
    ) -> None:
        self._settings = settings
        self._detector = detector or SceneDetector()
        self._proxy_manager = proxy_manager
        self._workers = workers

    def run(self, context: ProjectContext) -> StageResult:
        settings = (
            self._settings
            or context.montage_settings
            or QuickMontageSettings(story_style=context.style)
        )
        with MediaAssetRepository(context.database_path) as repository:
            repository.initialize()
            assets = repository.list_assets()
            existing = _group_scenes(repository.list_scenes())
            proxy_manager = self._proxy_manager or AnalysisProxyManager(
                context.cache_dir / "proxies",
                ffmpeg_binary=context.settings.ffmpeg_binary,
                ffprobe_binary=context.settings.ffprobe_binary,
                mode=context.settings.analysis_proxy_mode,
                max_dimension=context.settings.analysis_proxy_max_dimension,
                video_bitrate_mbps=context.settings.analysis_proxy_video_bitrate_mbps,
                timeout_seconds=context.settings.analysis_proxy_timeout_seconds,
            )
            proxy_cache_identity = proxy_manager.cache_identity()
            workers = _scene_workers(context, self._workers, len(assets))
            scenes, statistics = _detect_assets(
                assets,
                existing,
                settings=settings,
                detector=self._detector,
                proxy_manager=proxy_manager,
                proxy_cache_identity=proxy_cache_identity,
                shard_dir=context.artifacts_dir / "scene_detection_shards",
                workers=workers,
                progress=context.progress,
            )
            repository.synchronize_scenes(scenes)

        report = SceneDetectionReport(
            created_at=datetime.now(UTC),
            scenes=scenes,
            detected_count=len(scenes) - statistics.cached_count,
            cached_count=statistics.cached_count,
            fallback_count=statistics.fallback_count,
        )
        artifact = context.artifacts_dir / "scenes.json"
        write_json_atomic(artifact, report)
        return StageResult(
            stage=self.name,
            status=(
                StageStatus.DEGRADED
                if report.fallback_count > 0
                else StageStatus.CACHED
                if report.detected_count == 0 and report.cached_count > 0
                else StageStatus.NO_INPUT
                if not scenes
                else StageStatus.COMPLETED
            ),
            cache_hit=report.detected_count == 0 and report.cached_count > 0,
            artifacts=[context.database_path, artifact],
            message=(
                f"Scene detection produced {len(scenes)} scene(s): "
                f"{report.detected_count} detected, {report.cached_count} cached, "
                f"{report.fallback_count} from fallback; "
                f"proxies={statistics.generated_proxies} generated/"
                f"{statistics.cached_proxies} cached; workers={workers}."
            ),
            execution=StageExecutionMetadata(
                fallback_count=report.fallback_count,
                fallback_provider="uniform" if report.fallback_count else None,
            ),
        )


@dataclass(frozen=True, slots=True)
class _DetectionStatistics:
    cached_count: int
    fallback_count: int
    generated_proxies: int
    cached_proxies: int


def _detect_assets(
    assets: list[MediaAsset],
    existing: dict[str, list[Scene]],
    *,
    settings: QuickMontageSettings,
    detector: SceneDetector,
    proxy_manager: AnalysisProxyManager,
    proxy_cache_identity: str,
    shard_dir: Path,
    workers: int,
    progress: Callable[[int, int, str], None] | None,
) -> tuple[list[Scene], _DetectionStatistics]:
    ordered_assets = sorted(
        assets,
        key=lambda asset: (asset.relative_path.as_posix().casefold(), str(asset.id)),
    )
    total_assets = len(ordered_assets)
    results: dict[int, _AssetDetectionResult] = {}
    jobs: list[tuple[int, MediaAsset, str, Path]] = []
    cached_count = 0
    fallback_count = 0
    generated_proxies = 0
    cached_proxies = 0
    completed = 0
    shard_dir.mkdir(parents=True, exist_ok=True)
    if progress is not None:
        progress(0, total_assets, f"Scene detection: 0/{total_assets}")

    for index, asset in enumerate(ordered_assets):
        if asset.scan_error or asset.media_type not in {MediaType.VIDEO, MediaType.PHOTO}:
            results[index] = _AssetDetectionResult([], False, False, False)
            completed += 1
            if progress is not None:
                progress(completed, total_assets, f"Scene detection: {completed}/{total_assets}")
            continue
        expected_key = scene_cache_key(
            asset,
            settings,
            analysis_fingerprint=proxy_cache_identity,
        )
        cached = sorted(
            existing.get(str(asset.id), []),
            key=lambda scene: (scene.start_seconds, scene.end_seconds, str(scene.id)),
        )
        if cached and all(
            scene.metadata.get("cache_key") == expected_key
            and scene.metadata.get(_FALLBACK_METADATA_KEY) is False
            for scene in cached
        ):
            results[index] = _AssetDetectionResult(cached, False, False, False)
            cached_count += len(cached)
            completed += 1
            if progress is not None:
                progress(
                    completed,
                    total_assets,
                    f"Scene detection cache: {completed}/{total_assets}",
                )
            continue
        shard_path = shard_dir / f"{asset.id}.json"
        shard = _read_valid_shard(shard_path, asset.id, expected_key)
        if shard is not None:
            results[index] = _AssetDetectionResult(
                shard.scenes,
                shard.fallback_used,
                shard.proxied,
                shard.proxy_cache_hit,
            )
            cached_count += len(shard.scenes)
            if shard.fallback_used:
                fallback_count += len(shard.scenes)
            completed += 1
            if progress is not None:
                progress(
                    completed,
                    total_assets,
                    f"Scene detection checkpoint: {completed}/{total_assets}",
                )
            continue
        jobs.append((index, asset, expected_key, shard_path))

    worker_count = min(max(1, workers), max(1, len(jobs)))
    executor = ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="travelmovieai-scene-stage",
    )
    futures: dict[Future[_AssetDetectionResult], tuple[int, MediaAsset, str, Path]] = {}
    job_iterator = iter(jobs)

    def submit_next() -> bool:
        try:
            index, asset, expected_key, shard_path = next(job_iterator)
        except StopIteration:
            return False
        future = executor.submit(
            _detect_one_asset,
            asset,
            settings,
            detector,
            proxy_manager,
            proxy_cache_identity,
            expected_key,
        )
        futures[future] = (index, asset, expected_key, shard_path)
        return True

    try:
        for _ in range(worker_count):
            if not submit_next():
                break
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            if any(future.exception() is not None for future in done):
                pending = set(futures).difference(done)
                if pending:
                    completed_peers, _ = wait(pending)
                    done.update(completed_peers)
            failed: list[Future[_AssetDetectionResult]] = []
            progress_values: list[int] = []
            for future in sorted(done, key=lambda item: futures[item][0]):
                index, asset, expected_key, shard_path = futures.pop(future)
                if future.exception() is not None:
                    failed.append(future)
                    continue
                result = future.result()
                write_json_atomic(
                    shard_path,
                    _SceneAssetShard(
                        created_at=datetime.now(UTC),
                        asset_id=asset.id,
                        cache_key=expected_key,
                        scenes=result.scenes,
                        fallback_used=result.fallback_used,
                        proxied=result.proxied,
                        proxy_cache_hit=result.proxy_cache_hit,
                    ),
                )
                results[index] = result
                if result.fallback_used:
                    fallback_count += len(result.scenes)
                if result.proxied:
                    if result.proxy_cache_hit:
                        cached_proxies += 1
                    else:
                        generated_proxies += 1
                completed += 1
                progress_values.append(completed)
            if failed:
                failed[0].result()
                raise AssertionError("A failed scene future did not raise its exception.")
            for progress_value in progress_values:
                if progress is not None:
                    progress(
                        progress_value,
                        total_assets,
                        f"Scene detection: {progress_value}/{total_assets}",
                    )
                submit_next()
    finally:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)

    current_asset_ids = {str(asset.id) for asset in ordered_assets}
    for shard_path in shard_dir.glob("*.json"):
        if shard_path.stem not in current_asset_ids:
            shard_path.unlink(missing_ok=True)
    ordered_scenes = [
        scene for index in range(len(ordered_assets)) for scene in results[index].scenes
    ]
    return ordered_scenes, _DetectionStatistics(
        cached_count=cached_count,
        fallback_count=fallback_count,
        generated_proxies=generated_proxies,
        cached_proxies=cached_proxies,
    )


def _detect_one_asset(
    asset: MediaAsset,
    settings: QuickMontageSettings,
    detector: SceneDetector,
    proxy_manager: AnalysisProxyManager,
    proxy_cache_identity: str,
    expected_key: str,
) -> _AssetDetectionResult:
    analysis_media = proxy_manager.resolve(asset)
    analysis_asset = _analysis_asset(
        asset,
        analysis_media,
        proxy_cache_identity=proxy_cache_identity,
    )
    detected, used_fallback = detector.detect(analysis_asset, settings)
    normalized = [
        scene.model_copy(
            update={
                "metadata": {
                    **scene.metadata,
                    "cache_key": expected_key,
                    _FALLBACK_METADATA_KEY: used_fallback,
                }
            }
        )
        for scene in detected
    ]
    return _AssetDetectionResult(
        scenes=sorted(
            normalized,
            key=lambda scene: (scene.start_seconds, scene.end_seconds, str(scene.id)),
        ),
        fallback_used=used_fallback,
        proxied=analysis_media.proxied,
        proxy_cache_hit=analysis_media.cache_hit,
    )


def _read_valid_shard(
    path: Path,
    asset_id: UUID,
    expected_key: str,
) -> _SceneAssetShard | None:
    try:
        shard = _SceneAssetShard.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError):
        return None
    if shard.asset_id != asset_id or shard.cache_key != expected_key:
        return None
    if shard.fallback_used:
        return None
    if any(
        scene.asset_id != asset_id or scene.metadata.get("cache_key") != expected_key
        for scene in shard.scenes
    ):
        return None
    if any(
        scene.metadata.get(_FALLBACK_METADATA_KEY) is not shard.fallback_used
        for scene in shard.scenes
    ):
        return None
    return shard


def _scene_workers(
    context: ProjectContext,
    override: int | None,
    asset_count: int,
) -> int:
    if asset_count <= 0:
        return 1
    if override is not None:
        return min(max(1, override), asset_count)
    if context.resources is None:
        return 1
    cap = {"safe": 1, "balanced": 4, "performance": 8}[context.resources.resource_mode]
    return min(max(1, context.resources.analysis_workers), cap, asset_count)


def _group_scenes(scenes: list[Scene]) -> dict[str, list[Scene]]:
    grouped: dict[str, list[Scene]] = {}
    for scene in scenes:
        grouped.setdefault(str(scene.asset_id), []).append(scene)
    return grouped


def _analysis_asset(
    asset: MediaAsset,
    media: AnalysisMedia,
    *,
    proxy_cache_identity: str,
) -> MediaAsset:
    updates: dict[str, object] = {
        "probe_metadata": {
            **asset.probe_metadata,
            "scene_analysis_fingerprint": proxy_cache_identity,
        }
    }
    if media.proxied:
        updates.update(
            {
                "path": media.analysis_path,
                "width": media.width,
                "height": media.height,
                "duration_seconds": media.duration_seconds or asset.duration_seconds,
                "probe_metadata": {
                    **asset.probe_metadata,
                    "scene_analysis_fingerprint": proxy_cache_identity,
                    "analysis_proxy_fingerprint": media.cache_key,
                    "video_duration_seconds": media.duration_seconds,
                },
            }
        )
    return asset.model_copy(update=updates)
