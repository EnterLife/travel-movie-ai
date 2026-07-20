"""Pipeline stage that extracts representative scene contact sheets."""

import hashlib
from collections.abc import Callable, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from travelmovieai.analysis.scenes import (
    CONTACT_SHEET_SCHEMA_VERSION,
    RepresentativeFrameExtractor,
    contact_sheet_file_valid,
    frame_sample_count_for_mode,
)
from travelmovieai.application.context import ProjectContext
from travelmovieai.domain.enums import MediaType, PipelineStage, StageStatus
from travelmovieai.domain.models import (
    FrameSamplingReport,
    MediaAsset,
    QuickMontageSettings,
    Scene,
    StageResult,
)
from travelmovieai.infrastructure.artifacts import (
    artifact_fingerprint,
    stage_cache_manifest_matches,
    write_json_atomic,
    write_stage_cache_manifest,
)
from travelmovieai.infrastructure.database import MediaAssetRepository
from travelmovieai.infrastructure.system import detect_resource_profile
from travelmovieai.media.proxy import AnalysisMedia, AnalysisProxyManager
from travelmovieai.pipeline.base import Stage

ARTIFACT_SCHEMA_VERSION = "frame-sampling-v4-temporal-samples"


class FrameSamplingStage(Stage):
    name = PipelineStage.FRAME_SAMPLING

    def run(self, context: ProjectContext) -> StageResult:
        with MediaAssetRepository(context.database_path) as repository:
            repository.initialize()
            return self._run_with_repository(context, repository)

    def _run_with_repository(
        self,
        context: ProjectContext,
        repository: MediaAssetRepository,
    ) -> StageResult:
        assets = {asset.id: asset for asset in repository.list_assets()}
        source_scenes = repository.list_scenes()
        artifact = context.artifacts_dir / "frame_sampling.json"
        cache_artifact = context.artifacts_dir / "frame_sampling.cache.json"
        montage_settings = context.montage_settings or QuickMontageSettings()
        frame_sample_count = frame_sample_count_for_mode(montage_settings.analysis_quality_mode)
        input_fingerprint = artifact_fingerprint(
            _asset_inputs(list(assets.values())),
            _scene_inputs(source_scenes),
        )
        config_fingerprint = artifact_fingerprint(
            {
                "ffmpeg_binary": context.settings.ffmpeg_binary,
                "ffprobe_binary": context.settings.ffprobe_binary,
                "frame_extraction_timeout_seconds": (
                    context.settings.frame_extraction_timeout_seconds
                ),
                "analysis_quality_mode": montage_settings.analysis_quality_mode,
                "frame_sample_count": frame_sample_count,
                "analysis_proxy_mode": context.settings.analysis_proxy_mode,
                "analysis_proxy_max_dimension": (context.settings.analysis_proxy_max_dimension),
                "analysis_proxy_video_bitrate_mbps": (
                    context.settings.analysis_proxy_video_bitrate_mbps
                ),
                "schema": ARTIFACT_SCHEMA_VERSION,
            }
        )
        cached_report = _read_valid_cached_frame_sampling(
            artifact,
            source_scenes,
            assets,
            frame_sample_count,
        )
        if (
            stage_cache_manifest_matches(
                cache_artifact,
                stage=self.name,
                artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
                input_fingerprint=input_fingerprint,
                config_fingerprint=config_fingerprint,
                artifacts=[artifact],
            )
            and cached_report is not None
        ):
            restored_scenes = _restore_cached_frame_state(source_scenes, cached_report.scenes)
            if restored_scenes != source_scenes:
                repository.synchronize_scenes(restored_scenes)
            return StageResult(
                stage=self.name,
                status=StageStatus.CACHED,
                artifacts=[context.database_path, artifact, cache_artifact],
                message="Frame sampling reused cached contact sheets.",
            )

        resources = context.resources or detect_resource_profile(
            context.settings.ffmpeg_binary,
            worker_override=context.settings.workers,
            batch_override=context.settings.batch_size,
            resource_mode=context.settings.resource_mode,
            gpu_memory_reserve_mb=context.settings.gpu_memory_reserve_mb,
            max_gpu_processes=context.settings.max_gpu_processes,
        )
        use_cuda_decode = resources.nvenc and context.settings.device in {"auto", "cuda"}
        frame_workers = (
            min(
                resources.frame_workers,
                1 if resources.resource_mode == "safe" else context.settings.max_gpu_processes,
            )
            if use_cuda_decode
            else resources.frame_workers
        )
        extractor = RepresentativeFrameExtractor(
            context.settings.ffmpeg_binary,
            context.settings.ffprobe_binary,
            use_cuda_decode=use_cuda_decode,
            frame_sample_count=frame_sample_count,
            timeout_seconds=context.settings.frame_extraction_timeout_seconds,
        )
        proxy_manager = AnalysisProxyManager(
            context.cache_dir / "proxies",
            ffmpeg_binary=context.settings.ffmpeg_binary,
            ffprobe_binary=context.settings.ffprobe_binary,
            mode=context.settings.analysis_proxy_mode,
            max_dimension=context.settings.analysis_proxy_max_dimension,
            video_bitrate_mbps=context.settings.analysis_proxy_video_bitrate_mbps,
            timeout_seconds=context.settings.analysis_proxy_timeout_seconds,
        )
        analysis_assets, generated_proxies, cached_proxies = _prepare_analysis_assets(
            assets,
            proxy_manager,
            workers=min(2, max(1, frame_workers)),
            progress=_scaled_progress(context.progress, 0, 25),
        )
        scenes, extracted_count, cached_count = _extract_frames(
            source_scenes,
            analysis_assets,
            extractor,
            context.frames_dir,
            frame_workers,
            progress=_scaled_progress(context.progress, 25, 100),
        )

        repository.synchronize_scenes(scenes)
        report = FrameSamplingReport(
            created_at=datetime.now(UTC),
            scenes=scenes,
            extracted_count=extracted_count,
            cached_count=cached_count,
        )
        write_json_atomic(artifact, report)
        write_stage_cache_manifest(
            cache_artifact,
            stage=self.name,
            artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
            input_fingerprint=input_fingerprint,
            config_fingerprint=config_fingerprint,
            artifacts=[artifact],
        )
        return StageResult(
            stage=self.name,
            status=(
                StageStatus.CACHED
                if extracted_count == 0 and cached_count > 0
                else StageStatus.NO_INPUT
                if not scenes
                else StageStatus.COMPLETED
            ),
            artifacts=[context.database_path, artifact, cache_artifact],
            message=(
                f"Frame sampling prepared {len(scenes)} scene(s): "
                f"{extracted_count} extracted, {cached_count} cached, "
                f"proxies={generated_proxies} generated/{cached_proxies} cached, "
                f"workers={min(max(1, frame_workers), max(1, len(scenes)))}, "
                f"decode={'NVDEC' if use_cuda_decode else 'CPU'}."
            ),
        )


def _read_valid_cached_frame_sampling(
    artifact: Path,
    scenes: list[Scene],
    assets: Mapping[UUID, MediaAsset],
    frame_sample_count: int,
) -> FrameSamplingReport | None:
    try:
        report = FrameSamplingReport.model_validate_json(artifact.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if {scene.id for scene in report.scenes} != {scene.id for scene in scenes}:
        return None
    for scene in report.scenes:
        if scene.keyframe_path is None or not scene.keyframe_path.is_file():
            return None
        contact_sheet = scene.metadata.get("contact_sheet")
        if not isinstance(contact_sheet, dict):
            return None
        asset = assets.get(scene.asset_id)
        if asset is None:
            return None
        expected_sample_count = 1 if asset.media_type is MediaType.PHOTO else frame_sample_count
        if not contact_sheet_file_valid(
            scene.keyframe_path,
            contact_sheet,
            expected_sample_count=expected_sample_count,
        ):
            return None
    return report


def _restore_cached_frame_state(
    current_scenes: list[Scene],
    cached_scenes: list[Scene],
) -> list[Scene]:
    cached_by_id = {scene.id: scene for scene in cached_scenes}
    restored: list[Scene] = []
    for scene in current_scenes:
        cached = cached_by_id[scene.id]
        metadata = dict(scene.metadata)
        metadata["contact_sheet"] = cached.metadata["contact_sheet"]
        restored.append(
            scene.model_copy(
                update={
                    "keyframe_path": cached.keyframe_path,
                    "metadata": metadata,
                }
            )
        )
    return restored


def _extract_frames(
    source_scenes: list[Scene],
    assets: Mapping[UUID, MediaAsset],
    extractor: RepresentativeFrameExtractor,
    frames_dir: Path,
    workers: int,
    progress: Callable[[int, int, str], None] | None = None,
) -> tuple[list[Scene], int, int]:
    jobs = [
        (index, scene, asset)
        for index, scene in enumerate(source_scenes)
        if (asset := assets.get(scene.asset_id)) is not None
    ]
    if not jobs:
        return [], 0, 0

    worker_count = min(max(1, workers), len(jobs))
    results: dict[int, tuple[Scene, bool]] = {}
    if progress is not None:
        progress(0, len(jobs), f"Frames: 0/{len(jobs)}")
    executor = ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="travelmovieai-frame-stage",
    )
    futures: dict[Future[Path], tuple[int, Scene, MediaAsset, bool]] = {}
    job_iterator = iter(jobs)

    def submit_next() -> bool:
        try:
            index, scene, asset = next(job_iterator)
        except StopIteration:
            return False
        cache_was_valid = _scene_contact_sheet_valid(scene, asset, extractor)
        futures[executor.submit(extractor.extract, scene, asset, frames_dir)] = (
            index,
            scene,
            asset,
            cache_was_valid,
        )
        return True

    completed = 0
    try:
        for _ in range(worker_count):
            if not submit_next():
                break
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                index, scene, asset, cache_was_valid = futures.pop(future)
                previous = scene.keyframe_path
                frame_path = future.result()
                cached = cache_was_valid and previous == frame_path and frame_path.is_file()
                sampling_metadata = getattr(extractor, "sampling_metadata", None)
                contact_sheet = (
                    sampling_metadata(scene, asset, frame_path)
                    if callable(sampling_metadata)
                    else _single_frame_metadata(frame_path)
                )
                results[index] = (
                    scene.model_copy(
                        update={
                            "keyframe_path": frame_path,
                            "metadata": {
                                **scene.metadata,
                                "contact_sheet": contact_sheet,
                            },
                        }
                    ),
                    cached,
                )
                completed += 1
                if progress is not None:
                    progress(completed, len(jobs), f"Frames: {completed}/{len(jobs)}")
                submit_next()
    finally:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)

    ordered = [results[index] for index in sorted(results)]
    scenes = [scene for scene, _ in ordered]
    cached_count = sum(1 for _, cached in ordered if cached)
    return scenes, len(scenes) - cached_count, cached_count


def _scene_contact_sheet_valid(
    scene: Scene,
    asset: MediaAsset,
    extractor: RepresentativeFrameExtractor,
) -> bool:
    if scene.keyframe_path is None:
        return False
    metadata = scene.metadata.get("contact_sheet")
    if not isinstance(metadata, dict):
        return False
    extractor_sample_count = getattr(extractor, "frame_sample_count", None)
    expected_sample_count = (
        1
        if asset.media_type is MediaType.PHOTO
        else extractor_sample_count
        if isinstance(extractor_sample_count, int)
        else None
    )
    return contact_sheet_file_valid(
        scene.keyframe_path,
        metadata,
        expected_sample_count=expected_sample_count,
    )


def _single_frame_metadata(frame_path: Path) -> dict[str, object]:
    return {
        "schema_version": CONTACT_SHEET_SCHEMA_VERSION,
        "sample_count": 1,
        "sample_positions": [0.5],
        "sample_timestamps_seconds": [0.0],
        "columns": 1,
        "rows": 1,
        "content_sha256": hashlib.sha256(frame_path.read_bytes()).hexdigest(),
    }


def _prepare_analysis_assets(
    assets: Mapping[UUID, MediaAsset],
    manager: AnalysisProxyManager,
    *,
    workers: int,
    progress: Callable[[int, int, str], None] | None = None,
) -> tuple[dict[UUID, MediaAsset], int, int]:
    if not assets:
        return {}, 0, 0
    ordered_assets = sorted(assets.values(), key=lambda asset: str(asset.id))
    worker_count = min(max(1, workers), len(ordered_assets))
    resolved: dict[UUID, AnalysisMedia] = {}
    if progress is not None:
        progress(0, len(ordered_assets), f"Analysis proxies: 0/{len(ordered_assets)}")
    executor = ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="travelmovieai-proxy-stage",
    )
    futures: dict[Future[AnalysisMedia], UUID] = {}
    asset_iterator = iter(ordered_assets)

    def submit_next() -> bool:
        try:
            asset = next(asset_iterator)
        except StopIteration:
            return False
        futures[executor.submit(manager.resolve, asset)] = asset.id
        return True

    completed = 0
    try:
        for _ in range(worker_count):
            if not submit_next():
                break
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                asset_id = futures.pop(future)
                resolved[asset_id] = future.result()
                completed += 1
                if progress is not None:
                    progress(
                        completed,
                        len(ordered_assets),
                        f"Analysis proxies: {completed}/{len(ordered_assets)}",
                    )
                submit_next()
    finally:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)

    prepared: dict[UUID, MediaAsset] = {}
    for asset in ordered_assets:
        media = resolved[asset.id]
        if not media.proxied:
            prepared[asset.id] = asset
            continue
        prepared[asset.id] = asset.model_copy(
            update={
                "path": media.analysis_path,
                "width": media.width,
                "height": media.height,
                "duration_seconds": media.duration_seconds,
                "probe_metadata": {
                    **asset.probe_metadata,
                    "analysis_proxy_fingerprint": media.cache_key,
                    "video_duration_seconds": media.duration_seconds,
                },
            }
        )
    generated = sum(1 for media in resolved.values() if media.proxied and not media.cache_hit)
    cached = sum(1 for media in resolved.values() if media.proxied and media.cache_hit)
    return prepared, generated, cached


def _asset_inputs(assets: list[MediaAsset]) -> list[dict[str, object]]:
    return [
        {
            "id": str(asset.id),
            "path": asset.path,
            "size_bytes": asset.size_bytes,
            "modified_ns": asset.modified_ns,
            "duration_seconds": asset.duration_seconds,
            "width": asset.width,
            "height": asset.height,
        }
        for asset in sorted(assets, key=lambda item: str(item.id))
    ]


def _scene_inputs(scenes: list[Scene]) -> list[dict[str, object]]:
    return [
        {
            "id": str(scene.id),
            "asset_id": str(scene.asset_id),
            "start_seconds": scene.start_seconds,
            "end_seconds": scene.end_seconds,
            "cache_key": scene.metadata.get("cache_key"),
        }
        for scene in sorted(scenes, key=lambda item: str(item.id))
    ]


def _scaled_progress(
    progress: Callable[[int, int, str], None] | None,
    start_percent: int,
    end_percent: int,
) -> Callable[[int, int, str], None] | None:
    if progress is None:
        return None

    def report(current: int, total: int, message: str) -> None:
        fraction = current / total if total > 0 else 0.0
        bounded = max(0.0, min(1.0, fraction))
        percent = start_percent + (end_percent - start_percent) * bounded
        progress(round(percent * 10), 1000, message)

    return report
