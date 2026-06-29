"""Pipeline stage that extracts representative scene contact sheets."""

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from travelmovieai.analysis.scenes import RepresentativeFrameExtractor, frame_sample_count_for_mode
from travelmovieai.application.context import ProjectContext
from travelmovieai.domain.enums import PipelineStage
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
from travelmovieai.pipeline.base import Stage

ARTIFACT_SCHEMA_VERSION = "frame-sampling-v2"


class FrameSamplingStage(Stage):
    name = PipelineStage.FRAME_SAMPLING

    def run(self, context: ProjectContext) -> StageResult:
        repository = MediaAssetRepository(context.database_path)
        repository.initialize()
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
                "schema": ARTIFACT_SCHEMA_VERSION,
            }
        )
        if stage_cache_manifest_matches(
            cache_artifact,
            stage=self.name,
            artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
            input_fingerprint=input_fingerprint,
            config_fingerprint=config_fingerprint,
            artifacts=[artifact],
        ) and _cached_frame_sampling_valid(artifact, source_scenes):
            return StageResult(
                stage=self.name,
                skipped=True,
                artifacts=[context.database_path, artifact, cache_artifact],
                message="Frame sampling reused cached contact sheets.",
            )

        resources = detect_resource_profile(
            context.settings.ffmpeg_binary,
            worker_override=context.settings.workers,
            batch_override=context.settings.batch_size,
        )
        use_cuda_decode = resources.nvenc and context.settings.device in {"auto", "cuda"}
        frame_workers = 1 if use_cuda_decode else resources.frame_workers
        extractor = RepresentativeFrameExtractor(
            context.settings.ffmpeg_binary,
            context.settings.ffprobe_binary,
            use_cuda_decode=use_cuda_decode,
            frame_sample_count=frame_sample_count,
            timeout_seconds=context.settings.frame_extraction_timeout_seconds,
        )
        scenes, extracted_count, cached_count = _extract_frames(
            source_scenes,
            assets,
            extractor,
            context.frames_dir,
            frame_workers,
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
            skipped=extracted_count == 0,
            artifacts=[context.database_path, artifact, cache_artifact],
            message=(
                f"Frame sampling prepared {len(scenes)} scene(s): "
                f"{extracted_count} extracted, {cached_count} cached, "
                f"workers={min(max(1, frame_workers), max(1, len(scenes)))}, "
                f"decode={'NVDEC serial' if use_cuda_decode else 'CPU'}."
            ),
        )


def _cached_frame_sampling_valid(artifact: Path, scenes: list[Scene]) -> bool:
    try:
        report = FrameSamplingReport.model_validate_json(artifact.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if {scene.id for scene in report.scenes} != {scene.id for scene in scenes}:
        return False
    return all(
        scene.keyframe_path is not None and scene.keyframe_path.is_file() for scene in scenes
    )


def _extract_frames(
    source_scenes: list[Scene],
    assets: Mapping[UUID, MediaAsset],
    extractor: RepresentativeFrameExtractor,
    frames_dir: Path,
    workers: int,
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
    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="travelmovieai-frame-stage",
    ) as executor:
        futures = {
            executor.submit(extractor.extract, scene, asset, frames_dir): (index, scene)
            for index, scene, asset in jobs
        }
        for future in as_completed(futures):
            index, scene = futures[future]
            previous = scene.keyframe_path
            frame_path = future.result()
            cached = previous == frame_path and frame_path.is_file()
            results[index] = (scene.model_copy(update={"keyframe_path": frame_path}), cached)

    ordered = [results[index] for index in sorted(results)]
    scenes = [scene for scene, _ in ordered]
    cached_count = sum(1 for _, cached in ordered if cached)
    return scenes, len(scenes) - cached_count, cached_count


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
