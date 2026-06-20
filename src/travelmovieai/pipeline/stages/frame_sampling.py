"""Pipeline stage that extracts representative scene contact sheets."""

from datetime import UTC, datetime
from pathlib import Path

from travelmovieai.analysis.scenes import RepresentativeFrameExtractor
from travelmovieai.application.context import ProjectContext
from travelmovieai.domain.enums import PipelineStage
from travelmovieai.domain.models import FrameSamplingReport, MediaAsset, Scene, StageResult
from travelmovieai.infrastructure.artifacts import (
    artifact_fingerprint,
    stage_cache_manifest_matches,
    write_json_atomic,
    write_stage_cache_manifest,
)
from travelmovieai.infrastructure.database import MediaAssetRepository
from travelmovieai.infrastructure.system import check_cuda
from travelmovieai.pipeline.base import Stage

ARTIFACT_SCHEMA_VERSION = "frame-sampling-v1"


class FrameSamplingStage(Stage):
    name = PipelineStage.FRAME_SAMPLING

    def run(self, context: ProjectContext) -> StageResult:
        repository = MediaAssetRepository(context.database_path)
        repository.initialize()
        assets = {asset.id: asset for asset in repository.list_assets()}
        source_scenes = repository.list_scenes()
        artifact = context.artifacts_dir / "frame_sampling.json"
        cache_artifact = context.artifacts_dir / "frame_sampling.cache.json"
        input_fingerprint = artifact_fingerprint(
            _asset_inputs(list(assets.values())),
            _scene_inputs(source_scenes),
        )
        config_fingerprint = artifact_fingerprint(
            {
                "ffmpeg_binary": context.settings.ffmpeg_binary,
                "ffprobe_binary": context.settings.ffprobe_binary,
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

        extractor = RepresentativeFrameExtractor(
            context.settings.ffmpeg_binary,
            context.settings.ffprobe_binary,
            use_cuda_decode=check_cuda(context.settings.ffmpeg_binary).available,
        )
        scenes = []
        extracted_count = 0
        cached_count = 0
        for scene in source_scenes:
            asset = assets.get(scene.asset_id)
            if asset is None:
                continue
            previous = scene.keyframe_path
            frame_path = extractor.extract(scene, asset, context.frames_dir)
            if previous == frame_path and frame_path.is_file():
                cached_count += 1
            else:
                extracted_count += 1
            scenes.append(scene.model_copy(update={"keyframe_path": frame_path}))

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
                f"{extracted_count} extracted, {cached_count} cached."
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
