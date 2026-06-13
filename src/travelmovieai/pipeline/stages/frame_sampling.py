"""Pipeline stage that extracts representative scene contact sheets."""

from datetime import UTC, datetime

from travelmovieai.analysis.scenes import RepresentativeFrameExtractor
from travelmovieai.application.context import ProjectContext
from travelmovieai.domain.enums import PipelineStage
from travelmovieai.domain.models import FrameSamplingReport, StageResult
from travelmovieai.infrastructure.artifacts import write_json_atomic
from travelmovieai.infrastructure.database import MediaAssetRepository
from travelmovieai.pipeline.base import Stage


class FrameSamplingStage(Stage):
    name = PipelineStage.FRAME_SAMPLING

    def run(self, context: ProjectContext) -> StageResult:
        repository = MediaAssetRepository(context.database_path)
        repository.initialize()
        assets = {asset.id: asset for asset in repository.list_assets()}
        extractor = RepresentativeFrameExtractor(context.settings.ffmpeg_binary)
        scenes = []
        extracted_count = 0
        cached_count = 0
        for scene in repository.list_scenes():
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
        artifact = context.artifacts_dir / "frame_sampling.json"
        write_json_atomic(artifact, report)
        return StageResult(
            stage=self.name,
            skipped=extracted_count == 0,
            artifacts=[context.database_path, artifact],
            message=(
                f"Frame sampling prepared {len(scenes)} scene(s): "
                f"{extracted_count} extracted, {cached_count} cached."
            ),
        )
