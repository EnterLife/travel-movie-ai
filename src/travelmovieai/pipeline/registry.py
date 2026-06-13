"""Default stage registry."""

from travelmovieai.domain.enums import PipelineStage
from travelmovieai.pipeline.base import Stage
from travelmovieai.pipeline.stages.media_scan import MediaScanStage
from travelmovieai.pipeline.stages.placeholders import PlaceholderStage
from travelmovieai.pipeline.stages.scene_detection import SceneDetectionStage


def build_default_pipeline() -> list[Stage]:
    """Return stages in the order defined by the technical specification."""
    return [
        MediaScanStage(),
        SceneDetectionStage(),
        *[
            PlaceholderStage(stage)
            for stage in PipelineStage
            if stage not in {PipelineStage.MEDIA_SCAN, PipelineStage.SCENE_DETECTION}
        ],
    ]
