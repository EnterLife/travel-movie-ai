"""Default stage registry."""

from travelmovieai.domain.enums import PipelineStage
from travelmovieai.pipeline.base import Stage
from travelmovieai.pipeline.stages.media_scan import MediaScanStage
from travelmovieai.pipeline.stages.placeholders import PlaceholderStage


def build_default_pipeline() -> list[Stage]:
    """Return stages in the order defined by the technical specification."""
    return [
        MediaScanStage(),
        *[
            PlaceholderStage(stage)
            for stage in PipelineStage
            if stage is not PipelineStage.MEDIA_SCAN
        ],
    ]
