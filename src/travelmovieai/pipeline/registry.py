"""Default stage registry."""

from travelmovieai.domain.enums import PipelineStage
from travelmovieai.pipeline.base import Stage
from travelmovieai.pipeline.stages.audio_analysis import AudioAnalysisStage
from travelmovieai.pipeline.stages.duplicate_detection import DuplicateDetectionStage
from travelmovieai.pipeline.stages.event_detection import EventDetectionStage
from travelmovieai.pipeline.stages.frame_sampling import FrameSamplingStage
from travelmovieai.pipeline.stages.media_scan import MediaScanStage
from travelmovieai.pipeline.stages.placeholders import PlaceholderStage
from travelmovieai.pipeline.stages.quality_analysis import QualityAnalysisStage
from travelmovieai.pipeline.stages.scene_detection import SceneDetectionStage
from travelmovieai.pipeline.stages.speech_analysis import SpeechAnalysisStage
from travelmovieai.pipeline.stages.story_builder import SceneCaptioningStage
from travelmovieai.pipeline.stages.storyboard import StoryBuilderStage
from travelmovieai.pipeline.stages.vision_analysis import VisionAnalysisStage


def build_default_pipeline() -> list[Stage]:
    """Return stages in the order defined by the technical specification."""
    implemented: list[Stage] = [
        MediaScanStage(),
        SceneDetectionStage(),
        FrameSamplingStage(),
        QualityAnalysisStage(),
        VisionAnalysisStage(),
        SpeechAnalysisStage(),
        AudioAnalysisStage(),
        DuplicateDetectionStage(),
        SceneCaptioningStage(),
        EventDetectionStage(),
        StoryBuilderStage(),
    ]
    stages_by_name = {stage.name: stage for stage in implemented}
    return [
        stages_by_name.get(stage, PlaceholderStage(stage))
        for stage in PipelineStage
    ]
