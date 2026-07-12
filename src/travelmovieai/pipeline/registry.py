"""Default stage registry."""

from travelmovieai.domain.enums import PipelineStage
from travelmovieai.pipeline.base import Stage
from travelmovieai.pipeline.stages.audio_analysis import AudioAnalysisStage
from travelmovieai.pipeline.stages.duplicate_detection import DuplicateDetectionStage
from travelmovieai.pipeline.stages.embeddings import EmbeddingsStage
from travelmovieai.pipeline.stages.event_detection import EventDetectionStage
from travelmovieai.pipeline.stages.frame_sampling import FrameSamplingStage
from travelmovieai.pipeline.stages.media_scan import MediaScanStage
from travelmovieai.pipeline.stages.music_selection import MusicSelectionStage
from travelmovieai.pipeline.stages.narration import NarrationStage
from travelmovieai.pipeline.stages.quality_analysis import QualityAnalysisStage
from travelmovieai.pipeline.stages.rendering import RenderingStage
from travelmovieai.pipeline.stages.scene_detection import SceneDetectionStage
from travelmovieai.pipeline.stages.scene_ranking import SceneRankingStage
from travelmovieai.pipeline.stages.speech_analysis import SpeechAnalysisStage
from travelmovieai.pipeline.stages.story_builder import SceneCaptioningStage
from travelmovieai.pipeline.stages.storyboard import StoryBuilderStage
from travelmovieai.pipeline.stages.timeline_builder import TimelineBuilderStage
from travelmovieai.pipeline.stages.vision_analysis import VisionAnalysisStage
from travelmovieai.pipeline.stages.voice_synthesis import VoiceSynthesisStage


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
        EmbeddingsStage(),
        DuplicateDetectionStage(),
        SceneCaptioningStage(),
        EventDetectionStage(),
        StoryBuilderStage(),
        SceneRankingStage(),
        MusicSelectionStage(),
        NarrationStage(),
        VoiceSynthesisStage(),
        TimelineBuilderStage(),
        RenderingStage(),
    ]
    stages_by_name = {stage.name: stage for stage in implemented}
    return [stages_by_name[stage] for stage in PipelineStage]
