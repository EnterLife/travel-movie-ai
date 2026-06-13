from enum import StrEnum


class MediaType(StrEnum):
    VIDEO = "video"
    PHOTO = "photo"
    AUDIO = "audio"


class StoryStyle(StrEnum):
    CINEMATIC = "cinematic"
    DOCUMENTARY = "documentary"
    FAMILY = "family"
    VLOG = "vlog"
    ADVENTURE = "adventure"
    ROMANTIC = "romantic"


class PipelineStage(StrEnum):
    MEDIA_SCAN = "media_scan"
    SCENE_DETECTION = "scene_detection"
    FRAME_SAMPLING = "frame_sampling"
    VISION_ANALYSIS = "vision_analysis"
    QUALITY_ANALYSIS = "quality_analysis"
    SPEECH_ANALYSIS = "speech_analysis"
    AUDIO_ANALYSIS = "audio_analysis"
    EMBEDDINGS = "embeddings"
    EVENT_DETECTION = "event_detection"
    STORY_BUILDER = "story_builder"
    SCENE_RANKING = "scene_ranking"
    MUSIC_SELECTION = "music_selection"
    NARRATION = "narration"
    VOICE_SYNTHESIS = "voice_synthesis"
    TIMELINE_BUILDER = "timeline_builder"
    RENDERING = "rendering"
