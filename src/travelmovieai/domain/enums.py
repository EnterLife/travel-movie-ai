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
    QUALITY_ANALYSIS = "quality_analysis"
    VISION_ANALYSIS = "vision_analysis"
    SPEECH_ANALYSIS = "speech_analysis"
    AUDIO_ANALYSIS = "audio_analysis"
    EMBEDDINGS = "embeddings"
    DUPLICATE_DETECTION = "duplicate_detection"
    SCENE_CAPTIONING = "scene_captioning"
    EVENT_DETECTION = "event_detection"
    STORY_BUILDER = "story_builder"
    SCENE_RANKING = "scene_ranking"
    MUSIC_SELECTION = "music_selection"
    NARRATION = "narration"
    VOICE_SYNTHESIS = "voice_synthesis"
    TIMELINE_BUILDER = "timeline_builder"
    RENDERING = "rendering"


class LocationType(StrEnum):
    UNKNOWN = "unknown"
    BEACH = "beach"
    SEA = "sea"
    MOUNTAINS = "mountains"
    FOREST = "forest"
    CITY = "city"
    AIRPORT = "airport"
    HOTEL = "hotel"
    RESTAURANT = "restaurant"
    MUSEUM = "museum"
    PARK = "park"
    LANDMARK = "landmark"
    TRANSPORT = "transport"
    INDOOR = "indoor"
    OTHER = "other"


class ActivityType(StrEnum):
    UNKNOWN = "unknown"
    WALKING = "walking"
    SIGHTSEEING = "sightseeing"
    SWIMMING = "swimming"
    HIKING = "hiking"
    CYCLING = "cycling"
    TRAVELING = "traveling"
    RELAXING = "relaxing"
    SPORTS = "sports"
    DINING = "dining"
    BOATING = "boating"
    ARRIVING = "arriving"
    DEPARTING = "departing"
    OTHER = "other"


class EmotionType(StrEnum):
    NEUTRAL = "neutral"
    JOYFUL = "joyful"
    EXCITING = "exciting"
    RELAXING = "relaxing"
    ROMANTIC = "romantic"
    EMOTIONAL = "emotional"
    ADVENTUROUS = "adventurous"
    CINEMATIC = "cinematic"


class PersonGroup(StrEnum):
    NONE = "none"
    ADULTS = "adults"
    CHILDREN = "children"
    FAMILY = "family"
    GROUP = "group"
    SOLO = "solo"
    MIXED = "mixed"
