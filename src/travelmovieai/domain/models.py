"""Core data contracts for pipeline artifacts."""

from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from travelmovieai.domain.enums import (
    ActivityType,
    EmotionType,
    LocationType,
    MediaType,
    PersonGroup,
    PipelineStage,
    StoryStyle,
)


class MediaAsset(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    path: Path
    relative_path: Path
    media_type: MediaType
    extension: str
    size_bytes: int
    modified_at: datetime
    modified_ns: int
    created_at: datetime | None = None
    duration_seconds: float | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    latitude: float | None = None
    longitude: float | None = None
    probe_metadata: dict[str, Any] = Field(default_factory=dict)
    scan_error: str | None = None


class MediaScanReport(BaseModel):
    input_path: Path
    scanned_at: datetime
    assets: list[MediaAsset] = Field(default_factory=list)
    discovered_count: int = 0
    probed_count: int = 0
    cached_count: int = 0
    error_count: int = 0


class Scene(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    asset_id: UUID
    start_seconds: float
    end_seconds: float
    keyframe_path: Path | None = None
    caption: str | None = None
    transcript: str | None = None
    quality_score: float | None = Field(default=None, ge=0, le=100)
    importance_score: float | None = Field(default=None, ge=0, le=100)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SceneDetectionReport(BaseModel):
    created_at: datetime
    scenes: list[Scene] = Field(default_factory=list)
    detected_count: int = 0
    cached_count: int = 0
    fallback_count: int = 0


class FrameSamplingReport(BaseModel):
    created_at: datetime
    scenes: list[Scene] = Field(default_factory=list)
    extracted_count: int = 0
    cached_count: int = 0


class LandmarkDetection(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    confidence: float = Field(ge=0, le=1)
    evidence: str = Field(default="", max_length=300)


class VisionScoreFactors(BaseModel):
    uniqueness: float = Field(ge=0, le=100)
    people: float = Field(ge=0, le=100)
    emotion: float = Field(ge=0, le=100)
    visual_quality: float = Field(ge=0, le=100)
    landmark: float = Field(ge=0, le=100)
    unusual_event: float = Field(ge=0, le=100)


class SceneUnderstanding(BaseModel):
    caption: str = Field(min_length=1, max_length=500)
    detailed_description: str = Field(min_length=1, max_length=1500)
    location_type: LocationType = LocationType.UNKNOWN
    activity: ActivityType = ActivityType.UNKNOWN
    emotion: EmotionType = EmotionType.NEUTRAL
    people_count: int = Field(default=0, ge=0, le=1000)
    people_groups: list[PersonGroup] = Field(default_factory=list, max_length=6)
    landmarks: list[LandmarkDetection] = Field(default_factory=list, max_length=10)
    vision_score: float = Field(default=50, ge=0, le=100)
    score_factors: VisionScoreFactors
    story_relevance: str = Field(default="", max_length=500)
    tags: list[str] = Field(default_factory=list, max_length=20)


class VisionAnalysisReport(BaseModel):
    created_at: datetime
    provider: str
    model: str
    prompt_version: str
    scenes: list[Scene] = Field(default_factory=list)
    analyzed_count: int = 0
    cached_count: int = 0


class SpeechTranscript(BaseModel):
    text: str
    language: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)


class SpeechAnalysisReport(BaseModel):
    created_at: datetime
    provider: str
    model: str
    scenes: list[Scene] = Field(default_factory=list)
    transcribed_count: int = 0
    cached_count: int = 0


class VisualQualityMetrics(BaseModel):
    brightness: float = Field(ge=0, le=100)
    contrast: float = Field(ge=0, le=100)
    sharpness: float = Field(ge=0, le=100)
    saturation: float = Field(ge=0, le=100)
    colorfulness: float = Field(ge=0, le=100)
    exposure_score: float = Field(default=50, ge=0, le=100)
    noise_score: float = Field(default=0, ge=0, le=100)
    motion_score: float = Field(default=0, ge=0, le=100)
    camera_shake_score: float = Field(default=0, ge=0, le=100)
    quality_score: float = Field(ge=0, le=100)
    panel_quality_scores: list[float] = Field(default_factory=list, max_length=12)
    best_panel_index: int | None = Field(default=None, ge=0)
    best_panel_position: float | None = Field(default=None, ge=0, le=1)
    rejection_reasons: list[str] = Field(default_factory=list)
    backend: str


class QualityAnalysisReport(BaseModel):
    created_at: datetime
    scenes: list[Scene] = Field(default_factory=list)


class DuplicateGroup(BaseModel):
    keeper_scene_id: UUID
    duplicate_scene_ids: list[UUID] = Field(default_factory=list)
    similarity: float = Field(ge=0, le=1)


class DuplicateDetectionReport(BaseModel):
    created_at: datetime
    groups: list[DuplicateGroup] = Field(default_factory=list)
    unique_count: int = 0
    duplicate_count: int = 0


class SceneSelectionDecision(BaseModel):
    scene_id: UUID
    selected: bool
    reason: str
    score: float = Field(ge=0, le=100)


class SceneSelectionReport(BaseModel):
    created_at: datetime
    decisions: list[SceneSelectionDecision] = Field(default_factory=list)


class MusicAccent(BaseModel):
    time_seconds: float = Field(ge=0)
    kind: Literal["intro", "scene_change", "event_change", "highlight", "finale"]
    strength: float = Field(ge=0, le=1)
    scene_id: UUID | None = None
    label: str = Field(default="", max_length=300)


class MusicPlan(BaseModel):
    mode: Literal["none", "manual", "library", "generated"]
    source_path: Path | None = None
    profile: Literal["calm", "lounge", "cinematic", "warm", "energetic"] | None = None
    bpm: int | None = Field(default=None, ge=40, le=180)
    duration_seconds: float | None = Field(default=None, ge=0)
    accents: list[MusicAccent] = Field(default_factory=list)
    arrangement_version: str | None = None
    generator: Literal["procedural", "ace-step", "musicgen"] | None = None
    model: str | None = None
    fallback_used: bool = False
    cache_key: str | None = None
    reasoning: str = ""
    generated: bool = False


class Event(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    title: str
    scene_ids: list[UUID] = Field(default_factory=list)
    summary: str = ""
    importance_score: float = Field(default=0, ge=0, le=100)
    start_at: datetime | None = None
    end_at: datetime | None = None
    location_type: LocationType = LocationType.UNKNOWN
    activity: ActivityType = ActivityType.UNKNOWN
    landmarks: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0, ge=0, le=1)


class EventDetectionReport(BaseModel):
    created_at: datetime
    events: list[Event] = Field(default_factory=list)


class MultimodalSceneDescription(BaseModel):
    scene_id: UUID
    description: str
    vision_caption: str
    transcript: str | None = None
    quality_score: float | None = Field(default=None, ge=0, le=100)
    audio_context: list[str] = Field(default_factory=list)
    source_modalities: list[Literal["vision", "speech", "opencv", "audio"]] = Field(
        default_factory=list
    )


class MultimodalDescriptionReport(BaseModel):
    created_at: datetime
    descriptions: list[MultimodalSceneDescription] = Field(default_factory=list)


class StorySection(BaseModel):
    role: Literal["opening", "journey", "highlight", "finale"]
    title: str
    event_ids: list[UUID] = Field(default_factory=list)
    scene_ids: list[UUID] = Field(default_factory=list)


class Storyboard(BaseModel):
    title: str
    style: StoryStyle
    event_ids: list[UUID] = Field(default_factory=list)
    narration: list[str] = Field(default_factory=list)
    sections: list[StorySection] = Field(default_factory=list)


class TimelineItem(BaseModel):
    scene_id: UUID
    source_start_seconds: float
    source_end_seconds: float
    transition: str | None = None
    title: str | None = None


class Timeline(BaseModel):
    items: list[TimelineItem] = Field(default_factory=list)
    music_path: Path | None = None
    narration_path: Path | None = None


class QuickMontageSettings(BaseModel):
    target_duration_seconds: float = Field(default=90, ge=5, le=3600)
    max_video_clip_seconds: float = Field(default=6, ge=1, le=60)
    photo_duration_seconds: float = Field(default=3, ge=1, le=15)
    width: int = Field(default=1280, ge=320, le=3840)
    height: int = Field(default=720, ge=240, le=2160)
    fps: int = Field(default=30, ge=15, le=60)
    semantic_analysis: bool = False
    quality_analysis: bool = True
    speech_analysis: bool = False
    reject_technical_failures: bool = True
    min_quality_score: float = Field(default=22, ge=0, le=100)
    min_semantic_score: float = Field(default=52, ge=0, le=100)
    duplicate_detection: bool = True
    duplicate_similarity_threshold: float = Field(default=0.92, ge=0.5, le=1)
    max_scenes_per_source: int = Field(default=2, ge=1, le=20)
    max_scenes_per_event: int = Field(default=4, ge=1, le=20)
    story_style: StoryStyle = StoryStyle.CINEMATIC
    vision_provider: Literal["local", "qwen", "florence"] = "local"
    vision_model: str | None = Field(default=None, max_length=300)
    render_device: Literal["auto", "cuda", "cpu"] = "auto"
    scene_threshold: float = Field(default=27, ge=1, le=100)
    min_scene_duration_seconds: float = Field(default=1.5, ge=0.5, le=30)
    max_scene_duration_seconds: float = Field(default=12, ge=2, le=120)
    transition: Literal["none", "fade", "dissolve", "wipeleft", "slideright"] = "fade"
    transition_duration_seconds: float = Field(default=0.5, ge=0, le=3)
    music_enabled: bool = True
    music_mode: Literal["auto", "generated", "library", "manual", "none"] = "auto"
    music_profile: Literal[
        "auto",
        "calm",
        "lounge",
        "cinematic",
        "warm",
        "energetic",
    ] = "auto"
    music_path: Path | None = None
    music_volume: float = Field(default=0.12, ge=0, le=1)
    music_sync: bool = True
    music_engine: Literal["auto", "ace-step", "procedural"] = "auto"
    music_model: str | None = Field(default=None, max_length=300)
    preview_mode: bool = False


class MontageClip(BaseModel):
    asset_id: UUID
    source_path: Path
    relative_path: Path
    media_type: MediaType
    source_start_seconds: float = Field(default=0, ge=0)
    duration_seconds: float = Field(gt=0)
    has_audio: bool = False
    scene_id: UUID | None = None
    caption: str | None = None
    semantic_score: float | None = Field(default=None, ge=0, le=100)
    event_id: UUID | None = None
    selection_reason: str = ""


class QuickMontagePlan(BaseModel):
    created_at: datetime
    settings: QuickMontageSettings
    clips: list[MontageClip] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0, ge=0)
    music_path: Path | None = None
    music_plan: MusicPlan | None = None
    selection_mode: Literal["chronological", "semantic"] = "chronological"


class QuickMontageResult(BaseModel):
    output_path: Path
    timeline_path: Path
    clip_count: int
    duration_seconds: float
    selection_mode: Literal["chronological", "semantic"] = "chronological"
    render_encoder: str | None = None
    music_mode: str | None = None
    music_profile: str | None = None
    music_generator: str | None = None
    music_model: str | None = None


class StageResult(BaseModel):
    stage: PipelineStage
    skipped: bool = False
    artifacts: list[Path] = Field(default_factory=list)
    message: str = ""
