"""Core data contracts for pipeline artifacts."""

from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator

from travelmovieai.domain.enums import (
    ActivityType,
    EmotionType,
    LocationType,
    MediaType,
    PersonGroup,
    PipelineStage,
    StageStatus,
    StoryStyle,
)

type ShotScale = Literal[
    "unknown",
    "extreme_wide",
    "wide",
    "full",
    "medium",
    "close_up",
    "extreme_close_up",
]
type CameraMotion = Literal[
    "unknown",
    "static",
    "pan",
    "tilt",
    "tracking",
    "handheld",
    "zoom",
    "drone",
    "orbit",
]


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
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    probe_metadata: dict[str, Any] = Field(default_factory=dict)
    scan_error: str | None = None


class MediaScanReport(BaseModel):
    input_path: Path
    scanned_at: datetime
    assets: list[MediaAsset] = Field(default_factory=list)
    discovered_count: int = Field(default=0, ge=0)
    probed_count: int = Field(default=0, ge=0)
    cached_count: int = Field(default=0, ge=0)
    error_count: int = Field(default=0, ge=0)


class Scene(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    asset_id: UUID
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(ge=0)
    keyframe_path: Path | None = None
    caption: str | None = None
    transcript: str | None = None
    quality_score: float | None = Field(default=None, ge=0, le=100)
    importance_score: float | None = Field(default=None, ge=0, le=100)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_time_window(self) -> "Scene":
        if not self.end_seconds > self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")
        return self


class SceneDetectionReport(BaseModel):
    created_at: datetime
    scenes: list[Scene] = Field(default_factory=list)
    detected_count: int = Field(default=0, ge=0)
    cached_count: int = Field(default=0, ge=0)
    fallback_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_counts(self) -> "SceneDetectionReport":
        if self.detected_count + self.cached_count != len(self.scenes):
            raise ValueError("detected_count plus cached_count must match the scene count")
        if self.fallback_count > len(self.scenes):
            raise ValueError("fallback_count cannot exceed the scene count")
        return self


class FrameSamplingReport(BaseModel):
    created_at: datetime
    scenes: list[Scene] = Field(default_factory=list)
    extracted_count: int = Field(default=0, ge=0)
    cached_count: int = Field(default=0, ge=0)


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


class TemporalHighlightWindow(BaseModel):
    """A normalized, source-attributed interval inside one scene."""

    relative_start: float = Field(ge=0, le=1)
    relative_end: float = Field(ge=0, le=1)
    relative_position: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    source: Literal[
        "vision",
        "visual_quality",
        "audio",
        "speech",
        "combined",
        "fallback",
    ]
    score: float | None = Field(default=None, ge=0, le=100)
    label: str = Field(default="", max_length=200)

    @model_validator(mode="after")
    def validate_interval(self) -> "TemporalHighlightWindow":
        if self.relative_end <= self.relative_start:
            raise ValueError("relative_end must be greater than relative_start")
        if not self.relative_start <= self.relative_position <= self.relative_end:
            raise ValueError("relative_position must be inside the highlight interval")
        return self


class SceneUnderstanding(BaseModel):
    caption: str = Field(min_length=1, max_length=500)
    detailed_description: str = Field(min_length=1, max_length=1500)
    location_type: LocationType = LocationType.UNKNOWN
    activity: ActivityType = ActivityType.UNKNOWN
    emotion: EmotionType = EmotionType.NEUTRAL
    shot_scale: ShotScale = "unknown"
    camera_motion: CameraMotion = "unknown"
    focus_x: float | None = Field(default=None, ge=0, le=1)
    focus_y: float | None = Field(default=None, ge=0, le=1)
    focus_source: Literal["face", "object", "subject"] | None = None
    people_count: int = Field(default=0, ge=0, le=1000)
    people_groups: list[PersonGroup] = Field(default_factory=list, max_length=6)
    landmarks: list[LandmarkDetection] = Field(default_factory=list, max_length=10)
    vision_score: float = Field(default=50, ge=0, le=100)
    score_factors: VisionScoreFactors
    story_relevance: str = Field(default="", max_length=500)
    tags: list[str] = Field(default_factory=list, max_length=20)
    highlight_windows: list[TemporalHighlightWindow] = Field(
        default_factory=list,
        max_length=6,
    )

    @model_validator(mode="after")
    def validate_focus_contract(self) -> "SceneUnderstanding":
        focus_fields = (self.focus_x, self.focus_y, self.focus_source)
        if any(value is not None for value in focus_fields) and any(
            value is None for value in focus_fields
        ):
            raise ValueError("focus_x, focus_y, and focus_source must be provided together")
        return self


class VisionAnalysisReport(BaseModel):
    created_at: datetime
    provider: str = Field(min_length=1, max_length=100)
    model: str = Field(min_length=1, max_length=300)
    prompt_version: str = Field(min_length=1, max_length=100)
    scenes: list[Scene] = Field(default_factory=list)
    analyzed_count: int = Field(default=0, ge=0)
    cached_count: int = Field(default=0, ge=0)
    degraded_count: int = Field(default=0, ge=0)
    retry_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_counts(self) -> "VisionAnalysisReport":
        processed_count = self.analyzed_count + self.cached_count + self.degraded_count
        if processed_count > len(self.scenes):
            raise ValueError("Vision result counts cannot exceed the scene count")
        return self


class SceneEmbedding(BaseModel):
    scene_id: UUID
    vector: list[float] = Field(min_length=1, max_length=4096)


class EmbeddingAnalysisReport(BaseModel):
    created_at: datetime
    backend: str = Field(min_length=1, max_length=100)
    model: str | None = None
    dimensions: int = Field(ge=1, le=4096)
    embeddings: list[SceneEmbedding] = Field(default_factory=list)
    index_path: Path | None = None
    indexed_count: int = Field(default=0, ge=0)
    fallback_used: bool = False


class SemanticIndexManifest(BaseModel):
    created_at: datetime
    backend: str
    model: str | None = None
    dimensions: int = Field(ge=1, le=4096)
    scene_ids: list[UUID] = Field(default_factory=list)
    index_path: Path
    source_fingerprint: str = Field(min_length=64, max_length=64)


class SemanticSearchHit(BaseModel):
    scene_id: UUID
    score: float = Field(ge=-1, le=1)
    rank: int = Field(ge=1)


class SemanticSearchReport(BaseModel):
    backend: str
    model: str | None = None
    query: str = Field(min_length=1, max_length=1000)
    hits: list[SemanticSearchHit] = Field(default_factory=list)


class SpeechSegment(BaseModel):
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(ge=0)
    text: str = Field(default="", max_length=1000)
    confidence: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def validate_time_window(self) -> "SpeechSegment":
        if not self.end_seconds > self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")
        return self


class SpeechTranscript(BaseModel):
    text: str
    language: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    segments: list[SpeechSegment] = Field(default_factory=list, max_length=200)


class SpeechAnalysisReport(BaseModel):
    created_at: datetime
    provider: str = Field(min_length=1, max_length=100)
    model: str = Field(min_length=1, max_length=300)
    scenes: list[Scene] = Field(default_factory=list)
    transcribed_count: int = Field(default=0, ge=0)
    cached_count: int = Field(default=0, ge=0)


class AudioSceneAnalysis(BaseModel):
    scene_id: UUID
    has_audio: bool
    primary_label: Literal[
        "speech",
        "silence",
        "wind",
        "music",
        "crowd",
        "water",
        "transport",
        "ambient",
        "unknown",
    ]
    labels: list[str] = Field(default_factory=list, max_length=8)
    rms_dbfs: float | None = None
    peak_dbfs: float | None = None
    zero_crossing_rate: float | None = None
    spectral_centroid_hz: float | None = None
    low_frequency_ratio: float | None = None
    high_frequency_ratio: float | None = None
    dynamic_range_db: float | None = None
    speech_likelihood: float = Field(default=0, ge=0, le=1)
    noise_score: float = Field(default=0, ge=0, le=100)
    ambience_score: float = Field(default=0, ge=0, le=100)
    candidate_windows: list[dict[str, Any]] = Field(default_factory=list, max_length=12)


class AudioAnalysisReport(BaseModel):
    created_at: datetime
    scenes: list[Scene] = Field(default_factory=list)
    analyses: list[AudioSceneAnalysis] = Field(default_factory=list)
    analyzed_count: int = Field(default=0, ge=0)
    skipped_count: int = Field(default=0, ge=0)


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
    sample_count: int = Field(default=1, ge=1, le=9)
    sample_positions: list[float] = Field(default_factory=lambda: [0.5], max_length=9)
    panel_details: list[dict[str, Any]] = Field(default_factory=list, max_length=12)
    candidate_windows: list[TemporalHighlightWindow] = Field(default_factory=list, max_length=12)
    rejection_reasons: list[str] = Field(default_factory=list)
    backend: str = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def validate_sample_positions(self) -> "VisualQualityMetrics":
        if len(self.sample_positions) != self.sample_count:
            raise ValueError("sample_positions length must match sample_count")
        if any(position < 0 or position > 1 for position in self.sample_positions):
            raise ValueError("sample_positions must be normalized to 0..1")
        if any(
            second < first
            for first, second in zip(
                self.sample_positions,
                self.sample_positions[1:],
                strict=False,
            )
        ):
            raise ValueError("sample_positions must be chronological")
        return self


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
    unique_count: int = Field(default=0, ge=0)
    duplicate_count: int = Field(default=0, ge=0)


class SceneSelectionDecision(BaseModel):
    scene_id: UUID
    selected: bool
    reason: str
    score: float = Field(ge=0, le=100)


class SceneSelectionReport(BaseModel):
    created_at: datetime
    decisions: list[SceneSelectionDecision] = Field(default_factory=list)


class MontageQualityIssue(BaseModel):
    severity: Literal["info", "warning", "critical"]
    code: str = Field(min_length=1, max_length=80)
    message: str = Field(min_length=1, max_length=500)
    scene_id: UUID | None = None
    clip_index: int | None = Field(default=None, ge=0)


class RenderedMediaMetrics(BaseModel):
    """Full-duration delivery checks collected from FFmpeg/FFprobe."""

    scan_completed: bool = False
    scan_failure_reason: (
        Literal[
            "not_requested",
            "process_unavailable",
            "timeout",
            "ffmpeg_error",
        ]
        | None
    ) = None
    black_duration_seconds: float | None = Field(default=None, ge=0)
    black_ratio: float | None = Field(default=None, ge=0, le=1)
    freeze_duration_seconds: float | None = Field(default=None, ge=0)
    freeze_ratio: float | None = Field(default=None, ge=0, le=1)
    silence_duration_seconds: float | None = Field(default=None, ge=0)
    silence_ratio: float | None = Field(default=None, ge=0, le=1)
    integrated_loudness_lufs: float | None = None
    loudness_range_lu: float | None = Field(default=None, ge=0)
    true_peak_dbfs: float | None = None
    av_duration_delta_seconds: float | None = Field(default=None, ge=0)


class MontageQualityReport(BaseModel):
    created_at: datetime
    gate_status: Literal["passed", "degraded", "failed"] = "passed"
    score: float = Field(ge=0, le=100)
    target_duration_seconds: float = Field(ge=0)
    planned_duration_seconds: float = Field(ge=0)
    duration_ratio: float = Field(ge=0)
    clip_count: int = Field(ge=0)
    photo_clip_count: int = Field(default=0, ge=0)
    photo_duration_ratio: float = Field(default=0, ge=0, le=1)
    selected_scene_count: int = Field(ge=0)
    selected_event_count: int = Field(ge=0)
    total_event_count: int = Field(ge=0)
    event_coverage_ratio: float = Field(ge=0, le=1)
    source_count: int = Field(ge=0)
    dominant_source_ratio: float = Field(ge=0, le=1)
    dominant_event_ratio: float = Field(default=0, ge=0, le=1)
    dominant_role_ratio: float = Field(default=0, ge=0, le=1)
    adjacent_source_repeat_count: int = Field(default=0, ge=0)
    adjacent_source_repeat_ratio: float = Field(default=0, ge=0, le=1)
    average_semantic_score: float | None = Field(default=None, ge=0, le=100)
    minimum_semantic_score: float | None = Field(default=None, ge=0, le=100)
    semantic_score_p10: float | None = Field(default=None, ge=0, le=100)
    median_semantic_score: float | None = Field(default=None, ge=0, le=100)
    effective_semantic_threshold: float | None = Field(default=None, ge=0, le=100)
    average_quality_score: float | None = Field(default=None, ge=0, le=100)
    minimum_quality_score: float | None = Field(default=None, ge=0, le=100)
    quality_score_p10: float | None = Field(default=None, ge=0, le=100)
    median_quality_score: float | None = Field(default=None, ge=0, le=100)
    window_selection: dict[str, int] = Field(default_factory=dict)
    center_cut_ratio: float = Field(default=0, ge=0, le=1)
    generic_caption_count: int = Field(default=0, ge=0)
    generic_caption_ratio: float = Field(default=0, ge=0, le=1)
    generic_title_count: int = Field(default=0, ge=0)
    music_mode: str | None = None
    music_duration_seconds: float | None = Field(default=None, ge=0)
    music_accent_count: int = Field(default=0, ge=0)
    music_cue_section_count: int = Field(default=0, ge=0)
    music_beat_count: int = Field(default=0, ge=0)
    music_loudness_rms: float | None = Field(default=None, ge=0)
    music_peak_ratio: float | None = Field(default=None, ge=0)
    music_clipping_ratio: float | None = Field(default=None, ge=0, le=1)
    rendered_path: Path | None = None
    rendered_duration_seconds: float | None = Field(default=None, ge=0)
    rendered_duration_delta_seconds: float | None = None
    rendered_has_video: bool | None = None
    rendered_has_audio: bool | None = None
    render_encoder: str | None = None
    rendered_audio_rms: dict[str, float] = Field(default_factory=dict)
    rendered_video_luma: dict[str, float] = Field(default_factory=dict)
    rendered_media_metrics: RenderedMediaMetrics | None = None
    issues: list[MontageQualityIssue] = Field(default_factory=list)

    @model_validator(mode="after")
    def synchronize_gate_status(self) -> "MontageQualityReport":
        if any(issue.severity == "critical" for issue in self.issues):
            expected = "failed"
        elif any(issue.severity == "warning" for issue in self.issues):
            expected = "degraded"
        else:
            expected = "passed"
        object.__setattr__(self, "gate_status", expected)
        return self


class MusicAccent(BaseModel):
    time_seconds: float = Field(ge=0)
    kind: Literal["intro", "scene_change", "event_change", "highlight", "finale"]
    strength: float = Field(ge=0, le=1)
    scene_id: UUID | None = None
    label: str = Field(default="", max_length=300)


class MusicCueSection(BaseModel):
    role: Literal["intro", "journey", "highlight", "finale"]
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(ge=0)
    bpm: int = Field(ge=40, le=180)
    intensity: float = Field(ge=0, le=1)
    accent_count: int = Field(default=0, ge=0)
    description: str = Field(default="", max_length=300)


class MusicBeat(BaseModel):
    time_seconds: float = Field(ge=0)
    beat_index: int = Field(ge=0)
    bar_index: int = Field(ge=0)
    strength: float = Field(ge=0, le=1)
    nearest_accent_kind: str | None = Field(default=None, max_length=40)


class MusicPlan(BaseModel):
    mode: Literal["none", "manual", "library", "generated"]
    source_path: Path | None = None
    source_content_sha256: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    profile: Literal["calm", "lounge", "cinematic", "warm", "energetic"] | None = None
    bpm: int | None = Field(default=None, ge=40, le=180)
    duration_seconds: float | None = Field(default=None, ge=0)
    accents: list[MusicAccent] = Field(default_factory=list)
    cue_sections: list[MusicCueSection] = Field(default_factory=list)
    beat_grid: list[MusicBeat] = Field(default_factory=list)
    arrangement_version: str | None = None
    generator: Literal["procedural", "ace-step", "musicgen"] | None = None
    model: str | None = None
    fallback_used: bool = False
    cache_key: str | None = None
    reasoning: str = ""
    generated: bool = False

    @model_validator(mode="after")
    def validate_generator_model(self) -> "MusicPlan":
        if self.generator in {"ace-step", "musicgen"} and not self.model:
            raise ValueError("Neural music generators require a model identifier")
        if self.generator == "procedural" and self.model is not None:
            raise ValueError("Procedural music cannot declare a model identifier")
        if self.generated:
            if self.mode != "generated":
                raise ValueError("A generated soundtrack must use generated mode")
            if self.source_path is None or self.generator is None:
                raise ValueError("A generated soundtrack requires a source and generator")
            if self.duration_seconds is None or self.duration_seconds <= 0:
                raise ValueError("A generated soundtrack requires a positive duration")
            if self.cache_key is None or len(self.cache_key) != 64:
                raise ValueError("A generated soundtrack requires a 64-character cache key")
            if self.source_content_sha256 is None:
                raise ValueError("A generated soundtrack requires a content fingerprint")
        return self


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


class StoryModelSection(BaseModel):
    role: Literal["opening", "journey", "highlight", "finale"]
    title: str = Field(min_length=1, max_length=200)
    event_ids: list[UUID] = Field(min_length=1)


class StoryModelOutput(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    sections: list[StoryModelSection] = Field(min_length=1, max_length=20)


class Storyboard(BaseModel):
    title: str
    style: StoryStyle
    event_ids: list[UUID] = Field(default_factory=list)
    narration: list[str] = Field(default_factory=list)
    sections: list[StorySection] = Field(default_factory=list)
    provider: str = Field(default="deterministic", min_length=1, max_length=100)
    model: str | None = Field(default=None, max_length=300)
    prompt_version: str | None = Field(default=None, max_length=100)
    fallback_used: bool = False


class NarrationLine(BaseModel):
    section_role: Literal["opening", "journey", "highlight", "finale"]
    text: str = Field(min_length=1, max_length=1000)
    cue_start_seconds: float = Field(ge=0)
    cue_end_seconds: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_cue_window(self) -> "NarrationLine":
        if self.cue_end_seconds <= self.cue_start_seconds:
            raise ValueError("cue_end_seconds must be greater than cue_start_seconds")
        return self


class NarrationReport(BaseModel):
    created_at: datetime
    lines: list[NarrationLine] = Field(default_factory=list)


class NarrationAudioCue(BaseModel):
    line_index: int = Field(ge=0)
    section_role: Literal["opening", "journey", "highlight", "finale"]
    audio_path: Path
    cue_start_seconds: float = Field(ge=0)
    cue_end_seconds: float = Field(gt=0)
    duration_seconds: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_audio_cue(self) -> "NarrationAudioCue":
        if self.cue_end_seconds <= self.cue_start_seconds:
            raise ValueError("cue_end_seconds must be greater than cue_start_seconds")
        expected_end = self.cue_start_seconds + self.duration_seconds
        if abs(expected_end - self.cue_end_seconds) > 0.05:
            raise ValueError("cue_end_seconds must match cue_start_seconds plus duration_seconds")
        return self


class SynthesizedNarrationLine(BaseModel):
    line_index: int = Field(ge=0)
    section_role: Literal["opening", "journey", "highlight", "finale"]
    audio_path: Path
    duration_seconds: float = Field(gt=0)
    sample_rate: int = Field(gt=0)
    channels: int = Field(gt=0, le=8)


class VoiceSynthesisReport(BaseModel):
    created_at: datetime
    provider: str = Field(min_length=1, max_length=100)
    model: str = Field(min_length=1, max_length=300)
    line_count: int = Field(ge=1)
    lines: list[SynthesizedNarrationLine] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_lines(self) -> "VoiceSynthesisReport":
        if self.line_count != len(self.lines):
            raise ValueError("line_count must match the synthesized narration line count")
        if any(line.line_index != index for index, line in enumerate(self.lines)):
            raise ValueError("synthesized narration line indices must be contiguous")
        return self


class TimelineItem(BaseModel):
    scene_id: UUID
    source_start_seconds: float
    source_end_seconds: float
    transition: Literal["cut", "fade", "wipeleft", "slideright"] | None = None
    title: str | None = None


class Timeline(BaseModel):
    items: list[TimelineItem] = Field(default_factory=list)
    music_path: Path | None = None
    narration_path: Path | None = None


class QuickMontageSettings(BaseModel):
    target_duration_seconds: float = Field(default=90, ge=5, le=3600)
    max_video_clip_seconds: float = Field(default=6, ge=1, le=60)
    photo_duration_seconds: float = Field(default=3, ge=1, le=15)
    width: int = Field(default=1280, ge=320, le=3840, multiple_of=2)
    height: int = Field(default=720, ge=240, le=2160, multiple_of=2)
    fps: int = Field(default=30, ge=15, le=60)
    semantic_analysis: bool = False
    quality_analysis: bool = True
    speech_analysis: bool = False
    audio_analysis: bool = True
    reject_technical_failures: bool = True
    min_quality_score: float = Field(default=22, ge=0, le=100)
    min_semantic_score: float = Field(default=52, ge=0, le=100)
    duplicate_detection: bool = True
    duplicate_similarity_threshold: float = Field(default=0.92, ge=0.5, le=1)
    max_scenes_per_source: int = Field(default=2, ge=1, le=20)
    strict_source_diversity: bool = True
    max_scenes_per_event: int = Field(default=4, ge=1, le=20)
    preserve_chronology: bool = True
    chronology_tolerance_seconds: float = Field(default=0, ge=0, le=604800)
    semantic_diversity_weight: float = Field(default=1.0, ge=0, le=3)
    analysis_quality_mode: Literal["fast", "balanced", "deep"] = "balanced"
    story_style: StoryStyle = StoryStyle.CINEMATIC
    vision_provider: Literal["local", "qwen", "florence"] = "local"
    vision_model: str | None = Field(default=None, max_length=300)
    render_device: Literal["auto", "cuda", "cpu"] = "auto"
    framing_mode: Literal["fit", "fill", "smart"] = "fit"
    vertical_video_layout: Literal["fit", "blur", "crop"] = "fit"
    photo_motion: Literal["none", "ken_burns"] = "none"
    photo_zoom_ratio: float = Field(default=1.08, ge=1.0, le=1.35)
    color_normalization: bool = False
    hdr_to_sdr: bool = False
    event_titles_enabled: bool = False
    scene_subtitles_enabled: bool = False
    credits_text: str | None = Field(default=None, max_length=500)
    overlay_safe_margin: float = Field(default=0.05, ge=0.03, le=0.2)
    overlay_max_characters: int = Field(default=160, ge=20, le=500)
    overlay_font_path: Path | None = None
    caption_characters_per_second: float = Field(default=18.0, ge=8, le=30)
    credits_duration_seconds: float = Field(default=3.0, ge=1, le=15)
    scene_threshold: float = Field(default=27, ge=1, le=100)
    min_scene_duration_seconds: float = Field(default=1.5, ge=0.5, le=30)
    max_scene_duration_seconds: float = Field(default=12, ge=2, le=120)
    transition: Literal[
        "none",
        "cinematic",
        "fade",
        "wipeleft",
        "slideright",
    ] = "cinematic"
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
    music_volume: float = Field(default=1.0, ge=0, le=1)
    music_sync: bool = True
    music_bpm_analysis: bool = False
    music_volume_envelope: bool = False
    music_engine: Literal["auto", "ace-step", "procedural"] = "auto"
    music_model: str | None = Field(default=None, max_length=300)
    narration_enabled: bool = False
    narration_volume: float = Field(default=1.0, ge=0, le=2)
    background_volume_during_narration: float = Field(default=0.35, ge=0, le=1)
    source_audio_volume: float = Field(default=0.55, ge=0, le=1)
    source_audio_fade_seconds: float = Field(default=0.08, ge=0, le=2)
    music_fade_seconds: float = Field(default=1.5, ge=0, le=10)
    narration_fade_seconds: float = Field(default=0.08, ge=0, le=2)
    narration_characters_per_second: float = Field(default=14.0, ge=8, le=30)
    final_audio_fade_seconds: float = Field(default=0.35, ge=0, le=5)
    delivery_loudness_lufs: float = Field(default=-16.0, ge=-31, le=-9)
    delivery_true_peak_dbfs: float = Field(default=-1.5, ge=-9, le=-0.1)
    validate_full_render_decode: bool = False
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
    event_title: str | None = Field(default=None, max_length=300)
    source_width: int | None = Field(default=None, gt=0)
    source_height: int | None = Field(default=None, gt=0)
    rotation_degrees: Literal[0, 90, 180, 270] = 0
    color_transfer: str | None = Field(default=None, max_length=80)
    focus_x: float | None = Field(default=None, ge=0, le=1)
    focus_y: float | None = Field(default=None, ge=0, le=1)
    focus_source: Literal["face", "object", "subject", "manual"] | None = None
    brightness_adjustment: float = Field(default=0, ge=-0.25, le=0.25)
    contrast_multiplier: float = Field(default=1, ge=0.75, le=1.25)
    saturation_multiplier: float = Field(default=1, ge=0.75, le=1.25)
    window_source: Literal[
        "vision_highlight",
        "visual_quality",
        "speech",
        "people",
        "center",
        "scene_bounds",
        "other",
    ] = "other"
    selection_reason: str = ""
    transition: Literal["cut", "fade", "wipeleft", "slideright"] | None = None


class QuickMontagePlan(BaseModel):
    created_at: datetime
    settings: QuickMontageSettings
    clips: list[MontageClip] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0, ge=0)
    music_path: Path | None = None
    music_plan: MusicPlan | None = None
    narration_path: Path | None = None
    narration_cues: list[NarrationAudioCue] = Field(default_factory=list)
    selection_mode: Literal["chronological", "semantic"] = "chronological"

    @model_validator(mode="after")
    def validate_narration_cues(self) -> "QuickMontagePlan":
        previous_end = 0.0
        for index, cue in enumerate(self.narration_cues):
            if cue.line_index != index:
                raise ValueError("narration cue line indices must be contiguous")
            if cue.cue_start_seconds < previous_end - 0.01:
                raise ValueError("narration cues must not overlap")
            if cue.cue_end_seconds > self.total_duration_seconds + 0.05:
                raise ValueError("narration cue exceeds the montage duration")
            previous_end = cue.cue_end_seconds
        return self


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
    quality_score: float | None = Field(default=None, ge=0, le=100)
    quality_issue_count: int = Field(default=0, ge=0)
    quality_gate_status: Literal["passed", "degraded", "failed"] | None = None
    semantic_score_p10: float | None = Field(default=None, ge=0, le=100)
    dominant_event_ratio: float | None = Field(default=None, ge=0, le=1)
    adjacent_source_repeat_ratio: float | None = Field(default=None, ge=0, le=1)
    center_cut_ratio: float | None = Field(default=None, ge=0, le=1)
    full_media_qa_completed: bool = False


class StageExecutionMetadata(BaseModel):
    """Allow-listed runtime details that may be persisted in run manifests."""

    retry_count: int = Field(default=0, ge=0)
    fallback_count: int = Field(default=0, ge=0)
    provider: str | None = Field(default=None, min_length=1, max_length=100)
    fallback_provider: str | None = Field(default=None, min_length=1, max_length=100)
    model: str | None = Field(default=None, min_length=1, max_length=300)


class StageResult(BaseModel):
    stage: PipelineStage
    status: StageStatus = StageStatus.COMPLETED
    cache_hit: bool = False
    skipped: bool = False
    artifacts: list[Path] = Field(default_factory=list)
    message: str = ""
    trace: list["StageResult"] = Field(default_factory=list)
    execution: StageExecutionMetadata = Field(default_factory=StageExecutionMetadata)

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_skipped(cls, value: object) -> object:
        """Keep legacy ``skipped=`` callers compatible while exposing a status."""
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        explicit_status = normalized.get("status")
        if explicit_status is None:
            explicit_status = (
                StageStatus.NO_INPUT if normalized.get("skipped", False) else StageStatus.COMPLETED
            )
            normalized["status"] = explicit_status
        status = StageStatus(explicit_status)
        cache_hit = bool(normalized.get("cache_hit", status is StageStatus.CACHED))
        if status is StageStatus.CACHED and not cache_hit:
            raise ValueError("cached status requires cache_hit")
        normalized["cache_hit"] = cache_hit
        expected_skipped = cache_hit or status in {
            StageStatus.CACHED,
            StageStatus.DISABLED,
            StageStatus.NO_INPUT,
        }
        if "skipped" in normalized and bool(normalized["skipped"]) != expected_skipped:
            raise ValueError("skipped must match the explicit stage status")
        normalized["skipped"] = expected_skipped
        return normalized


class StageCacheManifest(BaseModel):
    stage: PipelineStage
    artifact_schema_version: str
    input_fingerprint: str = Field(min_length=64, max_length=64)
    config_fingerprint: str = Field(min_length=64, max_length=64)
    created_at: datetime
    artifacts: list[Path] = Field(default_factory=list)
