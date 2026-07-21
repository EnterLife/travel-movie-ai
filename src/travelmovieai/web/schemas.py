"""HTTP request and response contracts."""

from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator

from travelmovieai.application.variants import validate_variant_name
from travelmovieai.domain.enums import PipelineStage
from travelmovieai.domain.manual_editing import (
    EditableEvent,
    EditableScene,
    TimelineVersionComparison,
    TimelineVersionSnapshot,
    TimelineVersionSummary,
)
from travelmovieai.domain.models import QuickMontageResult, QuickMontageSettings


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    FAILED = "failed"


class ScanRequest(BaseModel):
    input_path: str = Field(min_length=1)
    workspace: str | None = None


class DirectoryDialogRequest(BaseModel):
    purpose: Literal["input", "workspace"]
    initial_path: str | None = None


class DirectoryDialogResponse(BaseModel):
    selected_path: Path | None = None


class ScanJobResponse(BaseModel):
    id: UUID
    status: JobStatus
    input_path: Path
    workspace: Path
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    message: str = ""
    error: str | None = None
    progress_current: int = Field(default=0, ge=0)
    progress_total: int = Field(default=0, ge=0)
    progress_percent: float = Field(default=0, ge=0, le=100)
    persistence_degraded: bool = False


class ScanJobHistory(BaseModel):
    jobs: list[ScanJobResponse] = Field(default_factory=list)


class DependencyStatus(BaseModel):
    name: str
    configured_value: str
    available: bool
    resolved_path: Path | None = None
    version: str | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: str
    service: str = "travelmovieai"
    ready: bool
    ffmpeg: DependencyStatus
    ffprobe: DependencyStatus


class ModelOption(BaseModel):
    id: str
    likely_vision: bool = False
    recommended: bool = False


class LocalAIStatus(BaseModel):
    available: bool
    configured_model: str
    resolved_model: str
    cache_dir: str
    downloads_enabled: bool
    models: list[ModelOption] = Field(default_factory=list)
    reason: str | None = None
    action: str | None = None


class MusicAIStatus(LocalAIStatus):
    runtime_installed: bool = False


class FeatureCapability(BaseModel):
    available: bool
    reason: str | None = None
    action: str | None = None


class CudaStatusResponse(BaseModel):
    available: bool
    gpu_name: str | None = None
    driver_version: str | None = None
    memory_mb: int | None = None
    free_memory_mb: int | None = None
    compute_capability: str | None = None
    ffmpeg_nvenc: bool = False
    opencv_cuda_devices: int = 0
    torch_cuda: bool = False
    torch_version: str | None = None
    note: str | None = None


class ResourceProfileResponse(BaseModel):
    logical_cores: int
    memory_mb: int | None = None
    gpu_name: str | None = None
    gpu_memory_mb: int | None = None
    nvenc: bool
    frame_workers: int
    analysis_workers: int
    render_workers: int
    ffmpeg_threads: int
    model_batch_size: int
    summary: str
    device: str = "cpu"
    resource_mode: str = "balanced"


class CapabilitiesResponse(BaseModel):
    default_workspace_root: str
    local_ai: LocalAIStatus
    music_ai: MusicAIStatus
    speech: FeatureCapability
    narration: FeatureCapability
    cuda: CudaStatusResponse
    resources: ResourceProfileResponse
    opencv_available: bool
    scenedetect_available: bool
    music_modes: list[str]
    render_devices: list[str]
    recommended_render_device: Literal["cuda", "cpu"]
    recommended_resource_mode: Literal["safe", "balanced", "performance"]


class MovieRequest(BaseModel):
    input_path: str = Field(min_length=1)
    workspace: str | None = None
    variant_name: str = Field(default="Default", min_length=1, max_length=80)
    settings: QuickMontageSettings = Field(default_factory=QuickMontageSettings)

    @field_validator("variant_name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return validate_variant_name(value)


class JobLogEntry(BaseModel):
    timestamp: datetime
    level: str = "info"
    phase: str
    message: str
    progress_percent: float = Field(ge=0, le=100)


class JobSubtaskProgress(BaseModel):
    id: str
    label: str
    status: str = Field(pattern=r"^(pending|running|completed|skipped|failed)$")
    progress_percent: float = Field(default=0, ge=0, le=100)
    message: str = ""


class MovieJobResponse(BaseModel):
    id: UUID
    status: JobStatus
    input_path: Path
    workspace: Path
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    message: str = ""
    error: str | None = None
    phase: str = "queued"
    pipeline_stage: PipelineStage | None = None
    progress_current: int = 0
    progress_total: int = 0
    progress_percent: float = Field(default=0, ge=0, le=100)
    elapsed_seconds: float = Field(default=0, ge=0)
    eta_seconds: float | None = Field(default=None, ge=0)
    resources: ResourceProfileResponse | None = None
    subtasks: list[JobSubtaskProgress] = Field(default_factory=list)
    logs: list[JobLogEntry] = Field(default_factory=list)
    output_path: Path | None = None
    variant_name: str = "Default"
    variant_slug: str = "default"
    clip_count: int | None = None
    duration_seconds: float | None = None
    selection_mode: str | None = None
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
    persistence_degraded: bool = False


class MovieJobHistory(BaseModel):
    jobs: list[MovieJobResponse] = Field(default_factory=list)


class MusicCandidateResponse(BaseModel):
    index: int = Field(ge=0)
    seed: int = Field(ge=0)
    score: float = Field(ge=0, le=100)
    technical_score: float = Field(ge=0, le=100)
    structure_score: float = Field(ge=0, le=100)
    style_score: float = Field(ge=0, le=100)
    selected: bool = False
    notes: list[str] = Field(default_factory=list)
    stream_url: str


class MusicCandidateListResponse(BaseModel):
    candidates: list[MusicCandidateResponse] = Field(default_factory=list)


class MovieJobState(BaseModel):
    """Restart-safe state stored locally for one movie job."""

    id: UUID
    status: JobStatus
    input_path: Path
    workspace: Path
    variant_name: str = "Default"
    variant_slug: str = "default"
    settings: QuickMontageSettings
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    phase_started_at: datetime | None = None
    phase_last_progress_at: datetime | None = None
    phase_last_progress_percent: float | None = Field(default=None, ge=0, le=100)
    message: str = ""
    error: str | None = None
    progress_current: int = Field(default=0, ge=0)
    progress_total: int = Field(default=0, ge=0)
    phase: str = "queued"
    pipeline_stage: PipelineStage | None = None
    resources: ResourceProfileResponse | None = None
    subtasks: list[JobSubtaskProgress] = Field(default_factory=list)
    logs: list[JobLogEntry] = Field(default_factory=list)
    result: QuickMontageResult | None = None
    paused_seconds: float = Field(default=0, ge=0)
    persistence_degraded: bool = False


class MovieJobStateHistory(BaseModel):
    schema_version: Literal[1] = 1
    jobs: list[MovieJobState] = Field(default_factory=list)


class SceneListResponse(BaseModel):
    scenes: list[EditableScene] = Field(default_factory=list)
    total: int = Field(default=0, ge=0)
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=120, ge=1)


class SceneOverrideRequest(BaseModel):
    input_path: str = Field(min_length=1)
    workspace: str | None = None
    decision: str | None = Field(default=None, pattern=r"^(auto|include|exclude)$")
    expected_version: int | None = Field(default=None, ge=1)
    caption: str | None = Field(default=None, max_length=500)
    transcript: str | None = Field(default=None, max_length=10_000)
    landmarks: list[Annotated[str, Field(min_length=1, max_length=200)]] | None = Field(
        default=None, max_length=20
    )

    @model_validator(mode="after")
    def validate_edits(self) -> "SceneOverrideRequest":
        edit_fields = {"caption", "transcript", "landmarks"} & self.model_fields_set
        if not edit_fields and self.decision is None:
            raise ValueError("At least one scene change is required.")
        if edit_fields and self.decision is not None:
            raise ValueError("Selection and scene metadata must be edited separately.")
        if edit_fields and self.expected_version is None:
            raise ValueError("expected_version is required for scene metadata edits.")
        return self


class EventListResponse(BaseModel):
    events: list[EditableEvent] = Field(default_factory=list)


class EventPatchRequest(BaseModel):
    input_path: str = Field(min_length=1)
    workspace: str | None = None
    expected_version: int = Field(ge=1)
    title: str | None = Field(default=None, min_length=1, max_length=300)
    summary: str | None = Field(default=None, max_length=2000)
    landmarks: list[Annotated[str, Field(min_length=1, max_length=200)]] | None = Field(
        default=None, max_length=20
    )

    @model_validator(mode="after")
    def require_change(self) -> "EventPatchRequest":
        if not ({"title", "summary", "landmarks"} & self.model_fields_set):
            raise ValueError("At least one event change is required.")
        return self


class ReorderRequest(BaseModel):
    input_path: str = Field(min_length=1)
    workspace: str | None = None
    ordered_ids: list[UUID] = Field(min_length=1)
    expected_versions: dict[UUID, Annotated[int, Field(ge=1)]] = Field(min_length=1)


class TimelineVersionListResponse(BaseModel):
    versions: list[TimelineVersionSummary] = Field(default_factory=list)


class TimelineVersionResponse(BaseModel):
    version: TimelineVersionSnapshot


class TimelineVersionCompareResponse(BaseModel):
    comparison: TimelineVersionComparison
