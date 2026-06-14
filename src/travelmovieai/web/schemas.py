"""HTTP request and response contracts."""

from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

from travelmovieai.domain.models import QuickMontageSettings, Scene


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


class AIProviderStatus(BaseModel):
    available: bool
    base_url: str
    configured_model: str
    models: list[ModelOption] = Field(default_factory=list)
    error: str | None = None


class LocalAIStatus(BaseModel):
    available: bool
    configured_model: str
    resolved_model: str
    cache_dir: str
    downloads_enabled: bool
    models: list[ModelOption] = Field(default_factory=list)


class MusicAIStatus(LocalAIStatus):
    runtime_installed: bool = False


class CudaStatusResponse(BaseModel):
    available: bool
    gpu_name: str | None = None
    driver_version: str | None = None
    memory_mb: int | None = None
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


class CapabilitiesResponse(BaseModel):
    default_workspace_root: str
    local_ai: LocalAIStatus
    music_ai: MusicAIStatus
    ai: AIProviderStatus
    cuda: CudaStatusResponse
    resources: ResourceProfileResponse
    opencv_available: bool
    scenedetect_available: bool
    music_modes: list[str]
    render_devices: list[str]


class MovieRequest(BaseModel):
    input_path: str = Field(min_length=1)
    workspace: str | None = None
    settings: QuickMontageSettings = Field(default_factory=QuickMontageSettings)


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
    progress_current: int = 0
    progress_total: int = 0
    progress_percent: float = Field(default=0, ge=0, le=100)
    elapsed_seconds: float = Field(default=0, ge=0)
    eta_seconds: float | None = Field(default=None, ge=0)
    resources: ResourceProfileResponse | None = None
    subtasks: list[JobSubtaskProgress] = Field(default_factory=list)
    logs: list[JobLogEntry] = Field(default_factory=list)
    output_path: Path | None = None
    clip_count: int | None = None
    duration_seconds: float | None = None
    selection_mode: str | None = None
    render_encoder: str | None = None
    music_mode: str | None = None
    music_profile: str | None = None
    music_generator: str | None = None
    music_model: str | None = None


class SceneListResponse(BaseModel):
    scenes: list[Scene] = Field(default_factory=list)


class SceneOverrideRequest(BaseModel):
    input_path: str = Field(min_length=1)
    workspace: str | None = None
    decision: str = Field(pattern=r"^(auto|include|exclude)$")
