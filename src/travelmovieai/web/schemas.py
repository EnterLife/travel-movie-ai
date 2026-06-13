"""HTTP request and response contracts."""

from datetime import datetime
from enum import StrEnum
from pathlib import Path
from uuid import UUID

from pydantic import BaseModel, Field

from travelmovieai.domain.models import QuickMontageSettings


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ScanRequest(BaseModel):
    input_path: str = Field(min_length=1)
    workspace: str | None = None


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


class MovieRequest(BaseModel):
    input_path: str = Field(min_length=1)
    workspace: str | None = None
    settings: QuickMontageSettings = Field(default_factory=QuickMontageSettings)


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
    progress_current: int = 0
    progress_total: int = 0
    output_path: Path | None = None
    clip_count: int | None = None
    duration_seconds: float | None = None
    selection_mode: str | None = None
