"""HTTP request and response contracts."""

from datetime import datetime
from enum import StrEnum
from pathlib import Path
from uuid import UUID

from pydantic import BaseModel, Field


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


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "travelmovieai"
