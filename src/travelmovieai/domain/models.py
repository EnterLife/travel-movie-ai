"""Core data contracts for pipeline artifacts."""

from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from travelmovieai.domain.enums import MediaType, PipelineStage, StoryStyle


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


class Event(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    title: str
    scene_ids: list[UUID] = Field(default_factory=list)
    summary: str = ""
    importance_score: float = Field(default=0, ge=0, le=100)


class Storyboard(BaseModel):
    title: str
    style: StoryStyle
    event_ids: list[UUID] = Field(default_factory=list)
    narration: list[str] = Field(default_factory=list)


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


class StageResult(BaseModel):
    stage: PipelineStage
    skipped: bool = False
    artifacts: list[Path] = Field(default_factory=list)
    message: str = ""
