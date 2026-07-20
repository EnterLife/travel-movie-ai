"""Typed progress and execution records for pipeline orchestration."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from travelmovieai.domain.enums import PipelineStage, StageStatus
from travelmovieai.domain.models import StageExecutionMetadata

LegacyProgressCallback = Callable[[int, int, str], None]


class StageProgress(BaseModel):
    """Progress reported inside one pipeline stage."""

    current: int = Field(ge=0)
    total: int = Field(ge=0)
    unit: str = Field(default="items", min_length=1, max_length=40)
    message: str = ""

    @property
    def fraction(self) -> float:
        if self.total <= 0:
            return 0.0
        return max(0.0, min(1.0, self.current / self.total))


class ProgressEvent(StageProgress):
    """Stage progress mapped onto a weighted, monotonic pipeline total."""

    stage: PipelineStage
    overall_current: int = Field(ge=0)
    overall_total: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_overall_progress(self) -> ProgressEvent:
        if self.overall_current > self.overall_total:
            raise ValueError("overall_current must not exceed overall_total")
        return self


ProgressEventCallback = Callable[[ProgressEvent], None]


class PipelineStageRun(BaseModel):
    stage: PipelineStage
    weight: float = Field(gt=0)
    started_at: datetime
    finished_at: datetime
    duration_seconds: float = Field(ge=0)
    status: StageStatus | Literal["failed"]
    cache_hit: bool = False
    artifact_count: int = Field(default=0, ge=0)
    execution: StageExecutionMetadata = Field(default_factory=StageExecutionMetadata)


class PipelineRunFailure(BaseModel):
    error_type: str = Field(min_length=1, max_length=200)
    message: str = Field(min_length=1, max_length=2000)


class PipelineRunManifest(BaseModel):
    schema_version: Literal[1] = 1
    run_id: UUID
    target: PipelineStage
    status: Literal["running", "completed", "failed"]
    started_at: datetime
    finished_at: datetime | None = None
    duration_seconds: float | None = Field(default=None, ge=0)
    stage_count: int = Field(ge=1)
    completed_stage_count: int = Field(default=0, ge=0)
    total_weight: float = Field(gt=0)
    stages: list[PipelineStageRun] = Field(default_factory=list)
    failure: PipelineRunFailure | None = None


# Weights reflect the dominant work observed in real local runs. They are relative,
# so a prefix pipeline is normalized to only the stages it will execute.
STAGE_WEIGHTS: dict[PipelineStage, float] = {
    PipelineStage.MEDIA_SCAN: 2.0,
    PipelineStage.SCENE_DETECTION: 10.0,
    PipelineStage.FRAME_SAMPLING: 7.0,
    PipelineStage.QUALITY_ANALYSIS: 8.0,
    PipelineStage.VISION_ANALYSIS: 30.0,
    PipelineStage.SPEECH_ANALYSIS: 7.0,
    PipelineStage.AUDIO_ANALYSIS: 5.0,
    PipelineStage.EMBEDDINGS: 8.0,
    PipelineStage.DUPLICATE_DETECTION: 3.0,
    PipelineStage.SCENE_CAPTIONING: 2.0,
    PipelineStage.EVENT_DETECTION: 2.0,
    PipelineStage.STORY_BUILDER: 3.0,
    PipelineStage.SCENE_RANKING: 2.0,
    PipelineStage.MUSIC_SELECTION: 3.0,
    PipelineStage.NARRATION: 2.0,
    PipelineStage.VOICE_SYNTHESIS: 3.0,
    PipelineStage.TIMELINE_BUILDER: 1.0,
    PipelineStage.RENDERING: 15.0,
}

STAGE_UNITS: dict[PipelineStage, str] = {
    PipelineStage.MEDIA_SCAN: "assets",
    PipelineStage.SCENE_DETECTION: "assets",
    PipelineStage.FRAME_SAMPLING: "scenes",
    PipelineStage.QUALITY_ANALYSIS: "scenes",
    PipelineStage.VISION_ANALYSIS: "scenes",
    PipelineStage.SPEECH_ANALYSIS: "scenes",
    PipelineStage.AUDIO_ANALYSIS: "scenes",
    PipelineStage.EMBEDDINGS: "steps",
    PipelineStage.DUPLICATE_DETECTION: "scenes",
    PipelineStage.SCENE_CAPTIONING: "scenes",
    PipelineStage.EVENT_DETECTION: "scenes",
    PipelineStage.STORY_BUILDER: "steps",
    PipelineStage.SCENE_RANKING: "scenes",
    PipelineStage.MUSIC_SELECTION: "steps",
    PipelineStage.NARRATION: "steps",
    PipelineStage.VOICE_SYNTHESIS: "lines",
    PipelineStage.TIMELINE_BUILDER: "steps",
    PipelineStage.RENDERING: "segments",
}


def stage_weight(stage: PipelineStage) -> float:
    return STAGE_WEIGHTS.get(stage, 1.0)


def weighted_stage_ranges(
    stages: Sequence[PipelineStage] = tuple(PipelineStage),
) -> dict[PipelineStage, tuple[float, float]]:
    """Return normalized percentage ranges for the requested stage sequence."""

    total = sum(stage_weight(stage) for stage in stages)
    completed = 0.0
    ranges: dict[PipelineStage, tuple[float, float]] = {}
    for stage in stages:
        start = completed / total * 100
        completed += stage_weight(stage)
        ranges[stage] = (start, completed / total * 100)
    return ranges
