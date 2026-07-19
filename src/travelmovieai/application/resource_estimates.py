"""Deterministic resource estimates derived from project metadata."""

from __future__ import annotations

import math
from collections.abc import Sequence

from pydantic import BaseModel, Field, model_validator

from travelmovieai.analysis.scenes import frame_sample_count_for_mode
from travelmovieai.application.disk_space import estimate_rendered_movie_bytes
from travelmovieai.core.config import Settings
from travelmovieai.domain.enums import MediaType
from travelmovieai.domain.models import MediaAsset, QuickMontageSettings
from travelmovieai.media.proxy import decide_analysis_proxy

_DATABASE_BYTES_PER_ASSET = 4 * 1024
_DATABASE_BYTES_PER_SCENE = 8 * 1024
_ARTIFACT_BYTES_PER_SCENE = 28 * 1024


class RuntimeEstimate(BaseModel):
    """A deliberately broad range rather than a false-precision ETA."""

    lower_seconds: float = Field(ge=0)
    likely_seconds: float = Field(ge=0)
    upper_seconds: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_order(self) -> RuntimeEstimate:
        if not self.lower_seconds <= self.likely_seconds <= self.upper_seconds:
            raise ValueError("runtime bounds must be ordered")
        return self


class ProjectWorkload(BaseModel):
    asset_count: int = Field(ge=0)
    video_count: int = Field(ge=0)
    photo_count: int = Field(ge=0)
    audio_count: int = Field(ge=0)
    source_bytes: int = Field(ge=0)
    video_duration_seconds: float = Field(ge=0)
    estimated_scene_count: int = Field(ge=0)
    proxy_candidate_count: int = Field(ge=0)
    proxy_candidate_duration_seconds: float = Field(ge=0)


class ProjectResourceEstimate(BaseModel):
    """Disk and runtime envelope suitable for large-project preflight UIs."""

    workload: ProjectWorkload
    estimated_proxy_bytes: int = Field(ge=0)
    estimated_frame_cache_bytes: int = Field(ge=0)
    estimated_database_and_artifact_bytes: int = Field(ge=0)
    estimated_analysis_workspace_bytes: int = Field(ge=0)
    estimated_rendered_movie_bytes: int = Field(ge=0)
    estimated_peak_workspace_bytes: int = Field(ge=0)
    runtime: RuntimeEstimate
    assumptions: list[str] = Field(default_factory=list)


def estimate_project_resources(
    assets: Sequence[MediaAsset],
    *,
    settings: Settings,
    montage_settings: QuickMontageSettings,
    known_scene_count: int | None = None,
) -> ProjectResourceEstimate:
    """Estimate an entire local run from bounded metadata, never source contents."""

    if known_scene_count is not None and known_scene_count < 0:
        raise ValueError("known_scene_count cannot be negative")
    videos = [asset for asset in assets if asset.media_type is MediaType.VIDEO]
    photos = [asset for asset in assets if asset.media_type is MediaType.PHOTO]
    audio = [asset for asset in assets if asset.media_type is MediaType.AUDIO]
    video_duration = sum(max(0.0, asset.duration_seconds or 0.0) for asset in videos)
    estimated_scenes = (
        known_scene_count
        if known_scene_count is not None
        else sum(
            max(
                1,
                math.ceil(
                    max(0.0, asset.duration_seconds or 0.0)
                    / montage_settings.max_scene_duration_seconds
                ),
            )
            for asset in videos
        )
        + len(photos)
    )
    proxy_candidates = [
        asset
        for asset in videos
        if decide_analysis_proxy(
            asset,
            mode=settings.analysis_proxy_mode,
            max_dimension=settings.analysis_proxy_max_dimension,
        ).required
    ]
    proxy_duration = sum(max(0.0, asset.duration_seconds or 0.0) for asset in proxy_candidates)
    estimated_proxy_bytes = math.ceil(
        proxy_duration * settings.analysis_proxy_video_bitrate_mbps * 1_000_000 / 8 * 1.08
    )
    sample_count = frame_sample_count_for_mode(montage_settings.analysis_quality_mode)
    contact_sheet_rows = math.ceil(sample_count / 3)
    estimated_frame_cache_bytes = math.ceil(
        estimated_scenes * 1440 * (270 * contact_sheet_rows) * 0.65
    )
    database_and_artifacts = len(assets) * _DATABASE_BYTES_PER_ASSET + estimated_scenes * (
        _DATABASE_BYTES_PER_SCENE + _ARTIFACT_BYTES_PER_SCENE
    )
    analysis_workspace = math.ceil(
        (estimated_proxy_bytes + estimated_frame_cache_bytes + database_and_artifacts) * 1.1
    )
    rendered_movie = estimate_rendered_movie_bytes(montage_settings)
    peak_workspace = analysis_workspace + math.ceil(
        rendered_movie * settings.render_disk_safety_factor
    )
    likely_runtime = _likely_runtime_seconds(
        asset_count=len(assets),
        video_duration=video_duration,
        proxy_duration=proxy_duration,
        scene_count=estimated_scenes,
        montage_settings=montage_settings,
    )
    workload = ProjectWorkload(
        asset_count=len(assets),
        video_count=len(videos),
        photo_count=len(photos),
        audio_count=len(audio),
        source_bytes=sum(max(0, asset.size_bytes) for asset in assets),
        video_duration_seconds=video_duration,
        estimated_scene_count=estimated_scenes,
        proxy_candidate_count=len(proxy_candidates),
        proxy_candidate_duration_seconds=proxy_duration,
    )
    return ProjectResourceEstimate(
        workload=workload,
        estimated_proxy_bytes=estimated_proxy_bytes,
        estimated_frame_cache_bytes=estimated_frame_cache_bytes,
        estimated_database_and_artifact_bytes=database_and_artifacts,
        estimated_analysis_workspace_bytes=analysis_workspace,
        estimated_rendered_movie_bytes=rendered_movie,
        estimated_peak_workspace_bytes=peak_workspace,
        runtime=RuntimeEstimate(
            lower_seconds=round(likely_runtime * 0.45, 1),
            likely_seconds=round(likely_runtime, 1),
            upper_seconds=round(likely_runtime * 3.0, 1),
        ),
        assumptions=[
            "Source media is streamed and is not copied into the workspace.",
            "Proxy estimates use the configured target bitrate plus 8% container overhead.",
            "The estimate describes a cold project run and does not deduct reusable caches.",
            "Shared model-download storage is excluded from the project workspace estimate.",
            "Runtime varies with codec complexity, storage throughput, and model hardware.",
        ],
    )


def _likely_runtime_seconds(
    *,
    asset_count: int,
    video_duration: float,
    proxy_duration: float,
    scene_count: int,
    montage_settings: QuickMontageSettings,
) -> float:
    sample_factor = {"fast": 0.16, "balanced": 0.28, "deep": 0.5}[
        montage_settings.analysis_quality_mode
    ]
    seconds = asset_count * 0.025
    seconds += video_duration * 0.06
    seconds += proxy_duration * 0.42
    seconds += scene_count * sample_factor
    if montage_settings.quality_analysis:
        seconds += scene_count * 0.08
    if montage_settings.semantic_analysis:
        seconds += scene_count * 3.5
    if montage_settings.speech_analysis:
        seconds += video_duration * 0.3
    if montage_settings.audio_analysis:
        seconds += video_duration * 0.04
    seconds += montage_settings.target_duration_seconds * 0.45
    return max(0.0, seconds)
