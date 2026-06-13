"""Timeline assembly."""

from datetime import UTC, datetime
from pathlib import Path

from travelmovieai.core.exceptions import MontageError
from travelmovieai.domain.enums import MediaType
from travelmovieai.domain.models import (
    MediaAsset,
    MontageClip,
    QuickMontagePlan,
    QuickMontageSettings,
    Scene,
)
from travelmovieai.story.ranking import rank_scenes


def build_quick_montage_plan(
    assets: list[MediaAsset],
    settings: QuickMontageSettings,
    music_path: Path | None = None,
) -> QuickMontagePlan:
    usable = [
        asset
        for asset in assets
        if asset.scan_error is None
        and asset.media_type in {MediaType.VIDEO, MediaType.PHOTO}
        and (
            asset.media_type is MediaType.PHOTO
            or (asset.duration_seconds is not None and asset.duration_seconds > 0)
        )
    ]
    usable.sort(
        key=lambda asset: (
            asset.created_at or asset.modified_at,
            asset.relative_path.as_posix().casefold(),
        )
    )

    clips: list[MontageClip] = []
    effective_duration = 0.0
    transition = _transition_duration(settings)
    for asset in usable:
        remaining = settings.target_duration_seconds - effective_duration
        if remaining < 0.1:
            break
        desired_duration = (
            settings.photo_duration_seconds
            if asset.media_type is MediaType.PHOTO
            else min(asset.duration_seconds or 0, settings.max_video_clip_seconds)
        )
        available_budget = remaining + (transition if clips else 0)
        duration = min(desired_duration, available_budget)
        if duration < 0.1:
            continue

        source_start = 0.0
        if asset.media_type is MediaType.VIDEO and asset.duration_seconds:
            source_start = max(0.0, (asset.duration_seconds - duration) / 2)

        clips.append(
            MontageClip(
                asset_id=asset.id,
                source_path=asset.path,
                relative_path=asset.relative_path,
                media_type=asset.media_type,
                source_start_seconds=source_start,
                duration_seconds=duration,
                has_audio=_has_audio(asset),
            )
        )
        effective_duration += duration - (transition if len(clips) > 1 else 0)

    if not clips:
        raise MontageError("В проекте нет пригодных видео или фотографий для монтажа.")

    return QuickMontagePlan(
        created_at=datetime.now(UTC),
        settings=settings,
        clips=clips,
        total_duration_seconds=_timeline_duration(clips, settings),
        music_path=music_path,
    )


def build_semantic_montage_plan(
    assets: list[MediaAsset],
    scenes: list[Scene],
    settings: QuickMontageSettings,
    music_path: Path | None = None,
) -> QuickMontagePlan:
    assets_by_id = {asset.id: asset for asset in assets}
    selected: list[MontageClip] = []
    effective_duration = 0.0
    transition = _transition_duration(settings)

    for scene in rank_scenes(scenes):
        asset = assets_by_id.get(scene.asset_id)
        if asset is None or asset.scan_error:
            continue
        available = (
            settings.photo_duration_seconds
            if asset.media_type is MediaType.PHOTO
            else scene.end_seconds - scene.start_seconds
        )
        duration = min(available, settings.max_video_clip_seconds)
        remaining = settings.target_duration_seconds - effective_duration
        if selected:
            remaining += transition
        duration = min(duration, remaining)
        if duration < 0.5:
            continue

        source_start = 0.0
        if asset.media_type is MediaType.VIDEO:
            source_start = scene.start_seconds + max(0.0, (available - duration) / 2)
        selected.append(
            MontageClip(
                asset_id=asset.id,
                scene_id=scene.id,
                source_path=asset.path,
                relative_path=asset.relative_path,
                media_type=asset.media_type,
                source_start_seconds=source_start,
                duration_seconds=duration,
                has_audio=_has_audio(asset),
                caption=scene.caption,
                semantic_score=float(scene.metadata.get("ranking_score", 50)),
            )
        )
        effective_duration += duration - (transition if len(selected) > 1 else 0)
        if effective_duration >= settings.target_duration_seconds - 0.05:
            break

    if not selected:
        raise MontageError("AI-анализ не нашёл пригодных сцен для монтажа.")

    selected.sort(
        key=lambda clip: (
            assets_by_id[clip.asset_id].created_at or assets_by_id[clip.asset_id].modified_at,
            clip.relative_path.as_posix().casefold(),
            clip.source_start_seconds,
        )
    )
    return QuickMontagePlan(
        created_at=datetime.now(UTC),
        settings=settings,
        clips=selected,
        total_duration_seconds=_timeline_duration(selected, settings),
        music_path=music_path,
        selection_mode="semantic",
    )


def _has_audio(asset: MediaAsset) -> bool:
    streams = asset.probe_metadata.get("streams", [])
    return any(stream.get("codec_type") == "audio" for stream in streams)


def _timeline_duration(
    clips: list[MontageClip],
    settings: QuickMontageSettings,
) -> float:
    transition = _transition_duration(settings)
    if clips:
        transition = min(transition, min(clip.duration_seconds for clip in clips) * 0.45)
    overlaps = max(0, len(clips) - 1) * transition
    return max(0.0, sum(clip.duration_seconds for clip in clips) - overlaps)


def _transition_duration(settings: QuickMontageSettings) -> float:
    if settings.transition == "none":
        return 0.0
    return settings.transition_duration_seconds
