"""Timeline assembly."""

from datetime import UTC, datetime
from uuid import UUID

from travelmovieai.core.exceptions import MontageError
from travelmovieai.domain.enums import MediaType
from travelmovieai.domain.models import (
    MediaAsset,
    MontageClip,
    MusicPlan,
    QuickMontagePlan,
    QuickMontageSettings,
    Scene,
    SceneSelectionDecision,
    SceneSelectionReport,
)
from travelmovieai.story.ranking import rank_scenes


def build_quick_montage_plan(
    assets: list[MediaAsset],
    settings: QuickMontageSettings,
    music_plan: MusicPlan | None = None,
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
        music_path=music_plan.source_path if music_plan else None,
        music_plan=music_plan,
    )


def build_semantic_montage_plan(
    assets: list[MediaAsset],
    scenes: list[Scene],
    settings: QuickMontageSettings,
    music_plan: MusicPlan | None = None,
) -> QuickMontagePlan:
    assets_by_id = {asset.id: asset for asset in assets}
    selected: list[MontageClip] = []
    effective_duration = 0.0
    transition = _transition_duration(settings)

    ranked = rank_scenes(scenes)
    candidates = _story_candidates(ranked, settings)
    for scene in candidates:
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
                event_id=_event_id(scene),
                selection_reason=_selection_reason(scene),
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
        music_path=music_plan.source_path if music_plan else None,
        music_plan=music_plan,
        selection_mode="semantic",
    )


def build_selection_report(
    scenes: list[Scene],
    plan: QuickMontagePlan,
    settings: QuickMontageSettings,
) -> SceneSelectionReport:
    selected = {
        clip.scene_id: clip
        for clip in plan.clips
        if clip.scene_id is not None
    }
    decisions = []
    for scene in rank_scenes(scenes):
        clip = selected.get(scene.id)
        if clip is not None:
            decisions.append(
                SceneSelectionDecision(
                    scene_id=scene.id,
                    selected=True,
                    reason=clip.selection_reason,
                    score=float(scene.metadata.get("ranking_score", 0)),
                )
            )
            continue
        decisions.append(
            SceneSelectionDecision(
                scene_id=scene.id,
                selected=False,
                reason=_rejection_reason(scene, settings),
                score=float(scene.metadata.get("ranking_score", 0)),
            )
        )
    return SceneSelectionReport(
        created_at=datetime.now(UTC),
        decisions=decisions,
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


def _story_candidates(
    ranked: list[Scene],
    settings: QuickMontageSettings,
) -> list[Scene]:
    eligible = [scene for scene in ranked if _eligible(scene, settings)]
    forced = [
        scene
        for scene in eligible
        if scene.metadata.get("selection_override") == "include"
    ]
    selected_ids = {scene.id for scene in forced}
    event_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    ordered = list(forced)
    for scene in forced:
        event_id = str(scene.metadata.get("event_id", scene.id))
        event_counts[event_id] = event_counts.get(event_id, 0) + 1
        source_counts[str(scene.asset_id)] = source_counts.get(str(scene.asset_id), 0) + 1

    for scene in eligible:
        event_id = str(scene.metadata.get("event_id", scene.id))
        source_id = str(scene.asset_id)
        if (
            scene.id in selected_ids
            or event_counts.get(event_id, 0) > 0
            or source_counts.get(source_id, 0) >= settings.max_scenes_per_source
        ):
            continue
        ordered.append(scene)
        selected_ids.add(scene.id)
        event_counts[event_id] = 1
        source_counts[source_id] = source_counts.get(source_id, 0) + 1

    for scene in eligible:
        if scene.id in selected_ids:
            continue
        event_id = str(scene.metadata.get("event_id", scene.id))
        source_id = str(scene.asset_id)
        if event_counts.get(event_id, 0) >= settings.max_scenes_per_event:
            continue
        if source_counts.get(source_id, 0) >= settings.max_scenes_per_source:
            continue
        ordered.append(scene)
        selected_ids.add(scene.id)
        event_counts[event_id] = event_counts.get(event_id, 0) + 1
        source_counts[source_id] = source_counts.get(source_id, 0) + 1
    return ordered


def _eligible(scene: Scene, settings: QuickMontageSettings) -> bool:
    override = str(scene.metadata.get("selection_override", "auto"))
    if override == "exclude":
        return False
    if override == "include":
        return True
    if settings.duplicate_detection and scene.metadata.get("duplicate_status") == "duplicate":
        return False
    technical_reasons = scene.metadata.get("technical_rejection_reasons", [])
    if settings.reject_technical_failures and technical_reasons:
        return False
    if scene.quality_score is not None and scene.quality_score < settings.min_quality_score:
        return False
    ranking_score = scene.metadata.get("ranking_score")
    return not (
        isinstance(ranking_score, int | float) and ranking_score < settings.min_semantic_score
    )


def _rejection_reason(scene: Scene, settings: QuickMontageSettings) -> str:
    override = str(scene.metadata.get("selection_override", "auto"))
    if override == "exclude":
        return "excluded by user"
    if settings.duplicate_detection and scene.metadata.get("duplicate_status") == "duplicate":
        return f"near duplicate of {scene.metadata.get('duplicate_of')}"
    technical = scene.metadata.get("technical_rejection_reasons", [])
    if settings.reject_technical_failures and technical:
        return f"technical rejection: {', '.join(technical)}"
    if scene.quality_score is not None and scene.quality_score < settings.min_quality_score:
        return f"quality below {settings.min_quality_score:.0f}"
    ranking_score = scene.metadata.get("ranking_score")
    if isinstance(ranking_score, int | float) and ranking_score < settings.min_semantic_score:
        return f"semantic score below {settings.min_semantic_score:.0f}"
    return "duration budget or event diversity limit"


def _selection_reason(scene: Scene) -> str:
    if scene.metadata.get("selection_override") == "include":
        return "required by user"
    reasons = scene.metadata.get("ranking_reasons", [])
    return "; ".join(str(reason) for reason in reasons) or "best scene for event"


def _event_id(scene: Scene) -> UUID | None:
    value = scene.metadata.get("event_id")
    try:
        return UUID(str(value)) if value else None
    except ValueError:
        return None
