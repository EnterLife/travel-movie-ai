"""Timeline assembly."""

import math
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
from travelmovieai.story.optimizer import optimize_story_timeline_candidates
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
        raise MontageError("The project has no usable videos or photos to edit.")

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
    candidates = optimize_story_timeline_candidates(
        _story_candidates(ranked, settings),
        assets_by_id,
        settings,
    )
    for scene in candidates:
        asset = assets_by_id.get(scene.asset_id)
        if asset is None or asset.scan_error:
            continue
        available = (
            settings.photo_duration_seconds
            if asset.media_type is MediaType.PHOTO
            else scene.end_seconds - scene.start_seconds
        )
        duration = _directed_clip_duration(scene, available, settings)
        remaining = settings.target_duration_seconds - effective_duration
        if selected:
            remaining += transition
        duration = min(duration, remaining)
        if duration < 0.5:
            continue

        source_start = 0.0
        window_reason = ""
        if asset.media_type is MediaType.VIDEO:
            source_start, window_reason = _best_scene_window(
                scene,
                available_seconds=available,
                duration_seconds=duration,
            )
        selection_reason = _selection_reason(scene)
        if window_reason:
            selection_reason = f"{selection_reason}; {window_reason}"
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
                selection_reason=selection_reason,
            )
        )
        effective_duration += duration - (transition if len(selected) > 1 else 0)
        if effective_duration >= settings.target_duration_seconds - 0.05:
            break

    if not selected:
        raise MontageError("AI analysis did not find any usable scenes for the edit.")

    scenes_by_id = {scene.id: scene for scene in candidates}
    if settings.preserve_chronology:
        selected.sort(
            key=lambda clip: _chronological_timeline_sort_key(
                clip,
                scenes_by_id,
                assets_by_id,
                settings,
            )
        )
    else:
        selected.sort(key=lambda clip: _story_timeline_sort_key(clip, scenes_by_id, assets_by_id))
    selected = _apply_transition_policy(selected, scenes_by_id, settings)
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
    ranked = rank_scenes(scenes)
    semantic_threshold = _semantic_score_threshold(ranked, settings)
    selected = {
        timeline_clip.scene_id: timeline_clip
        for timeline_clip in plan.clips
        if timeline_clip.scene_id is not None
    }
    selected_source_counts: dict[str, int] = {}
    for timeline_clip in plan.clips:
        source_id = str(timeline_clip.asset_id)
        selected_source_counts[source_id] = selected_source_counts.get(source_id, 0) + 1
    source_limit = _source_scene_limit(ranked, settings)
    decisions = []
    for scene in ranked:
        selected_clip = selected.get(scene.id)
        if selected_clip is not None:
            decisions.append(
                SceneSelectionDecision(
                    scene_id=scene.id,
                    selected=True,
                    reason=selected_clip.selection_reason,
                    score=float(scene.metadata.get("ranking_score", 0)),
                )
            )
            continue
        decisions.append(
            SceneSelectionDecision(
                scene_id=scene.id,
                selected=False,
                reason=_rejection_reason(
                    scene,
                    settings,
                    semantic_threshold,
                    selected_source_counts,
                    source_limit,
                ),
                score=float(scene.metadata.get("ranking_score", 0)),
            )
        )
    return SceneSelectionReport(
        created_at=datetime.now(UTC),
        decisions=decisions,
    )


def apply_music_directing(
    plan: QuickMontagePlan,
    scenes: list[Scene] | None = None,
) -> QuickMontagePlan:
    """Nudge clip boundaries toward strong music beats without changing selection."""

    music_plan = plan.music_plan
    if (
        music_plan is None
        or not plan.settings.music_sync
        or len(plan.clips) < 2
        or not music_plan.beat_grid
    ):
        return plan

    clips = list(plan.clips)
    scenes_by_id = {scene.id: scene for scene in scenes or []}
    beat_times = _sync_beat_times(plan)
    if not beat_times:
        return plan

    transition = _transition_duration(plan.settings)
    minimum_duration = max(1.0, transition / 0.45 + 0.05) if transition else 1.0
    changed = False
    for index in range(1, len(clips)):
        starts = _clip_starts(clips, plan.settings)
        boundary = starts[index]
        target = _nearest_sync_beat(boundary, beat_times)
        if target is None:
            continue
        delta = target - boundary
        if abs(delta) < 0.04:
            continue
        adjusted_delta = _bounded_boundary_delta(
            delta,
            previous=clips[index - 1],
            current=clips[index],
            scenes_by_id=scenes_by_id,
            settings=plan.settings,
            minimum_duration=minimum_duration,
        )
        if abs(adjusted_delta) < 0.04:
            continue
        clips[index - 1] = _clip_with_duration(
            clips[index - 1],
            clips[index - 1].duration_seconds + adjusted_delta,
        )
        clips[index] = _clip_with_duration(
            clips[index],
            clips[index].duration_seconds - adjusted_delta,
            reason="music beat start",
        )
        changed = True

    if not changed:
        return plan
    return plan.model_copy(
        update={
            "clips": clips,
            "total_duration_seconds": _timeline_duration(clips, plan.settings),
        }
    )


def _has_audio(asset: MediaAsset) -> bool:
    streams = asset.probe_metadata.get("streams", [])
    return any(stream.get("codec_type") == "audio" for stream in streams)


def _story_timeline_sort_key(
    clip: MontageClip,
    scenes_by_id: dict[UUID, Scene],
    assets_by_id: dict[UUID, MediaAsset],
) -> tuple[int, int, int, object, str, float]:
    scene = scenes_by_id.get(clip.scene_id) if clip.scene_id is not None else None
    asset = assets_by_id[clip.asset_id]
    return (
        _metadata_int(scene, "story_timeline_order", 9999),
        _metadata_int(scene, "story_section_index", 99),
        _metadata_int(scene, "story_role_order", 99),
        asset.created_at or asset.modified_at,
        clip.relative_path.as_posix().casefold(),
        clip.source_start_seconds,
    )


def _chronological_timeline_sort_key(
    clip: MontageClip,
    scenes_by_id: dict[UUID, Scene],
    assets_by_id: dict[UUID, MediaAsset],
    settings: QuickMontageSettings,
) -> tuple[object, ...]:
    scene = scenes_by_id.get(clip.scene_id) if clip.scene_id is not None else None
    asset = assets_by_id[clip.asset_id]
    captured_at = asset.created_at or asset.modified_at
    capture_seconds = _datetime_seconds(captured_at)
    if settings.chronology_tolerance_seconds > 0:
        bucket = math.floor(capture_seconds / settings.chronology_tolerance_seconds)
        return (
            float(bucket),
            _metadata_int(scene, "story_section_index", 99),
            _metadata_int(scene, "story_role_order", 99),
            _metadata_int(scene, "story_timeline_order", 9999),
            capture_seconds,
            clip.relative_path.as_posix().casefold(),
            clip.source_start_seconds,
        )
    return (
        capture_seconds,
        clip.relative_path.as_posix().casefold(),
        clip.source_start_seconds,
        _metadata_int(scene, "story_section_index", 99),
        _metadata_int(scene, "story_role_order", 99),
        _metadata_int(scene, "story_timeline_order", 9999),
    )


def _datetime_seconds(value: datetime) -> float:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.timestamp()


def _metadata_int(scene: Scene | None, key: str, default: int) -> int:
    if scene is None:
        return default
    value = scene.metadata.get(key)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


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


def _clip_starts(
    clips: list[MontageClip],
    settings: QuickMontageSettings,
) -> list[float]:
    transition = _transition_duration(settings)
    if clips:
        transition = min(transition, min(clip.duration_seconds for clip in clips) * 0.45)
    starts: list[float] = []
    elapsed = 0.0
    for index, clip in enumerate(clips):
        starts.append(elapsed)
        if index < len(clips) - 1:
            elapsed += clip.duration_seconds - transition
    return starts


def _sync_beat_times(plan: QuickMontagePlan) -> list[float]:
    music_plan = plan.music_plan
    if music_plan is None:
        return []
    timeline_end_guard = max(0.0, plan.total_duration_seconds - 0.25)
    return [
        beat.time_seconds
        for beat in music_plan.beat_grid
        if 0.25 <= beat.time_seconds <= timeline_end_guard
        and (
            beat.strength >= 0.68
            or beat.nearest_accent_kind in {"scene_change", "event_change", "highlight"}
        )
    ]


def _nearest_sync_beat(boundary_seconds: float, beat_times: list[float]) -> float | None:
    search_radius = 0.34
    nearby = [
        beat_time for beat_time in beat_times if abs(beat_time - boundary_seconds) <= search_radius
    ]
    if not nearby:
        return None
    return min(nearby, key=lambda beat_time: abs(beat_time - boundary_seconds))


def _bounded_boundary_delta(
    delta: float,
    *,
    previous: MontageClip,
    current: MontageClip,
    scenes_by_id: dict[UUID, Scene],
    settings: QuickMontageSettings,
    minimum_duration: float,
) -> float:
    previous_max = _clip_max_duration(previous, scenes_by_id, settings)
    current_max = _clip_max_duration(current, scenes_by_id, settings)
    if delta > 0:
        limit = min(
            previous_max - previous.duration_seconds,
            current.duration_seconds - minimum_duration,
        )
        return max(0.0, min(delta, limit))
    limit = min(
        previous.duration_seconds - minimum_duration,
        current_max - current.duration_seconds,
    )
    return min(0.0, max(delta, -limit))


def _clip_max_duration(
    clip: MontageClip,
    scenes_by_id: dict[UUID, Scene],
    settings: QuickMontageSettings,
) -> float:
    if clip.media_type is MediaType.PHOTO:
        return settings.photo_duration_seconds
    scene = scenes_by_id.get(clip.scene_id) if clip.scene_id is not None else None
    if scene is None:
        return clip.duration_seconds
    available_after_start = max(0.0, scene.end_seconds - clip.source_start_seconds)
    return max(
        clip.duration_seconds,
        min(settings.max_video_clip_seconds, available_after_start),
    )


def _clip_with_duration(
    clip: MontageClip,
    duration_seconds: float,
    *,
    reason: str | None = None,
) -> MontageClip:
    update: dict[str, object] = {"duration_seconds": round(duration_seconds, 3)}
    if reason and reason not in clip.selection_reason:
        separator = "; " if clip.selection_reason else ""
        update["selection_reason"] = f"{clip.selection_reason}{separator}{reason}"
    return clip.model_copy(update=update)


def _directed_clip_duration(
    scene: Scene,
    available_seconds: float,
    settings: QuickMontageSettings,
) -> float:
    duration = min(available_seconds, settings.max_video_clip_seconds)
    if settings.target_duration_seconds < 20:
        return duration
    role = _explicit_story_role(scene) or "journey"
    factor = {
        "opening": 1.08,
        "journey": 1.0,
        "highlight": 0.86,
        "finale": 1.15,
    }.get(role, 1.0)
    return min(available_seconds, settings.max_video_clip_seconds, max(1.0, duration * factor))


def _apply_transition_policy(
    clips: list[MontageClip],
    scenes_by_id: dict[UUID, Scene],
    settings: QuickMontageSettings,
) -> list[MontageClip]:
    if settings.transition == "none":
        return clips
    updated: list[MontageClip] = []
    previous_scene: Scene | None = None
    for index, clip in enumerate(clips):
        scene = scenes_by_id.get(clip.scene_id) if clip.scene_id is not None else None
        transition = (
            None
            if index == 0
            else _transition_for_scene_change(
                previous_scene,
                scene,
                clip,
                settings,
            )
        )
        updated.append(clip.model_copy(update={"transition": transition}))
        previous_scene = scene
    return updated


def _transition_for_scene_change(
    previous_scene: Scene | None,
    scene: Scene | None,
    clip: MontageClip,
    settings: QuickMontageSettings,
) -> str:
    if previous_scene is None or scene is None:
        return settings.transition
    if clip.event_id is not None and clip.event_id != _event_id(previous_scene):
        return "dissolve"
    role = _explicit_story_role(scene)
    if role in {"opening", "finale"}:
        return "fade"
    activity = str(scene.metadata.get("activity", "")).casefold()
    if activity in {"walking", "hiking", "transport", "driving"}:
        return "slideright"
    return settings.transition


def _story_candidates(
    ranked: list[Scene],
    settings: QuickMontageSettings,
) -> list[Scene]:
    semantic_threshold = _semantic_score_threshold(ranked, settings)
    eligible = [scene for scene in ranked if _eligible(scene, settings, semantic_threshold)]
    eligible = _include_story_role_representatives(
        ranked,
        eligible,
        settings,
        semantic_threshold,
    )
    forced = [scene for scene in eligible if scene.metadata.get("selection_override") == "include"]
    selected_ids = {scene.id for scene in forced}
    event_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    source_limit = _source_scene_limit(eligible, settings)
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
            or source_counts.get(source_id, 0) >= source_limit
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
        if source_counts.get(source_id, 0) >= source_limit:
            continue
        ordered.append(scene)
        selected_ids.add(scene.id)
        event_counts[event_id] = event_counts.get(event_id, 0) + 1
        source_counts[source_id] = source_counts.get(source_id, 0) + 1
    return ordered


def _include_story_role_representatives(
    ranked: list[Scene],
    eligible: list[Scene],
    settings: QuickMontageSettings,
    semantic_threshold: float,
) -> list[Scene]:
    explicit_roles = {role for scene in ranked if (role := _explicit_story_role(scene)) is not None}
    if not explicit_roles:
        return eligible

    eligible_ids = {scene.id for scene in eligible}
    eligible_roles = {
        role for scene in eligible if (role := _explicit_story_role(scene)) is not None
    }
    protected = list(eligible)
    floor = max(38.0, semantic_threshold - 12.0)
    for role in sorted(explicit_roles - eligible_roles):
        representative = next(
            (
                scene
                for scene in ranked
                if scene.id not in eligible_ids
                and _explicit_story_role(scene) == role
                and _technical_eligible(scene, settings)
                and _scene_ranking_score(scene) >= floor
            ),
            None,
        )
        if representative is None:
            continue
        protected.append(representative)
        eligible_ids.add(representative.id)
    return protected


def _explicit_story_role(scene: Scene) -> str | None:
    role = str(scene.metadata.get("story_section_role", "")).casefold()
    return role if role in {"opening", "journey", "highlight", "finale"} else None


def _scene_ranking_score(scene: Scene) -> float:
    score = scene.metadata.get("ranking_score")
    if isinstance(score, int | float):
        return float(score)
    return scene.importance_score if scene.importance_score is not None else 50.0


def _source_scene_limit(
    eligible: list[Scene],
    settings: QuickMontageSettings,
) -> int:
    source_count = len({scene.asset_id for scene in eligible})
    if source_count <= 0:
        return settings.max_scenes_per_source
    if settings.strict_source_diversity and source_count > 1:
        return settings.max_scenes_per_source

    effective_clip_duration = max(
        0.5,
        settings.max_video_clip_seconds - _transition_duration(settings),
    )
    estimated_needed_clips = max(
        1,
        math.ceil(settings.target_duration_seconds / effective_clip_duration),
    )
    adaptive_limit = math.ceil(estimated_needed_clips / source_count)
    return max(settings.max_scenes_per_source, adaptive_limit)


def _best_scene_window(
    scene: Scene,
    *,
    available_seconds: float,
    duration_seconds: float,
) -> tuple[float, str]:
    if available_seconds <= duration_seconds + 0.05:
        return scene.start_seconds, ""

    candidates: list[tuple[float, float, str]] = []
    candidates.extend(_speech_segment_candidates(scene, available_seconds, duration_seconds))
    candidates.extend(_explicit_window_candidates(scene, available_seconds, duration_seconds))
    if not candidates:
        candidates.extend(_people_window_candidates(scene, available_seconds, duration_seconds))
    candidates.extend(_quality_panel_candidates(scene, available_seconds, duration_seconds))
    if not candidates:
        middle_start = _clamp_window_start(
            available_seconds * 0.5 - duration_seconds / 2,
            available_seconds,
            duration_seconds,
        )
        return scene.start_seconds + middle_start, "center of scene"

    max_start = max(0.0, available_seconds - duration_seconds)
    best_score, best_start, reason = max(
        candidates,
        key=lambda candidate: (
            _director_window_score(
                scene,
                base_score=candidate[0],
                window_start_seconds=candidate[1],
                duration_seconds=duration_seconds,
            ),
            -abs((candidate[1] + duration_seconds / 2) / available_seconds - 0.52),
        ),
    )
    if best_score < 20:
        best_start = max_start / 2
        reason = "center of scene"
    return scene.start_seconds + _clamp_window_start(
        best_start,
        available_seconds,
        duration_seconds,
    ), reason


def _explicit_window_candidates(
    scene: Scene,
    available_seconds: float,
    duration_seconds: float,
) -> list[tuple[float, float, str]]:
    raw_windows: list[object] = []
    highlights = scene.metadata.get("highlight_windows", [])
    candidates = scene.metadata.get("candidate_windows", [])
    if isinstance(highlights, list):
        raw_windows.extend(highlights)
    if isinstance(candidates, list):
        raw_windows.extend(candidates)
    if not raw_windows:
        return []

    resolved: list[tuple[float, float, str]] = []
    for item in raw_windows:
        if not isinstance(item, dict):
            continue
        start = _window_start_from_metadata(scene, item)
        end = _window_end_from_metadata(scene, item)
        if start is None:
            position = _first_float(item.get("relative_position"), item.get("position"))
            if position is None:
                continue
            start = available_seconds * max(0.0, min(1.0, position)) - duration_seconds / 2
        if end is not None and end > start:
            center = start + (end - start) / 2
            start = center - duration_seconds / 2
        score = _float_value(
            item.get("score"),
            _float_value(item.get("importance_score"), 75.0),
        )
        label = str(item.get("label", "")).strip()
        source = str(item.get("source", "")).strip()
        reason = _candidate_reason(source, label)
        resolved.append(
            (
                score,
                _clamp_window_start(start, available_seconds, duration_seconds),
                reason,
            )
        )
    return resolved


def _speech_segment_candidates(
    scene: Scene,
    available_seconds: float,
    duration_seconds: float,
) -> list[tuple[float, float, str]]:
    candidates: list[tuple[float, float, str]] = []
    for segment in _speech_segments(scene):
        segment_start = _first_float(segment.get("start_seconds"), segment.get("start"))
        segment_end = _first_float(segment.get("end_seconds"), segment.get("end"))
        if segment_start is None or segment_end is None or segment_end <= segment_start:
            continue
        segment_start = max(0.0, segment_start)
        segment_end = min(available_seconds, segment_end)
        center = (segment_start + segment_end) / 2
        padding = min(0.35, max(0.0, (duration_seconds - (segment_end - segment_start)) / 2))
        start = center - duration_seconds / 2
        if segment_end - segment_start < duration_seconds:
            start = min(start, max(0.0, segment_start - padding))
        confidence = _float_value(segment.get("confidence"), 0.78)
        text = str(segment.get("text", "")).strip()
        label = text[:80] if text else "spoken moment"
        candidates.append(
            (
                min(96.0, 74.0 + confidence * 18.0),
                _clamp_window_start(start, available_seconds, duration_seconds),
                f"speech-safe window: {label}",
            )
        )
    return candidates


def _people_window_candidates(
    scene: Scene,
    available_seconds: float,
    duration_seconds: float,
) -> list[tuple[float, float, str]]:
    if not _has_people(scene):
        return []
    middle_start = _clamp_window_start(
        available_seconds * 0.5 - duration_seconds / 2,
        available_seconds,
        duration_seconds,
    )
    people_score = _float_value(
        _vision_score_factors(scene).get("people"),
        86.0,
    )
    return [
        (
            min(96.0, max(88.0, people_score + 8.0)),
            middle_start,
            "people-safe center",
        )
    ]


def _has_people(scene: Scene) -> bool:
    people_count = _float_value(scene.metadata.get("people_count"))
    groups = scene.metadata.get("people_groups", [])
    return people_count > 0 or (
        isinstance(groups, list)
        and any(str(group).strip().casefold() not in {"", "none"} for group in groups)
    )


def _vision_score_factors(scene: Scene) -> dict[object, object]:
    factors = scene.metadata.get("vision_score_factors", {})
    return factors if isinstance(factors, dict) else {}


def _candidate_reason(source: str, label: str) -> str:
    if source == "audio_analysis":
        return "audio candidate" if not label else f"audio candidate: {label[:80]}"
    if source in {"speech", "speech_analysis"}:
        return "speech-safe window" if not label else f"speech-safe window: {label[:80]}"
    if source == "visual_quality":
        return "visual candidate" if not label else f"visual candidate: {label[:80]}"
    return "highlight window" if not label else f"highlight window: {label[:80]}"


def _director_window_score(
    scene: Scene,
    *,
    base_score: float,
    window_start_seconds: float,
    duration_seconds: float,
) -> float:
    reason_bonus = 0.0
    speech_segments = _speech_segments(scene)
    window_end = window_start_seconds + duration_seconds
    if speech_segments:
        reason_bonus += _speech_window_bonus(speech_segments, window_start_seconds, window_end)
        reason_bonus -= _speech_boundary_penalty(
            speech_segments,
            window_start_seconds,
            window_end,
        )
    audio_features = scene.metadata.get("audio_features", {})
    if isinstance(audio_features, dict):
        ambience = _float_value(audio_features.get("ambience_score"), 0.0)
        noise = _float_value(audio_features.get("noise_score"), 0.0)
        reason_bonus += min(6.0, ambience / 100 * 6.0)
        reason_bonus -= max(0.0, noise - 70.0) * 0.22
    return base_score + reason_bonus


def _speech_segments(scene: Scene) -> list[dict[object, object]]:
    raw = scene.metadata.get("speech_segments", [])
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def _speech_window_bonus(
    segments: list[dict[object, object]],
    window_start_seconds: float,
    window_end_seconds: float,
) -> float:
    bonus = 0.0
    for segment in segments:
        start = _first_float(segment.get("start_seconds"), segment.get("start"))
        end = _first_float(segment.get("end_seconds"), segment.get("end"))
        if start is None or end is None or end <= start:
            continue
        if window_start_seconds <= start + 0.12 and window_end_seconds >= end - 0.12:
            bonus += 5.0
    return min(14.0, bonus)


def _speech_boundary_penalty(
    segments: list[dict[object, object]],
    window_start_seconds: float,
    window_end_seconds: float,
) -> float:
    penalty = 0.0
    for segment in segments:
        start = _first_float(segment.get("start_seconds"), segment.get("start"))
        end = _first_float(segment.get("end_seconds"), segment.get("end"))
        if start is None or end is None or end <= start:
            continue
        protected_start = start + 0.16
        protected_end = end - 0.16
        if protected_start < window_start_seconds < protected_end:
            penalty += 38.0
        if protected_start < window_end_seconds < protected_end:
            penalty += 38.0
    return penalty


def _quality_panel_candidates(
    scene: Scene,
    available_seconds: float,
    duration_seconds: float,
) -> list[tuple[float, float, str]]:
    metrics = scene.metadata.get("quality_metrics", {})
    if not isinstance(metrics, dict):
        return []
    metric_candidates = _quality_metric_window_candidates(
        metrics,
        available_seconds,
        duration_seconds,
    )
    if metric_candidates:
        return metric_candidates

    raw_scores = metrics.get("panel_quality_scores", [])
    if not isinstance(raw_scores, list):
        return _best_panel_position_candidate(metrics, available_seconds, duration_seconds)
    scores = [_float_value(score) for score in raw_scores if isinstance(score, int | float)]
    if not scores:
        return _best_panel_position_candidate(metrics, available_seconds, duration_seconds)

    candidates: list[tuple[float, float, str]] = []
    for index, score in enumerate(scores):
        position = _panel_position(index, len(scores))
        start = available_seconds * position - duration_seconds / 2
        candidates.append(
            (
                score,
                _clamp_window_start(start, available_seconds, duration_seconds),
                f"best visual window {index + 1}/{len(scores)}",
            )
        )
    return candidates


def _quality_metric_window_candidates(
    metrics: dict[object, object],
    available_seconds: float,
    duration_seconds: float,
) -> list[tuple[float, float, str]]:
    raw_windows = metrics.get("candidate_windows", [])
    if not isinstance(raw_windows, list):
        return []

    candidates: list[tuple[float, float, str]] = []
    for item in raw_windows:
        if not isinstance(item, dict):
            continue
        position = _first_float(
            item.get("relative_position"),
            item.get("position"),
            item.get("best_panel_position"),
        )
        start = _first_float(
            item.get("relative_start_seconds"),
            item.get("start_offset_seconds"),
            item.get("start"),
        )
        if position is not None:
            start = available_seconds * max(0.0, min(1.0, position)) - duration_seconds / 2
        if start is None:
            continue
        score = _float_value(item.get("score"), 60.0)
        label = str(item.get("label", "")).strip()
        source = str(item.get("source", "")).strip()
        if source == "visual_quality" and label.startswith("visual panel "):
            reason = f"best visual window {label.removeprefix('visual panel ')[:80]}"
        else:
            reason_label = label or source
            reason = (
                "visual candidate" if not reason_label else f"visual candidate: {reason_label[:80]}"
            )
        candidates.append(
            (
                score,
                _clamp_window_start(start, available_seconds, duration_seconds),
                reason,
            )
        )
    return candidates


def _best_panel_position_candidate(
    metrics: dict[object, object],
    available_seconds: float,
    duration_seconds: float,
) -> list[tuple[float, float, str]]:
    position = _first_float(metrics.get("best_panel_position"))
    if position is None:
        return []
    start = available_seconds * max(0.0, min(1.0, position)) - duration_seconds / 2
    score = _float_value(metrics.get("quality_score"), 60.0)
    index = _first_float(metrics.get("best_panel_index"))
    label = "best visual panel" if index is None else f"best visual panel {int(index) + 1}"
    return [
        (
            score,
            _clamp_window_start(start, available_seconds, duration_seconds),
            label,
        )
    ]


def _window_start_from_metadata(scene: Scene, item: dict[object, object]) -> float | None:
    start = _first_float(
        item.get("relative_start_seconds"),
        item.get("start_offset_seconds"),
        item.get("start"),
        item.get("start_seconds"),
    )
    if start is None:
        return None
    if scene.start_seconds <= start <= scene.end_seconds:
        return start - scene.start_seconds
    return max(0.0, start)


def _window_end_from_metadata(scene: Scene, item: dict[object, object]) -> float | None:
    end = _first_float(
        item.get("relative_end_seconds"),
        item.get("end_offset_seconds"),
        item.get("end"),
        item.get("end_seconds"),
    )
    if end is None:
        return None
    if scene.start_seconds <= end <= scene.end_seconds:
        return end - scene.start_seconds
    return max(0.0, end)


def _clamp_window_start(
    start_seconds: float,
    available_seconds: float,
    duration_seconds: float,
) -> float:
    return max(0.0, min(start_seconds, max(0.0, available_seconds - duration_seconds)))


def _first_float(*values: object) -> float | None:
    for value in values:
        if isinstance(value, int | float):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                continue
    return None


def _float_value(value: object, default: float = 0.0) -> float:
    parsed = _first_float(value)
    return parsed if parsed is not None else default


def _panel_position(index: int, count: int) -> float:
    if count == 3:
        return (0.12, 0.5, 0.88)[index]
    if count == 1:
        return 0.5
    return (index + 0.5) / count


def _estimated_needed_clips(settings: QuickMontageSettings) -> int:
    effective_clip_duration = max(
        0.5,
        settings.max_video_clip_seconds - _transition_duration(settings),
    )
    return max(1, math.ceil(settings.target_duration_seconds / effective_clip_duration))


def _semantic_score_threshold(
    ranked: list[Scene],
    settings: QuickMontageSettings,
) -> float:
    scores = [
        float(score)
        for scene in ranked
        if _technical_eligible(scene, settings)
        and scene.metadata.get("selection_override") != "include"
        and isinstance((score := scene.metadata.get("ranking_score")), int | float)
    ]
    if not scores:
        return settings.min_semantic_score

    ordered = sorted(scores, reverse=True)
    estimated_needed = _estimated_needed_clips(settings)
    pool_size = min(
        len(ordered),
        max(estimated_needed, math.ceil(estimated_needed * 1.5)),
    )
    distribution_threshold = ordered[pool_size - 1]
    best_score = ordered[0]
    relative_floor = best_score - 30
    if distribution_threshold >= settings.min_semantic_score:
        return min(78.0, distribution_threshold)
    return max(38.0, min(settings.min_semantic_score, max(distribution_threshold, relative_floor)))


def _technical_eligible(scene: Scene, settings: QuickMontageSettings) -> bool:
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
    return not (
        scene.quality_score is not None and scene.quality_score < settings.min_quality_score
    )


def _eligible(
    scene: Scene,
    settings: QuickMontageSettings,
    semantic_threshold: float,
) -> bool:
    if not _technical_eligible(scene, settings):
        return False
    if scene.metadata.get("selection_override") == "include":
        return True
    ranking_score = scene.metadata.get("ranking_score")
    return not (isinstance(ranking_score, int | float) and ranking_score < semantic_threshold)


def _rejection_reason(
    scene: Scene,
    settings: QuickMontageSettings,
    semantic_threshold: float,
    selected_source_counts: dict[str, int] | None = None,
    source_limit: int | None = None,
) -> str:
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
    if isinstance(ranking_score, int | float) and ranking_score < semantic_threshold:
        return f"semantic score below adaptive {semantic_threshold:.0f}"
    if (
        settings.strict_source_diversity
        and selected_source_counts is not None
        and source_limit is not None
        and selected_source_counts.get(str(scene.asset_id), 0) >= source_limit
    ):
        return f"source diversity limit of {source_limit} scene(s)"
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
