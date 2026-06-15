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
    ranked = rank_scenes(scenes)
    semantic_threshold = _semantic_score_threshold(ranked, settings)
    selected = {clip.scene_id: clip for clip in plan.clips if clip.scene_id is not None}
    decisions = []
    for scene in ranked:
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
                reason=_rejection_reason(scene, settings, semantic_threshold),
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
    semantic_threshold = _semantic_score_threshold(ranked, settings)
    eligible = [scene for scene in ranked if _eligible(scene, settings, semantic_threshold)]
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


def _source_scene_limit(
    eligible: list[Scene],
    settings: QuickMontageSettings,
) -> int:
    source_count = len({scene.asset_id for scene in eligible})
    if source_count <= 0:
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
    candidates.extend(_explicit_window_candidates(scene, available_seconds, duration_seconds))
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
            candidate[0],
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
    raw_windows = scene.metadata.get("highlight_windows", scene.metadata.get("candidate_windows"))
    if not isinstance(raw_windows, list):
        return []

    candidates: list[tuple[float, float, str]] = []
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
        reason = "highlight window" if not label else f"highlight window: {label[:80]}"
        candidates.append(
            (
                score,
                _clamp_window_start(start, available_seconds, duration_seconds),
                reason,
            )
        )
    return candidates


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
                "visual candidate"
                if not reason_label
                else f"visual candidate: {reason_label[:80]}"
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
    label = (
        "best visual panel"
        if index is None
        else f"best visual panel {int(index) + 1}"
    )
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
