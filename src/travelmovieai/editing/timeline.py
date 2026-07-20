"""Timeline assembly."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal
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
    TemporalHighlightWindow,
)
from travelmovieai.story.editorial import clean_caption
from travelmovieai.story.optimizer import optimize_story_timeline_candidates
from travelmovieai.story.ranking import rank_scenes

type _FocusSource = Literal["face", "object", "subject", "manual"]


@dataclass(frozen=True, slots=True)
class _RenderMetadata:
    source_width: int | None
    source_height: int | None
    rotation_degrees: Literal[0, 90, 180, 270]
    color_transfer: str | None
    focus_x: float | None = None
    focus_y: float | None = None
    focus_source: _FocusSource | None = None
    brightness_adjustment: float = 0
    contrast_multiplier: float = 1
    saturation_multiplier: float = 1


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
    for asset in usable:
        remaining = settings.target_duration_seconds - effective_duration
        if remaining < 0.1:
            break
        boundary_overlap = _planned_boundary_overlap(
            clips[-1].event_id if clips else None,
            None,
            settings,
        )
        desired_duration = (
            settings.photo_duration_seconds
            if asset.media_type is MediaType.PHOTO
            else min(asset.duration_seconds or 0, settings.max_video_clip_seconds)
        )
        available_budget = remaining + (boundary_overlap if clips else 0)
        duration = min(desired_duration, available_budget)
        if duration < 0.1:
            continue

        source_start = 0.0
        if asset.media_type is MediaType.VIDEO and asset.duration_seconds:
            source_start = max(0.0, (asset.duration_seconds - duration) / 2)
        render_metadata = _render_metadata(asset)

        clips.append(
            MontageClip(
                asset_id=asset.id,
                source_path=asset.path,
                relative_path=asset.relative_path,
                media_type=asset.media_type,
                source_start_seconds=source_start,
                duration_seconds=duration,
                has_audio=_has_audio(asset),
                source_width=render_metadata.source_width,
                source_height=render_metadata.source_height,
                rotation_degrees=render_metadata.rotation_degrees,
                color_transfer=render_metadata.color_transfer,
                focus_x=render_metadata.focus_x,
                focus_y=render_metadata.focus_y,
                focus_source=render_metadata.focus_source,
                brightness_adjustment=render_metadata.brightness_adjustment,
                contrast_multiplier=render_metadata.contrast_multiplier,
                saturation_multiplier=render_metadata.saturation_multiplier,
            )
        )
        effective_duration += duration - (boundary_overlap if len(clips) > 1 else 0)

    if not clips:
        raise MontageError("The project has no usable videos or photos to edit.")

    clips = _fit_clips_to_target_duration(clips, settings)
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

    ranked = rank_scenes(scenes)
    story_candidates = _story_candidates(ranked, settings)
    candidates = optimize_story_timeline_candidates(
        story_candidates,
        assets_by_id,
        settings,
    )
    candidates = _backfill_candidates_with_hard_caps(
        candidates,
        story_candidates,
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
        event_id = _event_id(scene)
        boundary_overlap = _planned_boundary_overlap(
            selected[-1].event_id if selected else None,
            event_id,
            settings,
        )
        remaining = settings.target_duration_seconds - effective_duration
        if selected:
            remaining += boundary_overlap
        duration = min(duration, remaining)
        if duration < 0.5:
            continue

        source_start = 0.0
        window_reason = ""
        window_source: Literal[
            "vision_highlight",
            "visual_quality",
            "speech",
            "people",
            "center",
            "scene_bounds",
            "other",
        ] = "scene_bounds"
        if asset.media_type is MediaType.VIDEO:
            source_start, window_reason, window_source = _best_scene_window(
                scene,
                available_seconds=available,
                duration_seconds=duration,
            )
        selection_reason = _selection_reason(scene, settings)
        if window_reason:
            selection_reason = f"{selection_reason}; {window_reason}"
        render_metadata = _render_metadata(asset, scene)
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
                caption=clean_caption(
                    scene.caption,
                    max_characters=settings.overlay_max_characters,
                ),
                semantic_score=float(scene.metadata.get("ranking_score", 50)),
                event_id=event_id,
                event_title=_optional_text(scene.metadata.get("event_title")),
                window_source=window_source,
                selection_reason=selection_reason,
                source_width=render_metadata.source_width,
                source_height=render_metadata.source_height,
                rotation_degrees=render_metadata.rotation_degrees,
                color_transfer=render_metadata.color_transfer,
                focus_x=render_metadata.focus_x,
                focus_y=render_metadata.focus_y,
                focus_source=render_metadata.focus_source,
                brightness_adjustment=render_metadata.brightness_adjustment,
                contrast_multiplier=render_metadata.contrast_multiplier,
                saturation_multiplier=render_metadata.saturation_multiplier,
            )
        )
        effective_duration += duration - (boundary_overlap if len(selected) > 1 else 0)
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
    selected = _apply_manual_timeline_order(selected, scenes_by_id)
    selected = _apply_transition_policy(selected, scenes_by_id, settings)
    selected = _fit_clips_to_target_duration(
        selected,
        settings,
        protected_scene_ids={
            scene.id for scene in scenes if scene.metadata.get("selection_override") == "include"
        },
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


def _apply_manual_timeline_order(
    clips: list[MontageClip],
    scenes_by_id: dict[UUID, Scene],
) -> list[MontageClip]:
    """Apply persisted user ordering after automatic story/chronology ordering."""

    ordered = list(clips)
    positions_by_event: dict[str, list[int]] = {}
    for index, clip in enumerate(ordered):
        event_key = _manual_order_event_key(clip, scenes_by_id)
        positions_by_event.setdefault(event_key, []).append(index)

    for positions in positions_by_event.values():
        event_clips = [ordered[index] for index in positions]
        if not any(
            _manual_order_value(clip, scenes_by_id, "manual_scene_order") is not None
            for clip in event_clips
        ):
            continue
        event_clips.sort(
            key=lambda clip: (
                _manual_order_value(clip, scenes_by_id, "manual_scene_order")
                if _manual_order_value(clip, scenes_by_id, "manual_scene_order") is not None
                else len(event_clips),
            )
        )
        for position, clip in zip(positions, event_clips, strict=True):
            ordered[position] = clip

    if any(
        _manual_order_value(clip, scenes_by_id, "manual_event_order") is not None
        for clip in ordered
    ):
        original_positions = {id(clip): index for index, clip in enumerate(ordered)}
        ordered.sort(
            key=lambda clip: (
                _manual_order_value(clip, scenes_by_id, "manual_event_order")
                if _manual_order_value(clip, scenes_by_id, "manual_event_order") is not None
                else len(positions_by_event),
                original_positions[id(clip)],
            )
        )
    return ordered


def _manual_order_event_key(
    clip: MontageClip,
    scenes_by_id: dict[UUID, Scene],
) -> str:
    if clip.event_id is not None:
        return str(clip.event_id)
    scene = scenes_by_id.get(clip.scene_id) if clip.scene_id is not None else None
    if scene is not None and scene.metadata.get("event_id") is not None:
        return str(scene.metadata["event_id"])
    return f"clip:{id(clip)}"


def _manual_order_value(
    clip: MontageClip,
    scenes_by_id: dict[UUID, Scene],
    field: str,
) -> int | None:
    scene = scenes_by_id.get(clip.scene_id) if clip.scene_id is not None else None
    value = scene.metadata.get(field) if scene is not None else None
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def build_selection_report(
    scenes: list[Scene],
    plan: QuickMontagePlan,
    settings: QuickMontageSettings,
) -> SceneSelectionReport:
    ranked = rank_scenes(scenes)
    semantic_threshold = semantic_score_threshold(ranked, settings)
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


def _render_metadata(asset: MediaAsset, scene: Scene | None = None) -> _RenderMetadata:
    stream = _video_stream(asset)
    width = _positive_int(stream.get("width")) or asset.width
    height = _positive_int(stream.get("height")) or asset.height
    rotation = _rotation_degrees(stream.get("rotation_degrees"))
    focus = _scene_focus(scene, width=width, height=height)
    brightness, contrast, saturation = _scene_color_adjustments(scene)
    return _RenderMetadata(
        source_width=width,
        source_height=height,
        rotation_degrees=rotation,
        color_transfer=_optional_text(stream.get("color_transfer")),
        focus_x=focus[0] if focus is not None else None,
        focus_y=focus[1] if focus is not None else None,
        focus_source=focus[2] if focus is not None else None,
        brightness_adjustment=brightness,
        contrast_multiplier=contrast,
        saturation_multiplier=saturation,
    )


def _video_stream(asset: MediaAsset) -> dict[str, object]:
    streams = asset.probe_metadata.get("streams", [])
    if not isinstance(streams, list):
        return {}
    return next(
        (
            stream
            for stream in streams
            if isinstance(stream, dict) and stream.get("codec_type") == "video"
        ),
        {},
    )


def _scene_focus(
    scene: Scene | None,
    *,
    width: int | None,
    height: int | None,
) -> tuple[float, float, _FocusSource] | None:
    if scene is None:
        return None
    direct = _focus_point(scene.metadata.get("focus_point"), width=width, height=height)
    if direct is not None:
        source = str(scene.metadata.get("focus_source", "manual"))
        source_values: dict[str, _FocusSource] = {
            "face": "face",
            "object": "object",
            "subject": "subject",
            "manual": "manual",
        }
        normalized_source = source_values.get(source, "manual")
        return direct[0], direct[1], normalized_source
    focus_sources: tuple[tuple[str, _FocusSource], ...] = (
        ("face_boxes", "face"),
        ("object_boxes", "object"),
        ("subject_boxes", "subject"),
        ("subject_bbox", "subject"),
    )
    for key, source in focus_sources:
        box = _best_box(scene.metadata.get(key))
        point = _box_center(box, width=width, height=height)
        if point is not None:
            return point[0], point[1], source
    return None


def _scene_color_adjustments(scene: Scene | None) -> tuple[float, float, float]:
    if scene is None:
        return 0.0, 1.0, 1.0
    raw_metrics = scene.metadata.get("quality_metrics")
    if not isinstance(raw_metrics, dict):
        return 0.0, 1.0, 1.0
    brightness = _optional_number(raw_metrics.get("brightness"))
    exposure = _optional_number(raw_metrics.get("exposure_score"))
    contrast = _optional_number(raw_metrics.get("contrast"))
    saturation = _optional_number(raw_metrics.get("saturation"))
    luminance_values = [value for value in (brightness, exposure) if value is not None]
    measured_luminance = sum(luminance_values) / len(luminance_values) if luminance_values else 50
    brightness_adjustment = _clamp((50 - measured_luminance) / 400, -0.12, 0.12)
    measured_contrast = contrast if contrast is not None else 50
    measured_saturation = saturation if saturation is not None else 50
    contrast_multiplier = _clamp(1 + (50 - measured_contrast) / 250, 0.8, 1.2)
    saturation_multiplier = _clamp(1 + (50 - measured_saturation) / 250, 0.8, 1.2)
    return (
        round(brightness_adjustment, 4),
        round(contrast_multiplier, 4),
        round(saturation_multiplier, 4),
    )


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _focus_point(
    value: object,
    *,
    width: int | None,
    height: int | None,
) -> tuple[float, float] | None:
    if not isinstance(value, dict):
        return None
    x = _optional_number(value.get("x"))
    y = _optional_number(value.get("y"))
    if x is None or y is None:
        return None
    if x > 1 or y > 1:
        if not width or not height:
            return None
        x /= width
        y /= height
    if 0 <= x <= 1 and 0 <= y <= 1:
        return x, y
    return None


def _best_box(value: object) -> dict[object, object] | None:
    direct = _coerce_box(value)
    if direct is not None:
        return direct
    if not isinstance(value, list):
        return None
    boxes = [box for item in value if (box := _coerce_box(item)) is not None]
    if not boxes:
        return None
    return max(
        boxes,
        key=lambda box: _optional_number(box.get("confidence")) or 0,
    )


def _coerce_box(value: object) -> dict[object, object] | None:
    if isinstance(value, dict):
        return value
    if not isinstance(value, list) or len(value) != 4:
        return None
    coordinates = [_optional_number(item) for item in value]
    if any(coordinate is None for coordinate in coordinates):
        return None
    return dict(zip(("x1", "y1", "x2", "y2"), coordinates, strict=True))


def _box_center(
    box: dict[object, object] | None,
    *,
    width: int | None,
    height: int | None,
) -> tuple[float, float] | None:
    if box is None:
        return None
    x = _optional_number(box.get("x"))
    y = _optional_number(box.get("y"))
    box_width = _optional_number(box.get("width"))
    box_height = _optional_number(box.get("height"))
    if None not in {x, y, box_width, box_height}:
        assert x is not None and y is not None
        assert box_width is not None and box_height is not None
        return _focus_point(
            {"x": x + box_width / 2, "y": y + box_height / 2},
            width=width,
            height=height,
        )
    x1 = _optional_number(box.get("x1"))
    y1 = _optional_number(box.get("y1"))
    x2 = _optional_number(box.get("x2"))
    y2 = _optional_number(box.get("y2"))
    if None in {x1, y1, x2, y2}:
        return None
    assert x1 is not None and y1 is not None and x2 is not None and y2 is not None
    if x2 <= x1 or y2 <= y1:
        return None
    return _focus_point(
        {"x": (x1 + x2) / 2, "y": (y1 + y2) / 2},
        width=width,
        height=height,
    )


def _rotation_degrees(value: object) -> Literal[0, 90, 180, 270]:
    number = _optional_number(value)
    if number is None:
        return 0
    rotation = int(round(number)) % 360
    if rotation == 90:
        return 90
    if rotation == 180:
        return 180
    if rotation == 270:
        return 270
    return 0


def _positive_int(value: object) -> int | None:
    number = _optional_number(value)
    if number is None or number <= 0:
        return None
    return int(number)


def _optional_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return None
        return number if math.isfinite(number) else None
    return None


def _optional_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.split())
    return text or None


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
            _metadata_int(scene, "story_timeline_order", 9999),
            capture_seconds,
            clip.relative_path.as_posix().casefold(),
            clip.source_start_seconds,
            _metadata_int(scene, "story_section_index", 99),
            _metadata_int(scene, "story_role_order", 99),
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
    transition = _bounded_transition_duration(clips, settings)
    overlaps = sum(
        transition for clip in clips[1:] if _resolved_clip_transition(clip, settings) is not None
    )
    return max(0.0, sum(clip.duration_seconds for clip in clips) - overlaps)


def _transition_duration(settings: QuickMontageSettings) -> float:
    if settings.transition == "none":
        return 0.0
    return settings.transition_duration_seconds


def _bounded_transition_duration(
    clips: list[MontageClip],
    settings: QuickMontageSettings,
) -> float:
    transition = _transition_duration(settings)
    if not clips:
        return transition
    return min(transition, min(clip.duration_seconds for clip in clips) * 0.45)


def _resolved_clip_transition(
    clip: MontageClip,
    settings: QuickMontageSettings,
) -> str | None:
    if settings.transition == "none" or settings.transition_duration_seconds <= 0:
        return None
    if settings.transition == "cinematic":
        return "fade" if clip.transition == "fade" else None
    if settings.transition in {"fade", "wipeleft", "slideright"}:
        return settings.transition
    return None


def _clip_starts(
    clips: list[MontageClip],
    settings: QuickMontageSettings,
) -> list[float]:
    transition = _bounded_transition_duration(clips, settings)
    starts: list[float] = []
    elapsed = 0.0
    for index, clip in enumerate(clips):
        starts.append(elapsed)
        if index < len(clips) - 1:
            next_clip = clips[index + 1]
            overlap = (
                transition if _resolved_clip_transition(next_clip, settings) is not None else 0.0
            )
            elapsed += clip.duration_seconds - overlap
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
    pacing_factor, _ = _scene_pacing(scene)
    return min(
        available_seconds,
        settings.max_video_clip_seconds,
        max(1.0, duration * factor * pacing_factor),
    )


def _scene_pacing(scene: Scene) -> tuple[float, str | None]:
    quality_metrics = scene.metadata.get("quality_metrics", {})
    metrics = quality_metrics if isinstance(quality_metrics, dict) else {}
    motion_score = _float_value(metrics.get("motion_score"))
    shake_score = _float_value(metrics.get("camera_shake_score"))
    emotion = str(scene.metadata.get("emotion", "")).strip().casefold()
    activity = str(scene.metadata.get("activity", "")).strip().casefold()
    camera_motion = str(scene.metadata.get("camera_motion", "")).strip().casefold()

    factor = 1.0
    reasons: list[str] = []
    if (
        motion_score >= 68
        or emotion in {"exciting", "adventurous"}
        or activity in {"sports", "cycling", "boating", "traveling"}
        or camera_motion in {"tracking", "pan", "tilt", "drone", "handheld"}
    ):
        factor *= 0.88
        reasons.append("high-energy pacing")

    audio_features = scene.metadata.get("audio_features", {})
    audio = audio_features if isinstance(audio_features, dict) else {}
    primary_audio = str(audio.get("primary_label", "")).strip().casefold()
    noise_score = _float_value(audio.get("noise_score"))
    if primary_audio in {"wind", "transport"} or noise_score >= 72 or shake_score >= 72:
        factor *= 0.92
        reasons.append("noise-trimmed pacing")

    has_speech = bool(scene.transcript and scene.transcript.strip()) or bool(
        _speech_segments(scene)
    )
    speech_likelihood = _float_value(audio.get("speech_likelihood"))
    if has_speech or speech_likelihood >= 0.55:
        factor = max(factor, 0.98)
        reasons.append("speech-paced hold")
    elif _has_people(scene):
        factor = max(factor, 0.96)
        reasons.append("people-paced hold")
    elif emotion in {"relaxing", "romantic", "cinematic"} or activity in {
        "relaxing",
        "sightseeing",
        "dining",
    }:
        factor = min(1.0, factor * 1.08)
        reasons.append("calm pacing")

    return max(0.72, min(1.0, factor)), _pacing_reason(factor, reasons)


def _pacing_reason(factor: float, reasons: list[str]) -> str | None:
    if factor >= 0.995 or not reasons:
        return None
    if "speech-paced hold" in reasons:
        return "pacing: speech hold"
    if "people-paced hold" in reasons:
        return "pacing: people hold"
    if "noise-trimmed pacing" in reasons:
        return "pacing: trim noisy motion"
    return "pacing: high energy"


def _apply_transition_policy(
    clips: list[MontageClip],
    scenes_by_id: dict[UUID, Scene],
    settings: QuickMontageSettings,
) -> list[MontageClip]:
    updated = [clips[0]] if clips else []
    for previous, current in zip(clips, clips[1:], strict=False):
        previous_event = _clip_event_id(previous, scenes_by_id)
        current_event = _clip_event_id(current, scenes_by_id)
        transition = _transition_for_events(previous_event, current_event, settings) or "cut"
        updated.append(current.model_copy(update={"transition": transition}))
    return updated


def _clip_event_id(
    clip: MontageClip,
    scenes_by_id: dict[UUID, Scene],
) -> UUID | None:
    if clip.event_id is not None:
        return clip.event_id
    scene = scenes_by_id.get(clip.scene_id) if clip.scene_id is not None else None
    return _event_id(scene) if scene is not None else None


def _transition_for_events(
    previous_event: UUID | None,
    current_event: UUID | None,
    settings: QuickMontageSettings,
) -> str | None:
    if settings.transition == "none" or settings.transition_duration_seconds <= 0:
        return None
    if settings.transition == "cinematic":
        if (
            previous_event is not None
            and current_event is not None
            and previous_event != current_event
        ):
            return "fade"
        return None
    if settings.transition in {"fade", "wipeleft", "slideright"}:
        return settings.transition
    return None


def _planned_boundary_overlap(
    previous_event: UUID | None,
    current_event: UUID | None,
    settings: QuickMontageSettings,
) -> float:
    if _transition_for_events(previous_event, current_event, settings) is None:
        return 0.0
    return _transition_duration(settings)


def _fit_clips_to_target_duration(
    clips: list[MontageClip],
    settings: QuickMontageSettings,
    *,
    protected_scene_ids: set[UUID] | None = None,
) -> list[MontageClip]:
    adjusted = list(clips)
    protected = protected_scene_ids or set()
    for index in range(len(adjusted) - 1, -1, -1):
        excess = _timeline_duration(adjusted, settings) - settings.target_duration_seconds
        if excess <= 0.05:
            break
        transition = _bounded_transition_duration(adjusted, settings)
        minimum_duration = max(
            0.5,
            transition / 0.45 + 0.01 if transition > 0 else 0.5,
        )
        reducible = adjusted[index].duration_seconds - minimum_duration
        if reducible <= 0:
            continue
        adjusted[index] = _clip_with_duration(
            adjusted[index],
            adjusted[index].duration_seconds - min(excess, reducible),
        )
    while (
        len(adjusted) > 1
        and _timeline_duration(adjusted, settings) - settings.target_duration_seconds > 0.05
    ):
        removable_index = next(
            (
                index
                for index in range(len(adjusted) - 1, -1, -1)
                if adjusted[index].scene_id not in protected
            ),
            None,
        )
        if removable_index is None:
            break
        adjusted.pop(removable_index)
        adjusted = _apply_transition_policy(adjusted, {}, settings)
    return adjusted


def _story_candidates(
    ranked: list[Scene],
    settings: QuickMontageSettings,
) -> list[Scene]:
    semantic_threshold = semantic_score_threshold(ranked, settings)
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

    if _estimated_candidate_duration(ordered, settings) < settings.target_duration_seconds - 0.05:
        for scene in eligible:
            if scene.id in selected_ids:
                continue
            source_id = str(scene.asset_id)
            if source_counts.get(source_id, 0) >= source_limit:
                continue
            event_id = str(scene.metadata.get("event_id", scene.id))
            if event_counts.get(event_id, 0) >= settings.max_scenes_per_event:
                continue
            ordered.append(scene)
            selected_ids.add(scene.id)
            event_counts[event_id] = event_counts.get(event_id, 0) + 1
            source_counts[source_id] = source_counts.get(source_id, 0) + 1
            if _estimated_candidate_duration(ordered, settings) >= (
                settings.target_duration_seconds - 0.05
            ):
                break
    return ordered


def _estimated_candidate_duration(
    scenes: list[Scene],
    settings: QuickMontageSettings,
) -> float:
    if not scenes:
        return 0.0
    duration = 0.0
    previous_event: UUID | None = None
    for index, scene in enumerate(scenes):
        available = max(0.0, scene.end_seconds - scene.start_seconds)
        duration += _directed_clip_duration(scene, available, settings)
        if index > 0:
            duration -= _planned_boundary_overlap(
                previous_event,
                _event_id(scene),
                settings,
            )
        previous_event = _event_id(scene)
    return max(0.0, duration)


def _backfill_candidates_with_hard_caps(
    selected: list[Scene],
    pool: list[Scene],
    settings: QuickMontageSettings,
) -> list[Scene]:
    """Fill pacing-induced duration deficits without bypassing event/source limits."""

    if _estimated_candidate_duration(selected, settings) >= settings.target_duration_seconds - 0.05:
        return selected
    result = list(selected)
    selected_ids = {scene.id for scene in result}
    event_counts: dict[str, int] = {}
    source_counts: dict[UUID, int] = {}
    for scene in result:
        event_key = str(scene.metadata.get("event_id", scene.id))
        event_counts[event_key] = event_counts.get(event_key, 0) + 1
        source_counts[scene.asset_id] = source_counts.get(scene.asset_id, 0) + 1
    for scene in pool:
        if scene.id in selected_ids:
            continue
        event_key = str(scene.metadata.get("event_id", scene.id))
        if event_counts.get(event_key, 0) >= settings.max_scenes_per_event:
            continue
        if (
            settings.strict_source_diversity
            and source_counts.get(scene.asset_id, 0) >= settings.max_scenes_per_source
        ):
            continue
        result.append(scene)
        selected_ids.add(scene.id)
        event_counts[event_key] = event_counts.get(event_key, 0) + 1
        source_counts[scene.asset_id] = source_counts.get(scene.asset_id, 0) + 1
        if _estimated_candidate_duration(result, settings) >= (
            settings.target_duration_seconds - 0.05
        ):
            break
    return result


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
) -> tuple[
    float,
    str,
    Literal[
        "vision_highlight",
        "visual_quality",
        "speech",
        "people",
        "center",
        "scene_bounds",
        "other",
    ],
]:
    if available_seconds <= duration_seconds + 0.05:
        return scene.start_seconds, "", "scene_bounds"

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
        return scene.start_seconds + middle_start, "center of scene", "center"

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
    return (
        scene.start_seconds
        + _clamp_window_start(
            best_start,
            available_seconds,
            duration_seconds,
        ),
        reason,
        _window_source_from_reason(reason),
    )


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
        if "relative_start" in item or "relative_end" in item:
            try:
                item = TemporalHighlightWindow.model_validate(item).model_dump()
            except ValueError:
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
        confidence = _first_float(item.get("confidence"))
        score = _float_value(
            item.get("score"),
            _float_value(
                item.get("importance_score"),
                (confidence * 100) if confidence is not None else 75.0,
            ),
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


def _window_source_from_reason(
    reason: str,
) -> Literal[
    "vision_highlight",
    "visual_quality",
    "speech",
    "people",
    "center",
    "scene_bounds",
    "other",
]:
    if reason.startswith("speech-safe"):
        return "speech"
    if reason.startswith("people-safe"):
        return "people"
    if reason.startswith(("visual candidate", "best visual")):
        return "visual_quality"
    if reason.startswith("highlight window"):
        return "vision_highlight"
    if reason == "center of scene":
        return "center"
    return "other"


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
        if "relative_start" in item or "relative_end" in item:
            try:
                item = TemporalHighlightWindow.model_validate(item).model_dump()
            except ValueError:
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
        confidence = _first_float(item.get("confidence"))
        score = _float_value(
            item.get("score"),
            (confidence * 100) if confidence is not None else 60.0,
        )
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


def _window_start_from_metadata(
    scene: Scene,
    item: Mapping[Any, Any],
) -> float | None:
    relative_start = _first_float(item.get("relative_start"))
    if relative_start is not None:
        if not 0 <= relative_start <= 1:
            return None
        return max(0.0, scene.end_seconds - scene.start_seconds) * relative_start
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


def _window_end_from_metadata(
    scene: Scene,
    item: Mapping[Any, Any],
) -> float | None:
    relative_end = _first_float(item.get("relative_end"))
    if relative_end is not None:
        if not 0 <= relative_end <= 1:
            return None
        return max(0.0, scene.end_seconds - scene.start_seconds) * relative_end
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


def semantic_score_threshold(
    ranked: list[Scene],
    settings: QuickMontageSettings,
) -> float:
    """Return the adaptive score floor used by automatic semantic selection."""

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


def _selection_reason(scene: Scene, settings: QuickMontageSettings) -> str:
    if scene.metadata.get("selection_override") == "include":
        return "required by user"
    reasons = scene.metadata.get("ranking_reasons", [])
    rendered_reasons = [str(reason) for reason in reasons]
    if settings.target_duration_seconds >= 20:
        _, pacing_reason = _scene_pacing(scene)
        if pacing_reason:
            rendered_reasons.append(pacing_reason)
    return "; ".join(rendered_reasons) or "best scene for event"


def _event_id(scene: Scene) -> UUID | None:
    value = scene.metadata.get("event_id")
    try:
        return UUID(str(value)) if value else None
    except ValueError:
        return None
