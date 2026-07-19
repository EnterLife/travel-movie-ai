"""Typed contracts for manual edits and timeline version history."""

from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field

from travelmovieai.domain.models import Event, QuickMontagePlan, Scene


class EditableScene(Scene):
    """A scene together with its editable metadata revision."""

    landmarks: list[str] = Field(default_factory=list)
    edit_version: int = Field(default=1, ge=1)


class EditableEvent(Event):
    """An event together with its editable metadata revision."""

    edit_version: int = Field(default=1, ge=1)


class TimelineVersionSnapshot(BaseModel):
    """Immutable timeline snapshot stored in a project database."""

    id: UUID
    created_at: datetime
    phase: Literal["built", "rendered"]
    variant_name: str
    variant_slug: str
    plan: QuickMontagePlan
    output_path: Path | None = None


class TimelineVersionSummary(BaseModel):
    id: UUID
    created_at: datetime
    phase: Literal["built", "rendered"]
    variant_name: str
    variant_slug: str
    output_path: Path | None = None
    clip_count: int = Field(ge=0)
    duration_seconds: float = Field(ge=0)
    selection_mode: Literal["chronological", "semantic"]


class TimelineOrderChange(BaseModel):
    scene_id: UUID
    before_index: int = Field(ge=0)
    after_index: int = Field(ge=0)


class TimelineSettingChange(BaseModel):
    before: Any
    after: Any


class TimelineClipOrderChange(BaseModel):
    clip_key: str = Field(min_length=1)
    before_index: int = Field(ge=0)
    after_index: int = Field(ge=0)


class TimelineClipChange(BaseModel):
    clip_key: str = Field(min_length=1)
    changed_fields: dict[str, TimelineSettingChange] = Field(default_factory=dict)


class TimelineVersionComparison(BaseModel):
    before_version_id: UUID
    after_version_id: UUID
    selected_scene_ids_added: list[UUID] = Field(default_factory=list)
    selected_scene_ids_removed: list[UUID] = Field(default_factory=list)
    order_changes: list[TimelineOrderChange] = Field(default_factory=list)
    settings_changes: dict[str, TimelineSettingChange] = Field(default_factory=dict)
    clip_keys_added: list[str] = Field(default_factory=list)
    clip_keys_removed: list[str] = Field(default_factory=list)
    clip_order_changes: list[TimelineClipOrderChange] = Field(default_factory=list)
    clip_changes: list[TimelineClipChange] = Field(default_factory=list)
    plan_changes: dict[str, TimelineSettingChange] = Field(default_factory=dict)


def summarize_timeline_version(snapshot: TimelineVersionSnapshot) -> TimelineVersionSummary:
    return TimelineVersionSummary(
        id=snapshot.id,
        created_at=snapshot.created_at,
        phase=snapshot.phase,
        variant_name=snapshot.variant_name,
        variant_slug=snapshot.variant_slug,
        output_path=snapshot.output_path,
        clip_count=len(snapshot.plan.clips),
        duration_seconds=snapshot.plan.total_duration_seconds,
        selection_mode=snapshot.plan.selection_mode,
    )


def compare_timeline_versions(
    before: TimelineVersionSnapshot,
    after: TimelineVersionSnapshot,
) -> TimelineVersionComparison:
    """Compare observable selection, order, and render setting decisions."""

    before_order = _selected_scene_order(before.plan)
    after_order = _selected_scene_order(after.plan)
    before_set = set(before_order)
    after_set = set(after_order)
    before_indexes = {scene_id: index for index, scene_id in enumerate(before_order)}
    after_indexes = {scene_id: index for index, scene_id in enumerate(after_order)}
    order_changes = [
        TimelineOrderChange(
            scene_id=scene_id,
            before_index=before_indexes[scene_id],
            after_index=after_indexes[scene_id],
        )
        for scene_id in after_order
        if scene_id in before_indexes and before_indexes[scene_id] != after_indexes[scene_id]
    ]

    before_settings = before.plan.settings.model_dump(mode="json")
    after_settings = after.plan.settings.model_dump(mode="json")
    settings_changes = {
        key: TimelineSettingChange(before=before_settings.get(key), after=after_settings.get(key))
        for key in sorted(before_settings.keys() | after_settings.keys())
        if before_settings.get(key) != after_settings.get(key)
    }
    before_clips = _clip_entries(before.plan)
    after_clips = _clip_entries(after.plan)
    before_clip_map = {key: payload for key, payload in before_clips}
    after_clip_map = {key: payload for key, payload in after_clips}
    before_clip_order = [key for key, _ in before_clips]
    after_clip_order = [key for key, _ in after_clips]
    before_clip_indexes = {key: index for index, key in enumerate(before_clip_order)}
    after_clip_indexes = {key: index for index, key in enumerate(after_clip_order)}
    common_clip_keys = set(before_clip_map) & set(after_clip_map)
    clip_changes = []
    for key in after_clip_order:
        if key not in common_clip_keys:
            continue
        changed_fields = _changed_values(before_clip_map[key], after_clip_map[key])
        if changed_fields:
            clip_changes.append(TimelineClipChange(clip_key=key, changed_fields=changed_fields))
    return TimelineVersionComparison(
        before_version_id=before.id,
        after_version_id=after.id,
        selected_scene_ids_added=[
            scene_id for scene_id in after_order if scene_id not in before_set
        ],
        selected_scene_ids_removed=[
            scene_id for scene_id in before_order if scene_id not in after_set
        ],
        order_changes=order_changes,
        settings_changes=settings_changes,
        clip_keys_added=[key for key in after_clip_order if key not in before_clip_map],
        clip_keys_removed=[key for key in before_clip_order if key not in after_clip_map],
        clip_order_changes=[
            TimelineClipOrderChange(
                clip_key=key,
                before_index=before_clip_indexes[key],
                after_index=after_clip_indexes[key],
            )
            for key in after_clip_order
            if key in before_clip_indexes and before_clip_indexes[key] != after_clip_indexes[key]
        ],
        clip_changes=clip_changes,
        plan_changes=_changed_values(_plan_payload(before.plan), _plan_payload(after.plan)),
    )


def _selected_scene_order(plan: QuickMontagePlan) -> list[UUID]:
    return [clip.scene_id for clip in plan.clips if clip.scene_id is not None]


def _clip_entries(plan: QuickMontagePlan) -> list[tuple[str, dict[str, Any]]]:
    occurrences: dict[str, int] = {}
    entries: list[tuple[str, dict[str, Any]]] = []
    for clip in plan.clips:
        base = f"scene:{clip.scene_id}" if clip.scene_id is not None else f"asset:{clip.asset_id}"
        occurrence = occurrences.get(base, 0) + 1
        occurrences[base] = occurrence
        key = f"{base}#{occurrence}"
        payload = clip.model_dump(
            mode="json",
            exclude={"source_path"},
        )
        entries.append((key, payload))
    return entries


def _plan_payload(plan: QuickMontagePlan) -> dict[str, Any]:
    return {
        "selection_mode": plan.selection_mode,
        "total_duration_seconds": plan.total_duration_seconds,
        "music_file": plan.music_path.name if plan.music_path is not None else None,
        "music_plan": (
            plan.music_plan.model_dump(mode="json", exclude={"source_path"})
            if plan.music_plan is not None
            else None
        ),
        "narration_file": (plan.narration_path.name if plan.narration_path is not None else None),
    }


def _changed_values(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, TimelineSettingChange]:
    return {
        key: TimelineSettingChange(before=before.get(key), after=after.get(key))
        for key in sorted(before.keys() | after.keys())
        if before.get(key) != after.get(key)
    }
