"""Story-aware timeline metadata, budgets, and ordering helpers."""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC
from uuid import UUID

from travelmovieai.domain.enums import MediaType
from travelmovieai.domain.models import (
    MediaAsset,
    QuickMontageSettings,
    Scene,
    Storyboard,
)

ROLE_ORDER = {
    "opening": 0,
    "journey": 1,
    "highlight": 2,
    "finale": 3,
}
ROLE_BUDGET_RATIOS = {
    "opening": 0.14,
    "journey": 0.38,
    "highlight": 0.34,
    "finale": 0.14,
}
STORY_ROLES = ("opening", "journey", "highlight", "finale")
_BUDGET_EPSILON_SECONDS = 0.05


@dataclass(frozen=True)
class SelectionCaps:
    """Typed automatic-selection limits; explicit user includes may exceed them."""

    max_scenes_per_event: int
    max_scenes_per_source: int | None


def apply_story_structure(
    scenes: list[Scene],
    storyboard: Storyboard,
) -> list[Scene]:
    """Annotate scenes with storyboard section metadata for timeline ordering."""

    section_by_scene: dict[str, tuple[int, str, str]] = {}
    for index, section in enumerate(storyboard.sections):
        for scene_id in section.scene_ids:
            section_by_scene[str(scene_id)] = (index, section.role, section.title)

    updated: list[Scene] = []
    for scene in scenes:
        section_data = section_by_scene.get(str(scene.id))
        if section_data is None:
            updated.append(scene)
            continue
        index, role, title = section_data
        updated.append(
            scene.model_copy(
                update={
                    "metadata": {
                        **scene.metadata,
                        "story_section_index": index,
                        "story_section_role": role,
                        "story_section_title": title,
                        "story_role_order": ROLE_ORDER.get(role, 99),
                    }
                }
            )
        )
    return updated


def optimize_story_timeline_candidates(
    scenes: list[Scene],
    assets_by_id: dict[UUID, MediaAsset],
    settings: QuickMontageSettings,
) -> list[Scene]:
    """Select and order scenes by story budgets while reducing adjacent repeats."""

    if not scenes:
        return []

    roles = {_story_role(scene) for scene in scenes}
    budgets = _role_budgets(settings.target_duration_seconds, roles)
    caps = SelectionCaps(
        max_scenes_per_event=settings.max_scenes_per_event,
        max_scenes_per_source=(
            settings.max_scenes_per_source if settings.strict_source_diversity else None
        ),
    )
    forced = [scene for scene in scenes if scene.metadata.get("selection_override") == "include"]
    forced_ids = {scene.id for scene in forced}
    available = [scene for scene in scenes if scene.id not in forced_ids]
    selected: list[Scene] = []
    selected_ids: set[UUID] = set()
    used_by_role = {role: 0.0 for role in STORY_ROLES}
    event_counts: dict[str, int] = {}
    source_counts: dict[UUID, int] = {}

    for scene in sorted(forced, key=lambda item: _chronology_key(item, assets_by_id)):
        _record_selection(
            scene,
            selected=selected,
            selected_ids=selected_ids,
            event_counts=event_counts,
            source_counts=source_counts,
            used_by_role=used_by_role,
            assets_by_id=assets_by_id,
            settings=settings,
        )

    for role in STORY_ROLES:
        role_pool = [
            scene
            for scene in available
            if scene.id not in selected_ids and _story_role(scene) == role
        ]
        role_selected = _select_role_scenes(
            role_pool,
            assets_by_id,
            settings,
            budget_seconds=budgets.get(role, 0.0),
            selected_so_far=selected,
            event_counts=event_counts,
            source_counts=source_counts,
            caps=caps,
            used_seconds=used_by_role[role],
        )
        for scene in role_selected:
            _record_selection(
                scene,
                selected=selected,
                selected_ids=selected_ids,
                event_counts=event_counts,
                source_counts=source_counts,
                used_by_role=used_by_role,
                assets_by_id=assets_by_id,
                settings=settings,
            )

    selected_duration = _estimated_timeline_duration(selected, assets_by_id, settings)
    remaining_pool = [scene for scene in scenes if scene.id not in selected_ids]
    while selected_duration < settings.target_duration_seconds - 0.05 and remaining_pool:
        eligible = [
            scene
            for scene in remaining_pool
            if _within_caps(scene, event_counts, source_counts, caps)
            and _within_role_budget(
                scene,
                used_by_role,
                budgets,
            )
        ]
        if not eligible:
            break
        scene = _best_role_deficit_scene(
            eligible,
            selected,
            used_by_role,
            budgets,
            settings,
        )
        _record_selection(
            scene,
            selected=selected,
            selected_ids=selected_ids,
            event_counts=event_counts,
            source_counts=source_counts,
            used_by_role=used_by_role,
            assets_by_id=assets_by_id,
            settings=settings,
        )
        remaining_pool = [item for item in remaining_pool if item.id != scene.id]
        selected_duration = _estimated_timeline_duration(selected, assets_by_id, settings)

    ordered = _order_selected_scenes(selected, assets_by_id, settings)
    return [
        scene.model_copy(
            update={
                "metadata": {
                    **scene.metadata,
                    "story_timeline_order": index,
                    "story_section_role": _story_role(scene),
                    "story_role_order": ROLE_ORDER[_story_role(scene)],
                    "story_section_budget_seconds": budgets.get(_story_role(scene), 0.0),
                    "story_section_used_seconds": used_by_role.get(_story_role(scene), 0.0),
                    "story_diversity_signature": _diversity_signature(scene),
                    "story_selection_caps": {
                        "max_scenes_per_event": caps.max_scenes_per_event,
                        "max_scenes_per_source": caps.max_scenes_per_source,
                    },
                    "story_cap_override": scene.id in forced_ids,
                }
            }
        )
        for index, scene in enumerate(ordered)
    ]


def _select_role_scenes(
    pool: list[Scene],
    assets_by_id: dict[UUID, MediaAsset],
    settings: QuickMontageSettings,
    *,
    budget_seconds: float,
    selected_so_far: list[Scene],
    event_counts: dict[str, int],
    source_counts: dict[UUID, int],
    caps: SelectionCaps,
    used_seconds: float,
) -> list[Scene]:
    selected: list[Scene] = []
    used = used_seconds
    local_event_counts = dict(event_counts)
    local_source_counts = dict(source_counts)
    remaining = list(pool)
    while remaining:
        eligible = [
            scene
            for scene in remaining
            if _within_caps(scene, local_event_counts, local_source_counts, caps)
            and (
                used <= 0
                or used + _estimated_scene_duration(scene, assets_by_id, settings)
                <= budget_seconds + _BUDGET_EPSILON_SECONDS
            )
        ]
        if not eligible:
            break
        scene = _best_diverse_scene(eligible, [*selected_so_far, *selected], settings)
        duration = _estimated_scene_duration(scene, assets_by_id, settings)
        selected.append(scene)
        used += duration
        _increment_counts(scene, local_event_counts, local_source_counts)
        remaining = [item for item in remaining if item.id != scene.id]
        if used >= budget_seconds:
            break
    return selected


def _record_selection(
    scene: Scene,
    *,
    selected: list[Scene],
    selected_ids: set[UUID],
    event_counts: dict[str, int],
    source_counts: dict[UUID, int],
    used_by_role: dict[str, float],
    assets_by_id: dict[UUID, MediaAsset],
    settings: QuickMontageSettings,
) -> None:
    selected.append(scene)
    selected_ids.add(scene.id)
    _increment_counts(scene, event_counts, source_counts)
    used_by_role[_story_role(scene)] += _estimated_scene_duration(
        scene,
        assets_by_id,
        settings,
    )


def _increment_counts(
    scene: Scene,
    event_counts: dict[str, int],
    source_counts: dict[UUID, int],
) -> None:
    event_key = _event_key(scene)
    event_counts[event_key] = event_counts.get(event_key, 0) + 1
    source_counts[scene.asset_id] = source_counts.get(scene.asset_id, 0) + 1


def _within_caps(
    scene: Scene,
    event_counts: dict[str, int],
    source_counts: dict[UUID, int],
    caps: SelectionCaps,
) -> bool:
    if event_counts.get(_event_key(scene), 0) >= caps.max_scenes_per_event:
        return False
    return caps.max_scenes_per_source is None or (
        source_counts.get(scene.asset_id, 0) < caps.max_scenes_per_source
    )


def _within_role_budget(
    scene: Scene,
    used_by_role: dict[str, float],
    budgets: dict[str, float],
) -> bool:
    role = _story_role(scene)
    return used_by_role.get(role, 0.0) < budgets.get(role, 0.0) + _BUDGET_EPSILON_SECONDS


def _best_role_deficit_scene(
    pool: list[Scene],
    selected: list[Scene],
    used_by_role: dict[str, float],
    budgets: dict[str, float],
    settings: QuickMontageSettings,
) -> Scene:
    previous = selected[-1] if selected else None
    recent = selected[-3:]

    def key(scene: Scene) -> tuple[float, float, int]:
        role = _story_role(scene)
        budget = max(budgets.get(role, 0.0), 0.001)
        deficit = max(0.0, budget - used_by_role.get(role, 0.0)) / budget
        adjusted_score = _ranking_score(scene) - (
            _diversity_penalty(scene, previous, recent) * settings.semantic_diversity_weight
        )
        return deficit, adjusted_score, -ROLE_ORDER[role]

    return max(pool, key=key)


def _best_diverse_scene(
    pool: list[Scene],
    selected: list[Scene],
    settings: QuickMontageSettings,
) -> Scene:
    previous = selected[-1] if selected else None
    recent = selected[-3:]
    return max(
        pool,
        key=lambda scene: (
            _ranking_score(scene)
            - _diversity_penalty(scene, previous, recent) * settings.semantic_diversity_weight,
            -ROLE_ORDER[_story_role(scene)],
            -_story_order_key(scene)[1],
        ),
    )


def _role_budgets(
    target_duration_seconds: float,
    roles: Iterable[str],
) -> dict[str, float]:
    active_roles = [role for role in STORY_ROLES if role in set(roles)]
    if not active_roles:
        active_roles = ["journey"]
    ratio_sum = sum(ROLE_BUDGET_RATIOS[role] for role in active_roles)
    return {
        role: target_duration_seconds * ROLE_BUDGET_RATIOS[role] / ratio_sum
        for role in active_roles
    }


def _order_selected_scenes(
    selected: list[Scene],
    assets_by_id: dict[UUID, MediaAsset],
    settings: QuickMontageSettings,
) -> list[Scene]:
    ordered: list[Scene] = []
    for role in STORY_ROLES:
        role_scenes = [scene for scene in selected if _story_role(scene) == role]
        ordered.extend(_order_role_scenes(role_scenes, assets_by_id, settings))
    return ordered


def _order_role_scenes(
    scenes: list[Scene],
    assets_by_id: dict[UUID, MediaAsset],
    settings: QuickMontageSettings,
) -> list[Scene]:
    chronological = sorted(scenes, key=lambda scene: _chronology_key(scene, assets_by_id))
    if len(chronological) < 2:
        return chronological
    if not settings.preserve_chronology:
        return _greedy_diverse_order(chronological, settings)

    events: dict[str, list[Scene]] = {}
    for scene in chronological:
        events.setdefault(_event_key(scene), []).append(scene)
    return [
        scene
        for event_scenes in events.values()
        for scene in _greedy_source_order(event_scenes, assets_by_id, settings)
    ]


def _greedy_diverse_order(
    scenes: list[Scene],
    settings: QuickMontageSettings,
) -> list[Scene]:
    remaining = list(scenes)
    ordered: list[Scene] = []
    while remaining:
        scene = _best_diverse_scene(remaining, ordered, settings)
        ordered.append(scene)
        remaining = [item for item in remaining if item.id != scene.id]
    return ordered


def _greedy_source_order(
    scenes: list[Scene],
    assets_by_id: dict[UUID, MediaAsset],
    settings: QuickMontageSettings,
) -> list[Scene]:
    """Diversify one event while retaining order inside every source file."""

    chronological = sorted(scenes, key=lambda item: _chronology_key(item, assets_by_id))
    queues: dict[UUID, list[Scene]] = {}
    for scene in chronological:
        queues.setdefault(scene.asset_id, []).append(scene)
    first = chronological[0]
    ordered = [first]
    queues[first.asset_id].pop(0)
    if not queues[first.asset_id]:
        del queues[first.asset_id]
    while queues:
        heads = [queue[0] for queue in queues.values()]
        scene = _best_diverse_scene(heads, ordered, settings)
        ordered.append(scene)
        queue = queues[scene.asset_id]
        queue.pop(0)
        if not queue:
            del queues[scene.asset_id]
    return ordered


def _chronology_key(
    scene: Scene,
    assets_by_id: dict[UUID, MediaAsset],
) -> tuple[float, float, str]:
    asset = assets_by_id.get(scene.asset_id)
    captured_at = asset.created_at or asset.modified_at if asset is not None else None
    if captured_at is None:
        timestamp = 0.0
    else:
        if captured_at.tzinfo is None:
            captured_at = captured_at.replace(tzinfo=UTC)
        timestamp = captured_at.astimezone(UTC).timestamp()
    return timestamp + scene.start_seconds, scene.start_seconds, str(scene.id)


def _event_key(scene: Scene) -> str:
    value = scene.metadata.get("event_id")
    return str(value) if value else str(scene.id)


def _story_role(scene: Scene) -> str:
    role = str(scene.metadata.get("story_section_role", "")).casefold()
    if role in ROLE_ORDER:
        return role
    score = _ranking_score(scene)
    if score >= 82 or (scene.importance_score is not None and scene.importance_score >= 85):
        return "highlight"
    return "journey"


def _story_order_key(scene: Scene) -> tuple[int, float]:
    return ROLE_ORDER[_story_role(scene)], -_ranking_score(scene)


def _ranking_score(scene: Scene) -> float:
    score = scene.metadata.get("ranking_score")
    if isinstance(score, int | float):
        return float(score)
    if scene.importance_score is not None:
        return scene.importance_score
    return 50.0


def _estimated_scene_duration(
    scene: Scene,
    assets_by_id: dict[UUID, MediaAsset],
    settings: QuickMontageSettings,
) -> float:
    asset = assets_by_id.get(scene.asset_id)
    if asset is None:
        available = max(0.0, scene.end_seconds - scene.start_seconds)
        return min(settings.max_video_clip_seconds, available)
    if asset.media_type is MediaType.PHOTO:
        return settings.photo_duration_seconds
    available = max(0.0, scene.end_seconds - scene.start_seconds)
    return min(available, settings.max_video_clip_seconds)


def _estimated_timeline_duration(
    scenes: list[Scene],
    assets_by_id: dict[UUID, MediaAsset],
    settings: QuickMontageSettings,
) -> float:
    if not scenes:
        return 0.0
    durations = [_estimated_scene_duration(scene, assets_by_id, settings) for scene in scenes]
    return max(0.0, sum(durations))


def _diversity_penalty(
    scene: Scene,
    previous: Scene | None,
    recent: list[Scene],
) -> float:
    penalty = 0.0
    if previous is not None:
        penalty += _pair_penalty(scene, previous, adjacent=True)
    for item in recent[:-1]:
        penalty += _pair_penalty(scene, item, adjacent=False)
    return penalty


def _pair_penalty(scene: Scene, other: Scene, *, adjacent: bool) -> float:
    multiplier = 1.0 if adjacent else 0.45
    penalty = 0.0
    if scene.asset_id == other.asset_id:
        penalty += 18 * multiplier
    location = _metadata_text(scene, "location_type")
    other_location = _metadata_text(other, "location_type")
    activity = _metadata_text(scene, "activity")
    other_activity = _metadata_text(other, "activity")
    shot_type = _metadata_text(scene, "shot_type")
    other_shot_type = _metadata_text(other, "shot_type")
    shot_scale = _metadata_text(scene, "shot_scale")
    other_shot_scale = _metadata_text(other, "shot_scale")
    camera_motion = _metadata_text(scene, "camera_motion")
    other_camera_motion = _metadata_text(other, "camera_motion")
    movement_direction = _metadata_text(scene, "movement_direction")
    other_movement_direction = _metadata_text(other, "movement_direction")
    lighting = _metadata_text(scene, "lighting")
    other_lighting = _metadata_text(other, "lighting")
    if location and location == other_location:
        penalty += 7 * multiplier
    if activity and activity == other_activity:
        penalty += 6 * multiplier
    if shot_type and shot_type == other_shot_type:
        penalty += 6 * multiplier
    if shot_scale and shot_scale == other_shot_scale:
        penalty += 5 * multiplier
    if camera_motion and camera_motion == other_camera_motion:
        penalty += 4 * multiplier
    if movement_direction and movement_direction == other_movement_direction:
        penalty += 4 * multiplier
    if lighting and lighting == other_lighting:
        penalty += 3 * multiplier
    brightness = _quality_metric(scene, "brightness")
    other_brightness = _quality_metric(other, "brightness")
    if brightness is not None and other_brightness is not None:
        if abs(brightness - other_brightness) < 8:
            penalty += 3 * multiplier
        elif abs(brightness - other_brightness) > 42:
            penalty += 5 * multiplier
    shared_tags = _metadata_tags(scene) & _metadata_tags(other)
    if shared_tags:
        penalty += min(8, len(shared_tags) * 2) * multiplier
    return penalty


def _metadata_text(scene: Scene, key: str) -> str:
    value = scene.metadata.get(key)
    return str(value).casefold().strip() if value else ""


def _metadata_tags(scene: Scene) -> set[str]:
    tags = scene.metadata.get("tags", [])
    if not isinstance(tags, list):
        return set()
    return {str(tag).casefold().strip() for tag in tags if str(tag).strip()}


def _quality_metric(scene: Scene, key: str) -> float | None:
    metrics = scene.metadata.get("quality_metrics", {})
    if not isinstance(metrics, dict):
        return None
    value = metrics.get(key)
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _diversity_signature(scene: Scene) -> dict[str, object]:
    return {
        "asset_id": str(scene.asset_id),
        "location_type": _metadata_text(scene, "location_type"),
        "activity": _metadata_text(scene, "activity"),
        "shot_type": _metadata_text(scene, "shot_type"),
        "shot_scale": _metadata_text(scene, "shot_scale"),
        "camera_motion": _metadata_text(scene, "camera_motion"),
        "movement_direction": _metadata_text(scene, "movement_direction"),
        "lighting": _metadata_text(scene, "lighting"),
        "tags": sorted(_metadata_tags(scene)),
    }
