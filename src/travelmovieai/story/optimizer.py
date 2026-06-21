"""Story-aware timeline metadata, budgets, and ordering helpers."""

from collections.abc import Iterable
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
    forced = [scene for scene in scenes if scene.metadata.get("selection_override") == "include"]
    forced_ids = {scene.id for scene in forced}
    available = [scene for scene in scenes if scene.id not in forced_ids]
    selected: list[Scene] = []
    selected_ids: set[UUID] = set()
    used_by_role = {role: 0.0 for role in STORY_ROLES}

    for scene in sorted(forced, key=_story_order_key):
        selected.append(scene)
        selected_ids.add(scene.id)
        role = _story_role(scene)
        used_by_role[role] += _estimated_scene_duration(scene, assets_by_id, settings)

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
        )
        for scene in role_selected:
            selected.append(scene)
            selected_ids.add(scene.id)
            used_by_role[role] += _estimated_scene_duration(scene, assets_by_id, settings)

    selected_duration = _estimated_timeline_duration(selected, assets_by_id, settings)
    remaining_pool = [scene for scene in scenes if scene.id not in selected_ids]
    while selected_duration < settings.target_duration_seconds - 0.05 and remaining_pool:
        scene = _best_diverse_scene(remaining_pool, selected, settings)
        selected.append(scene)
        selected_ids.add(scene.id)
        remaining_pool = [item for item in remaining_pool if item.id != scene.id]
        selected_duration = _estimated_timeline_duration(selected, assets_by_id, settings)

    selected_order = {scene.id: index for index, scene in enumerate(selected)}
    ordered = sorted(
        selected,
        key=lambda scene: (
            ROLE_ORDER[_story_role(scene)],
            selected_order[scene.id],
        ),
    )
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
) -> list[Scene]:
    selected: list[Scene] = []
    used = 0.0
    remaining = list(pool)
    while remaining:
        scene = _best_diverse_scene(remaining, [*selected_so_far, *selected], settings)
        duration = _estimated_scene_duration(scene, assets_by_id, settings)
        would_exceed = used + duration > budget_seconds
        if selected and would_exceed:
            break
        selected.append(scene)
        used += duration
        remaining = [item for item in remaining if item.id != scene.id]
        if used >= budget_seconds:
            break
    return selected


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
    transition = 0.0 if settings.transition == "none" else settings.transition_duration_seconds
    durations = [_estimated_scene_duration(scene, assets_by_id, settings) for scene in scenes]
    if durations:
        transition = min(transition, min(durations) * 0.45)
    return max(0.0, sum(durations) - max(0, len(durations) - 1) * transition)


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
