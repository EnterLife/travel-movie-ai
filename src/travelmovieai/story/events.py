"""Deterministic scene-to-event clustering based on Vision AI metadata."""

from datetime import UTC, datetime, timedelta
from uuid import UUID

from travelmovieai.domain.enums import ActivityType, LocationType
from travelmovieai.domain.models import (
    Event,
    EventDetectionReport,
    MediaAsset,
    Scene,
)

MAX_EVENT_GAP = timedelta(minutes=90)


def detect_events(
    scenes: list[Scene],
    assets: list[MediaAsset],
) -> tuple[EventDetectionReport, list[Scene]]:
    """Group chronological scenes using capture time and semantic continuity."""
    assets_by_id = {asset.id: asset for asset in assets}
    ordered = sorted(
        scenes,
        key=lambda scene: (
            _scene_time(scene, assets_by_id),
            scene.start_seconds,
            str(scene.id),
        ),
    )
    groups: list[list[Scene]] = []
    for scene in ordered:
        if not groups or not _same_event(groups[-1][-1], scene, assets_by_id):
            groups.append([scene])
        else:
            groups[-1].append(scene)

    events: list[Event] = []
    event_by_scene: dict[UUID, Event] = {}
    for group in groups:
        event = _build_event(group, assets_by_id)
        events.append(event)
        event_by_scene.update((scene.id, event) for scene in group)

    updated_scenes = [
        scene.model_copy(
            update={
                "metadata": {
                    **scene.metadata,
                    "event_id": str(event_by_scene[scene.id].id),
                    "event_title": event_by_scene[scene.id].title,
                    "event_importance": event_by_scene[scene.id].importance_score,
                }
            }
        )
        for scene in scenes
    ]
    return (
        EventDetectionReport(created_at=datetime.now(UTC), events=events),
        updated_scenes,
    )


def _same_event(
    previous: Scene,
    current: Scene,
    assets: dict[UUID, MediaAsset],
) -> bool:
    previous_time = _scene_time(previous, assets)
    current_time = _scene_time(current, assets)
    if current_time - previous_time > MAX_EVENT_GAP:
        return False
    if previous.asset_id == current.asset_id:
        return True

    previous_landmarks = _landmark_names(previous)
    current_landmarks = _landmark_names(current)
    if previous_landmarks & current_landmarks:
        return True

    location_matches = _metadata_value(previous, "location_type") == _metadata_value(
        current, "location_type"
    )
    activity_matches = _metadata_value(previous, "activity") == _metadata_value(
        current, "activity"
    )
    meaningful_location = _metadata_value(previous, "location_type") not in {
        "",
        LocationType.UNKNOWN.value,
        LocationType.OTHER.value,
    }
    return meaningful_location and (location_matches or activity_matches)


def _build_event(
    scenes: list[Scene],
    assets: dict[UUID, MediaAsset],
) -> Event:
    locations = [_metadata_value(scene, "location_type") for scene in scenes]
    activities = [_metadata_value(scene, "activity") for scene in scenes]
    location = _most_common_location(locations)
    activity = _most_common_activity(activities)
    landmarks = sorted(
        {name for scene in scenes for name in _landmark_names(scene)},
        key=str.casefold,
    )
    start_at = min(_scene_time(scene, assets) for scene in scenes)
    end_at = max(_scene_end_time(scene, assets) for scene in scenes)
    scores = [
        scene.importance_score
        for scene in scenes
        if scene.importance_score is not None
    ]
    importance = sum(scores) / len(scores) if scores else 50.0
    semantic_fields = int(location != LocationType.UNKNOWN)
    semantic_fields += int(activity != ActivityType.UNKNOWN)
    semantic_fields += int(bool(landmarks))
    confidence = min(1.0, 0.45 + semantic_fields * 0.15 + min(len(scenes), 4) * 0.025)
    captions = [scene.caption for scene in scenes if scene.caption]
    return Event(
        title=_event_title(location, activity, landmarks),
        scene_ids=[scene.id for scene in scenes],
        summary=" ".join(captions[:3]),
        importance_score=importance,
        start_at=start_at,
        end_at=end_at,
        location_type=location,
        activity=activity,
        landmarks=landmarks,
        confidence=confidence,
    )


def _event_title(
    location: LocationType,
    activity: ActivityType,
    landmarks: list[str],
) -> str:
    if landmarks:
        return landmarks[0]
    if location == LocationType.AIRPORT or activity == ActivityType.ARRIVING:
        return "Arrival Day"
    if activity == ActivityType.DEPARTING:
        return "Departure"
    labels = {
        LocationType.BEACH: "Beach Day",
        LocationType.SEA: "Day at Sea",
        LocationType.MOUNTAINS: "Mountain Adventure",
        LocationType.FOREST: "Forest Walk",
        LocationType.CITY: "City Exploration",
        LocationType.HOTEL: "Hotel Stay",
        LocationType.RESTAURANT: "Local Dining",
        LocationType.MUSEUM: "Museum Visit",
        LocationType.PARK: "Park Walk",
        LocationType.LANDMARK: "Landmark Visit",
        LocationType.TRANSPORT: "On the Road",
    }
    return labels.get(location, activity.value.replace("_", " ").title() or "Travel Event")


def _scene_time(scene: Scene, assets: dict[UUID, MediaAsset]) -> datetime:
    asset = assets[scene.asset_id]
    base = asset.created_at or asset.modified_at
    if base.tzinfo is None:
        base = base.replace(tzinfo=UTC)
    return base.astimezone(UTC) + timedelta(seconds=scene.start_seconds)


def _scene_end_time(scene: Scene, assets: dict[UUID, MediaAsset]) -> datetime:
    return _scene_time(scene, assets) + timedelta(
        seconds=max(0, scene.end_seconds - scene.start_seconds)
    )


def _metadata_value(scene: Scene, key: str) -> str:
    return str(scene.metadata.get(key, "")).strip().casefold()


def _landmark_names(scene: Scene) -> set[str]:
    values = scene.metadata.get("landmarks", [])
    return {
        str(item.get("name", "")).strip()
        for item in values
        if isinstance(item, dict)
        and float(item.get("confidence", 0)) >= 0.55
        and str(item.get("name", "")).strip()
    }


def _most_common_value(values: list[str]) -> str | None:
    filtered = [value for value in values if value and value != "unknown"]
    if not filtered:
        return None
    return max(set(filtered), key=lambda item: (filtered.count(item), item))


def _most_common_location(values: list[str]) -> LocationType:
    value = _most_common_value(values)
    if value is None:
        return LocationType.UNKNOWN
    try:
        return LocationType(value)
    except ValueError:
        return LocationType.UNKNOWN


def _most_common_activity(values: list[str]) -> ActivityType:
    value = _most_common_value(values)
    if value is None:
        return ActivityType.UNKNOWN
    try:
        return ActivityType(value)
    except ValueError:
        return ActivityType.UNKNOWN
