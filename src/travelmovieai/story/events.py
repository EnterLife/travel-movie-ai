"""Deterministic scene-to-event clustering based on Vision AI metadata."""

import math
from datetime import UTC, datetime, timedelta
from uuid import NAMESPACE_URL, UUID, uuid5

from travelmovieai.analysis.embeddings import semantic_similarity
from travelmovieai.domain.enums import ActivityType, LocationType
from travelmovieai.domain.models import (
    Event,
    EventDetectionReport,
    MediaAsset,
    Scene,
)
from travelmovieai.story.editorial import clean_caption, clean_title

MAX_EVENT_GAP = timedelta(minutes=90)
MAX_EVENT_DURATION = timedelta(minutes=30)
MAX_EVENT_SCENES = 8
MAX_EVENT_DISTANCE_KM = 5.0
MIN_EVENT_SEMANTIC_SIMILARITY = 0.72
MIN_SAME_ASSET_SEMANTIC_SIMILARITY = 0.42
_EVENT_ID_NAMESPACE = uuid5(NAMESPACE_URL, "https://travelmovieai.local/event/v1")


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
        if not groups or not _can_append_to_event(groups[-1], scene, assets_by_id):
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
                    **_without_stale_event_overrides(scene.metadata),
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


def _without_stale_event_overrides(metadata: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in metadata.items()
        if key not in {"manual_event_order", "event_summary", "event_landmarks"}
    }


def _can_append_to_event(
    group: list[Scene],
    current: Scene,
    assets: dict[UUID, MediaAsset],
) -> bool:
    if len(group) >= MAX_EVENT_SCENES:
        return False
    if _scene_end_time(current, assets) - _scene_time(group[0], assets) > MAX_EVENT_DURATION:
        return False

    previous = group[-1]
    previous_time = _scene_time(previous, assets)
    current_time = _scene_time(current, assets)
    if current_time - previous_time > MAX_EVENT_GAP:
        return False

    previous_location = _metadata_value(previous, "location_type")
    current_location = _metadata_value(current, "location_type")
    meaningful_locations = _meaningful_location(previous_location) and _meaningful_location(
        current_location
    )
    similarity = semantic_similarity(previous, current)

    if previous.asset_id == current.asset_id:
        if meaningful_locations and previous_location != current_location:
            return similarity is not None and similarity >= MIN_EVENT_SEMANTIC_SIMILARITY
        return similarity is None or similarity >= MIN_SAME_ASSET_SEMANTIC_SIMILARITY

    gps_distance = _gps_distance_km(assets[previous.asset_id], assets[current.asset_id])
    if gps_distance is not None:
        return gps_distance <= MAX_EVENT_DISTANCE_KM

    previous_landmarks = _landmark_names(previous)
    current_landmarks = _landmark_names(current)
    if previous_landmarks & current_landmarks:
        return True

    if similarity is not None and similarity >= MIN_EVENT_SEMANTIC_SIMILARITY:
        return True

    location_matches = previous_location == current_location
    previous_activity = _metadata_value(previous, "activity")
    current_activity = _metadata_value(current, "activity")
    activity_matches = previous_activity == current_activity and previous_activity not in {
        "",
        ActivityType.UNKNOWN.value,
        ActivityType.OTHER.value,
    }
    if meaningful_locations and location_matches and activity_matches:
        return True

    return previous_activity == current_activity and previous_activity in {
        ActivityType.ARRIVING.value,
        ActivityType.DEPARTING.value,
        ActivityType.TRAVELING.value,
    }


def _meaningful_location(value: str) -> bool:
    return value not in {
        "",
        LocationType.UNKNOWN.value,
        LocationType.OTHER.value,
    }


def _gps_distance_km(first: MediaAsset, second: MediaAsset) -> float | None:
    if not _valid_coordinates(first.latitude, first.longitude) or not _valid_coordinates(
        second.latitude,
        second.longitude,
    ):
        return None
    assert first.latitude is not None
    assert first.longitude is not None
    assert second.latitude is not None
    assert second.longitude is not None
    latitude_delta = math.radians(second.latitude - first.latitude)
    longitude_delta = math.radians(second.longitude - first.longitude)
    first_latitude = math.radians(first.latitude)
    second_latitude = math.radians(second.latitude)
    haversine = (
        math.sin(latitude_delta / 2) ** 2
        + math.cos(first_latitude) * math.cos(second_latitude) * math.sin(longitude_delta / 2) ** 2
    )
    return 6371.0088 * 2 * math.asin(min(1.0, math.sqrt(haversine)))


def _valid_coordinates(latitude: float | None, longitude: float | None) -> bool:
    return (
        latitude is not None
        and longitude is not None
        and math.isfinite(latitude)
        and math.isfinite(longitude)
        and -90 <= latitude <= 90
        and -180 <= longitude <= 180
    )


def _build_event(
    scenes: list[Scene],
    assets: dict[UUID, MediaAsset],
) -> Event:
    locations = [_metadata_value(scene, "location_type") for scene in scenes]
    activities = [_metadata_value(scene, "activity") for scene in scenes]
    location = _most_common_location(locations)
    activity = _most_common_activity(activities)
    landmarks = _corroborated_landmarks(scenes)
    start_at = min(_scene_time(scene, assets) for scene in scenes)
    end_at = max(_scene_end_time(scene, assets) for scene in scenes)
    scores = [scene.importance_score for scene in scenes if scene.importance_score is not None]
    importance = sum(scores) / len(scores) if scores else 50.0
    semantic_fields = int(location != LocationType.UNKNOWN)
    semantic_fields += int(activity != ActivityType.UNKNOWN)
    semantic_fields += int(bool(landmarks))
    confidence = min(1.0, 0.45 + semantic_fields * 0.15 + min(len(scenes), 4) * 0.025)
    captions = list(
        dict.fromkeys(
            caption for scene in scenes if (caption := clean_caption(scene.caption)) is not None
        )
    )
    return Event(
        id=_event_id(scenes),
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


def _event_id(scenes: list[Scene]) -> UUID:
    scene_identity = ":".join(str(scene.id) for scene in scenes)
    return uuid5(_EVENT_ID_NAMESPACE, scene_identity)


def _event_title(
    location: LocationType,
    activity: ActivityType,
    landmarks: list[str],
) -> str:
    if landmarks:
        return clean_title(landmarks[0]) or "Travel Moment"
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
    if location in labels:
        return labels[location]
    if activity not in {ActivityType.UNKNOWN, ActivityType.OTHER}:
        return activity.value.replace("_", " ").title()
    return "Travel Moment"


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
        cleaned.casefold()
        for item in values
        if isinstance(item, dict)
        and _float_value(item.get("confidence")) >= 0.55
        and (cleaned := clean_title(item.get("name"))) is not None
    }


def _corroborated_landmarks(scenes: list[Scene]) -> list[str]:
    names_by_key: dict[str, str] = {}
    evidence_counts: dict[str, int] = {}
    trusted_keys: set[str] = set()
    for scene in scenes:
        event_landmarks = scene.metadata.get("event_landmarks", [])
        if isinstance(event_landmarks, list):
            for value in event_landmarks:
                name = clean_title(value)
                if name is not None:
                    key = name.casefold()
                    names_by_key.setdefault(key, name)
                    trusted_keys.add(key)
        values = scene.metadata.get("landmarks", [])
        if not isinstance(values, list):
            continue
        seen_in_scene: set[str] = set()
        for item in values:
            if not isinstance(item, dict) or _float_value(item.get("confidence")) < 0.55:
                continue
            name = clean_title(item.get("name"))
            if name is None:
                continue
            key = name.casefold()
            names_by_key.setdefault(key, name)
            seen_in_scene.add(key)
            if str(item.get("evidence", "")).strip().casefold() == "manual edit":
                trusted_keys.add(key)
        for key in seen_in_scene:
            evidence_counts[key] = evidence_counts.get(key, 0) + 1
    return sorted(
        (
            name
            for key, name in names_by_key.items()
            if key in trusted_keys or evidence_counts.get(key, 0) >= 2
        ),
        key=str.casefold,
    )


def _float_value(value: object) -> float:
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


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
