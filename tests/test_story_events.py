from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from travelmovieai.domain.enums import MediaType, StoryStyle
from travelmovieai.domain.models import Event, MediaAsset, Scene
from travelmovieai.story.builder import build_multimodal_descriptions, build_storyboard
from travelmovieai.story.events import detect_events
from travelmovieai.story.narration import build_narration


def test_event_detection_groups_semantically_related_scenes() -> None:
    first_asset = _asset("airport.mp4", datetime(2026, 1, 1, 10, tzinfo=UTC))
    second_asset = _asset(
        "taxi.mp4",
        datetime(2026, 1, 1, 10, 20, tzinfo=UTC),
    )
    scenes = [
        _scene(first_asset, "airport", "arriving", "Leaving the airport."),
        _scene(second_asset, "transport", "arriving", "Taking a taxi."),
    ]

    report, updated = detect_events(scenes, [first_asset, second_asset])

    assert len(report.events) == 1
    assert report.events[0].title == "Arrival Day"
    assert all(scene.metadata.get("event_id") for scene in updated)


def test_event_detection_splits_large_time_gap() -> None:
    morning = _asset("morning.mp4", datetime(2026, 1, 1, 8, tzinfo=UTC))
    evening = _asset("evening.mp4", datetime(2026, 1, 1, 18, tzinfo=UTC))

    report, _ = detect_events(
        [
            _scene(morning, "city", "walking", "Morning walk."),
            _scene(evening, "city", "walking", "Evening walk."),
        ],
        [morning, evening],
    )

    assert len(report.events) == 2


def test_event_detection_splits_different_places_with_generic_activity() -> None:
    beach = _asset("beach.mp4", datetime(2026, 1, 1, 10, tzinfo=UTC))
    city = _asset("city.mp4", datetime(2026, 1, 1, 10, 20, tzinfo=UTC))

    report, _ = detect_events(
        [
            _scene(beach, "beach", "walking", "Walking on the beach."),
            _scene(city, "city", "walking", "Walking downtown."),
        ],
        [beach, city],
    )

    assert len(report.events) == 2
    assert [event.title for event in report.events] == ["Beach Day", "City Exploration"]


def test_event_detection_uses_nearby_gps_to_join_semantically_different_scenes() -> None:
    first = _asset(
        "park.mp4",
        datetime(2026, 1, 1, 10, tzinfo=UTC),
        latitude=43.5855,
        longitude=39.7231,
    )
    second = _asset(
        "museum.mp4",
        datetime(2026, 1, 1, 10, 20, tzinfo=UTC),
        latitude=43.5880,
        longitude=39.7200,
    )

    report, _ = detect_events(
        [
            _scene(first, "park", "walking", "A quiet park."),
            _scene(second, "museum", "sightseeing", "An indoor museum."),
        ],
        [first, second],
    )

    assert len(report.events) == 1


def test_event_detection_uses_distant_gps_to_split_matching_scene_labels() -> None:
    first = _asset(
        "city-a.mp4",
        datetime(2026, 1, 1, 10, tzinfo=UTC),
        latitude=43.5855,
        longitude=39.7231,
    )
    second = _asset(
        "city-b.mp4",
        datetime(2026, 1, 1, 10, 20, tzinfo=UTC),
        latitude=44.6167,
        longitude=33.5254,
    )

    report, _ = detect_events(
        [
            _scene(first, "city", "walking", "A city street."),
            _scene(second, "city", "walking", "Another city street."),
        ],
        [first, second],
    )

    assert len(report.events) == 2


def test_event_detection_uses_semantic_embeddings_without_gps() -> None:
    first = _asset("coast.mp4", datetime(2026, 1, 1, 10, tzinfo=UTC))
    second = _asset("boat.mp4", datetime(2026, 1, 1, 10, 20, tzinfo=UTC))
    scenes = [
        _scene(first, "beach", "walking", "A coast at sunrise."),
        _scene(second, "sea", "boating", "A boat near the coast."),
    ]
    scenes = [
        scene.model_copy(
            update={"metadata": {**scene.metadata, "semantic_embedding": [1.0, 0.0, 1.0]}}
        )
        for scene in scenes
    ]

    report, _ = detect_events(scenes, [first, second])

    assert len(report.events) == 1


def test_event_detection_is_deterministic_without_gps_or_embeddings() -> None:
    first = _asset("first.mp4", datetime(2026, 1, 1, 10, tzinfo=UTC))
    second = _asset("second.mp4", datetime(2026, 1, 1, 10, 20, tzinfo=UTC))
    scenes = [
        _scene(first, "city", "walking", "First walk."),
        _scene(second, "city", "walking", "Second walk."),
    ]

    first_report, _ = detect_events(scenes, [first, second])
    second_report, _ = detect_events(scenes, [first, second])

    assert [event.id for event in first_report.events] == [
        event.id for event in second_report.events
    ]


def test_event_detection_caps_long_same_asset_sequence() -> None:
    asset = _asset("long-roll.mp4", datetime(2026, 1, 1, 10, tzinfo=UTC))
    scenes = [
        _scene(asset, "city", "walking", f"Street moment {index}.").model_copy(
            update={"start_seconds": index * 5, "end_seconds": index * 5 + 4}
        )
        for index in range(10)
    ]

    report, _ = detect_events(scenes, [asset])

    assert [len(event.scene_ids) for event in report.events] == [8, 2]


def test_event_detection_splits_conflicting_locations_inside_one_asset() -> None:
    asset = _asset("mixed-roll.mp4", datetime(2026, 1, 1, 10, tzinfo=UTC))
    first = _scene(asset, "beach", "walking", "Beach walk.")
    second = _scene(asset, "museum", "sightseeing", "Museum hall.").model_copy(
        update={"start_seconds": 5, "end_seconds": 10}
    )

    report, _ = detect_events([first, second], [asset])

    assert len(report.events) == 2


def test_event_title_requires_corroborated_non_generic_landmark() -> None:
    asset = _asset("landmarks.mp4", datetime(2026, 1, 1, 10, tzinfo=UTC))
    scenes = []
    for index, landmark in enumerate(["town", "Sochi Olympic Park", "Sochi Olympic Park"]):
        scene = _scene(asset, "landmark", "sightseeing", f"View {index}.")
        scenes.append(
            scene.model_copy(
                update={
                    "start_seconds": index * 4,
                    "end_seconds": index * 4 + 3,
                    "metadata": {
                        **scene.metadata,
                        "landmarks": [{"name": landmark, "confidence": 0.9}],
                    },
                }
            )
        )

    report, _ = detect_events(scenes, [asset])

    assert len(report.events) == 1
    assert report.events[0].title == "Sochi Olympic Park"
    assert report.events[0].landmarks == ["Sochi Olympic Park"]


def test_event_regrouping_removes_stale_manual_event_metadata() -> None:
    asset = _asset("new-group.mp4", datetime(2026, 1, 1, 10, tzinfo=UTC))
    scene = _scene(asset, "beach", "walking", "Fresh grouping.").model_copy(
        update={
            "metadata": {
                **_scene(asset, "beach", "walking", "Fresh grouping.").metadata,
                "event_id": "old-event",
                "event_title": "Old manual title",
                "event_summary": "Old manual summary",
                "event_landmarks": ["Old landmark"],
                "manual_event_order": 7,
            }
        }
    )

    report, updated = detect_events([scene], [asset])

    assert updated[0].metadata["event_id"] == str(report.events[0].id)
    assert updated[0].metadata["event_title"] == report.events[0].title
    assert "event_summary" not in updated[0].metadata
    assert "event_landmarks" not in updated[0].metadata
    assert "manual_event_order" not in updated[0].metadata


def test_media_asset_rejects_invalid_gps_coordinates() -> None:
    with pytest.raises(ValueError, match="latitude"):
        _asset(
            "invalid.mp4",
            datetime(2026, 1, 1, tzinfo=UTC),
            latitude=91,
            longitude=0,
        )


def test_multimodal_description_records_used_sources() -> None:
    asset = _asset("scene.mp4", datetime.now(UTC))
    scene = _scene(asset, "museum", "sightseeing", "A museum hall.").model_copy(
        update={
            "transcript": "This is the main gallery.",
            "quality_score": 78,
            "metadata": {
                "detailed_description": "Visitors explore a bright museum hall.",
                "location_type": "museum",
                "activity": "sightseeing",
                "audio_context": ["quiet ambience"],
            },
        }
    )

    report = build_multimodal_descriptions([scene])

    assert report.descriptions[0].source_modalities == [
        "vision",
        "speech",
        "opencv",
        "audio",
    ]
    assert "main gallery" in report.descriptions[0].description


def test_storyboard_builds_opening_highlight_and_finale() -> None:
    assets = [
        _asset(f"scene-{index}.mp4", datetime(2026, 1, index + 1, tzinfo=UTC)) for index in range(3)
    ]
    event_report, scenes = detect_events(
        [
            _scene(assets[0], "airport", "arriving", "Arrival."),
            _scene(assets[1], "city", "sightseeing", "Main city highlight."),
            _scene(assets[2], "hotel", "departing", "Last evening."),
        ],
        assets,
    )

    storyboard = build_storyboard(
        event_report.events,
        scenes,
        StoryStyle.CINEMATIC,
    )

    assert [section.role for section in storyboard.sections] == [
        "opening",
        "highlight",
        "finale",
    ]
    assert storyboard.event_ids == [event.id for event in event_report.events]

    narration = build_narration(
        storyboard,
        event_report.events,
        target_duration_seconds=60,
    )

    assert [line.section_role for line in narration.lines] == [
        "opening",
        "highlight",
        "finale",
    ]
    assert narration.lines[0].text.startswith("Our journey begins")
    assert [line.cue_start_seconds for line in narration.lines] == [1, 21, 41]
    assert [line.cue_end_seconds for line in narration.lines] == [19, 39, 59]


def test_storyboard_distributes_large_event_set_across_story_arc() -> None:
    events = [
        Event(title=f"Event {index}", importance_score=float(index * 10)) for index in range(10)
    ]
    scenes: list[Scene] = []
    for index, event in enumerate(events):
        asset = _asset(
            f"event-{index}.mp4",
            datetime(2026, 1, index + 1, tzinfo=UTC),
        )
        scene = _scene(asset, "city", "sightseeing", f"Event {index}.")
        scenes.append(
            scene.model_copy(update={"metadata": {**scene.metadata, "event_id": str(event.id)}})
        )

    storyboard = build_storyboard(events, scenes, StoryStyle.CINEMATIC)

    sections = {section.role: section for section in storyboard.sections}
    assert {role: len(section.event_ids) for role, section in sections.items()} == {
        "opening": 2,
        "journey": 3,
        "highlight": 3,
        "finale": 2,
    }
    assert sections["highlight"].event_ids == [events[5].id, events[6].id, events[7].id]
    assert {event_id for section in storyboard.sections for event_id in section.event_ids} == {
        event.id for event in events
    }
    assert sum(len(section.scene_ids) for section in storyboard.sections) == len(scenes)


def test_narration_text_respects_available_speech_budget() -> None:
    asset = _asset("arrival.mp4", datetime(2026, 1, 1, tzinfo=UTC))
    event_report, scenes = detect_events(
        [
            _scene(
                asset,
                "airport",
                "arriving",
                "A deliberately detailed arrival summary with more words than the cue can fit.",
            )
        ],
        [asset],
    )
    storyboard = build_storyboard(event_report.events, scenes, StoryStyle.CINEMATIC)

    narration = build_narration(
        storyboard,
        event_report.events,
        target_duration_seconds=5,
        characters_per_second=8,
    )

    assert narration.lines
    for line in narration.lines:
        available_characters = max(
            1,
            int((line.cue_end_seconds - line.cue_start_seconds) * 8),
        )
        assert len(line.text) <= available_characters
    assert any(line.text.endswith("...") for line in narration.lines)


def _asset(
    name: str,
    created_at: datetime,
    *,
    latitude: float | None = None,
    longitude: float | None = None,
) -> MediaAsset:
    return MediaAsset(
        path=Path(name),
        relative_path=Path(name),
        media_type=MediaType.VIDEO,
        extension=".mp4",
        size_bytes=1,
        modified_at=created_at + timedelta(seconds=1),
        modified_ns=1,
        created_at=created_at,
        duration_seconds=10,
        latitude=latitude,
        longitude=longitude,
    )


def _scene(
    asset: MediaAsset,
    location: str,
    activity: str,
    caption: str,
) -> Scene:
    return Scene(
        asset_id=asset.id,
        start_seconds=0,
        end_seconds=5,
        caption=caption,
        importance_score=80,
        metadata={
            "location_type": location,
            "activity": activity,
            "emotion": "cinematic",
            "landmarks": [],
        },
    )
