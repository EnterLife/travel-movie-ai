from datetime import UTC, datetime, timedelta
from pathlib import Path

from travelmovieai.domain.enums import MediaType, StoryStyle
from travelmovieai.domain.models import MediaAsset, Scene
from travelmovieai.story.builder import build_multimodal_descriptions, build_storyboard
from travelmovieai.story.events import detect_events


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
        _asset(f"scene-{index}.mp4", datetime(2026, 1, index + 1, tzinfo=UTC))
        for index in range(3)
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


def _asset(name: str, created_at: datetime) -> MediaAsset:
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
