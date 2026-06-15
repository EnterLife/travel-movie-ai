from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageDraw

from travelmovieai.analysis.duplicates import detect_duplicate_scenes
from travelmovieai.domain.enums import MediaType
from travelmovieai.domain.models import MediaAsset, QuickMontageSettings, Scene
from travelmovieai.editing.timeline import (
    build_selection_report,
    build_semantic_montage_plan,
)


def test_duplicate_detection_keeps_stronger_scene(tmp_path: Path) -> None:
    first_image = tmp_path / "first.jpg"
    duplicate_image = tmp_path / "duplicate.jpg"
    different_image = tmp_path / "different.jpg"
    _pattern(first_image, 25)
    _pattern(duplicate_image, 25)
    _pattern(different_image, 80)
    asset_id = uuid4()
    weaker = Scene(
        asset_id=asset_id,
        start_seconds=0,
        end_seconds=3,
        keyframe_path=first_image,
        quality_score=40,
        importance_score=55,
    )
    stronger = Scene(
        asset_id=asset_id,
        start_seconds=3,
        end_seconds=6,
        keyframe_path=duplicate_image,
        quality_score=85,
        importance_score=90,
    )
    unique = Scene(
        asset_id=asset_id,
        start_seconds=6,
        end_seconds=9,
        keyframe_path=different_image,
        quality_score=70,
        importance_score=70,
    )

    report, scenes = detect_duplicate_scenes([weaker, stronger, unique])
    by_id = {scene.id: scene for scene in scenes}

    assert report.duplicate_count == 1
    assert report.groups[0].keeper_scene_id == stronger.id
    assert by_id[weaker.id].metadata["duplicate_of"] == str(stronger.id)
    assert by_id[unique.id].metadata["duplicate_status"] == "unique"


def test_story_selection_honors_overrides_and_event_diversity(tmp_path: Path) -> None:
    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    assets = [
        _asset(tmp_path / f"clip-{index}.mp4", created_at)
        for index in range(4)
    ]
    first_event = uuid4()
    second_event = uuid4()
    forced = _scene(
        assets[0],
        first_event,
        45,
        selection_override="include",
        duplicate_status="duplicate",
    )
    excluded = _scene(
        assets[1],
        first_event,
        95,
        selection_override="exclude",
    )
    duplicate = _scene(
        assets[2],
        first_event,
        90,
        duplicate_status="duplicate",
    )
    second_event_scene = _scene(assets[3], second_event, 70)
    settings = QuickMontageSettings(
        semantic_analysis=True,
        target_duration_seconds=8,
        max_video_clip_seconds=3,
        transition="none",
    )

    plan = build_semantic_montage_plan(
        assets,
        [forced, excluded, duplicate, second_event_scene],
        settings,
    )
    report = build_selection_report(
        [forced, excluded, duplicate, second_event_scene],
        plan,
        settings,
    )
    selected_ids = {clip.scene_id for clip in plan.clips}
    decisions = {decision.scene_id: decision for decision in report.decisions}

    assert forced.id in selected_ids
    assert second_event_scene.id in selected_ids
    assert excluded.id not in selected_ids
    assert duplicate.id not in selected_ids
    assert decisions[forced.id].reason == "required by user"
    assert decisions[duplicate.id].reason.startswith("near duplicate")


def test_semantic_selection_skips_weak_scenes_even_when_duration_remains(
    tmp_path: Path,
) -> None:
    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    strong_asset = _asset(tmp_path / "strong.mp4", created_at)
    weak_asset = _asset(tmp_path / "weak.mp4", created_at)
    strong = _scene(strong_asset, uuid4(), 90)
    weak = _scene(weak_asset, uuid4(), 22)
    settings = QuickMontageSettings(
        semantic_analysis=True,
        target_duration_seconds=8,
        max_video_clip_seconds=3,
        transition="none",
    )

    plan = build_semantic_montage_plan([strong_asset, weak_asset], [strong, weak], settings)
    report = build_selection_report([strong, weak], plan, settings)
    selected_ids = {clip.scene_id for clip in plan.clips}
    decisions = {decision.scene_id: decision for decision in report.decisions}

    assert strong.id in selected_ids
    assert weak.id not in selected_ids
    assert decisions[weak.id].reason.startswith("semantic score below")


def test_semantic_selection_limits_scenes_from_one_source_video(tmp_path: Path) -> None:
    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    source = _asset(tmp_path / "long-roll.mp4", created_at, duration=30)
    other = _asset(tmp_path / "other.mp4", created_at)
    scenes = [
        _scene(source, uuid4(), 95, start=index * 4)
        for index in range(4)
    ]
    other_scene = _scene(other, uuid4(), 78)
    settings = QuickMontageSettings(
        semantic_analysis=True,
        target_duration_seconds=15,
        max_video_clip_seconds=3,
        max_scenes_per_source=2,
        transition="none",
    )

    plan = build_semantic_montage_plan([source, other], [*scenes, other_scene], settings)

    source_clips = [clip for clip in plan.clips if clip.asset_id == source.id]
    assert len(source_clips) == 2
    assert other_scene.id in {clip.scene_id for clip in plan.clips}


def _pattern(path: Path, offset: int) -> None:
    image = Image.new("RGB", (180, 90), "black")
    draw = ImageDraw.Draw(image)
    draw.rectangle((offset, 15, offset + 35, 75), fill="white")
    draw.line((0, offset, 179, 89 - offset // 2), fill="gray", width=4)
    image.save(path)


def _asset(path: Path, created_at: datetime, duration: float = 4) -> MediaAsset:
    return MediaAsset(
        path=path,
        relative_path=Path(path.name),
        media_type=MediaType.VIDEO,
        extension=".mp4",
        size_bytes=1,
        modified_at=created_at,
        modified_ns=1,
        created_at=created_at,
        duration_seconds=duration,
    )


def _scene(
    asset: MediaAsset,
    event_id: object,
    score: float,
    start: float = 0,
    **metadata: object,
) -> Scene:
    return Scene(
        asset_id=asset.id,
        start_seconds=start,
        end_seconds=start + 3,
        quality_score=75,
        importance_score=score,
        caption=asset.relative_path.stem,
        metadata={
            "event_id": str(event_id),
            "event_importance": score,
            "location_type": "city",
            "activity": "walking",
            "emotion": "joyful",
            **metadata,
        },
    )
