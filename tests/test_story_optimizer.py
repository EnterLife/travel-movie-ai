from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

from travelmovieai.domain.enums import MediaType
from travelmovieai.domain.models import MediaAsset, QuickMontageSettings, Scene
from travelmovieai.story.optimizer import optimize_story_timeline_candidates


def test_story_optimizer_enforces_automatic_event_cap() -> None:
    assets = [_asset(index) for index in range(6)]
    event_id = uuid4()
    scenes = [
        _scene(asset, event_id, role="journey", score=90 - index)
        for index, asset in enumerate(assets)
    ]
    settings = QuickMontageSettings(
        target_duration_seconds=30,
        max_video_clip_seconds=4,
        max_scenes_per_event=2,
        strict_source_diversity=False,
    )

    selected = optimize_story_timeline_candidates(
        scenes,
        {asset.id: asset for asset in assets},
        settings,
    )

    assert len(selected) == 2
    assert all(
        scene.metadata["story_selection_caps"]["max_scenes_per_event"] == 2 for scene in selected
    )


def test_story_optimizer_preserves_explicit_includes_over_caps() -> None:
    assets = [_asset(index) for index in range(4)]
    event_id = uuid4()
    scenes = [
        _scene(asset, event_id, role="journey", score=90 - index).model_copy(
            update={
                "metadata": {
                    **_scene(asset, event_id, role="journey", score=90 - index).metadata,
                    "selection_override": "include" if index < 3 else "auto",
                }
            }
        )
        for index, asset in enumerate(assets)
    ]
    settings = QuickMontageSettings(
        target_duration_seconds=20,
        max_scenes_per_event=2,
        strict_source_diversity=False,
    )

    selected = optimize_story_timeline_candidates(
        scenes,
        {asset.id: asset for asset in assets},
        settings,
    )

    assert len(selected) == 3
    assert all(scene.metadata["story_cap_override"] is True for scene in selected)


def test_story_optimizer_keeps_role_backfill_inside_budgets() -> None:
    roles = ("opening", "journey", "highlight", "finale")
    assets = [_asset(index) for index in range(24)]
    scenes = [
        _scene(asset, uuid4(), role=roles[index % len(roles)], score=95 - index)
        for index, asset in enumerate(assets)
    ]
    settings = QuickMontageSettings(
        target_duration_seconds=40,
        max_video_clip_seconds=4,
        strict_source_diversity=False,
    )

    selected = optimize_story_timeline_candidates(
        scenes,
        {asset.id: asset for asset in assets},
        settings,
    )

    counts = {
        role: sum(scene.metadata["story_section_role"] == role for scene in selected)
        for role in roles
    }
    assert counts == {"opening": 2, "journey": 3, "highlight": 3, "finale": 2}
    assert all(
        scene.metadata["story_section_used_seconds"]
        <= scene.metadata["story_section_budget_seconds"] + settings.max_video_clip_seconds
        for scene in selected
    )


def test_story_optimizer_keeps_strict_source_cap() -> None:
    asset = _asset(0)
    scenes = [
        _scene(asset, uuid4(), role="journey", score=95 - index).model_copy(
            update={"start_seconds": index * 4, "end_seconds": index * 4 + 4}
        )
        for index in range(4)
    ]
    settings = QuickMontageSettings(
        target_duration_seconds=16,
        max_scenes_per_source=1,
        strict_source_diversity=True,
    )

    selected = optimize_story_timeline_candidates(scenes, {asset.id: asset}, settings)

    assert len(selected) == 1


def test_story_optimizer_preserves_chronology_inside_story_role() -> None:
    assets = [_asset(index) for index in range(3)]
    event_ids = [uuid4() for _ in assets]
    scenes = [
        _scene(asset, event_id, role="journey", score=70 + (2 - index) * 10)
        for index, (asset, event_id) in enumerate(zip(assets, event_ids, strict=True))
    ]
    settings = QuickMontageSettings(
        target_duration_seconds=12,
        max_video_clip_seconds=4,
        strict_source_diversity=False,
        preserve_chronology=True,
    )

    selected = optimize_story_timeline_candidates(
        scenes,
        {asset.id: asset for asset in assets},
        settings,
    )

    assert [scene.asset_id for scene in selected] == [asset.id for asset in assets]


def test_story_optimizer_diversifies_sources_inside_chronological_event() -> None:
    first, second, finale = (_asset(index) for index in range(3))
    event_id = uuid4()
    scenes = [
        _scene(first, event_id, role="journey", score=92).model_copy(
            update={"start_seconds": 0, "end_seconds": 4}
        ),
        _scene(first, event_id, role="journey", score=91).model_copy(
            update={"start_seconds": 4, "end_seconds": 8}
        ),
        _scene(second, event_id, role="journey", score=90).model_copy(
            update={"start_seconds": 0, "end_seconds": 4}
        ),
        _scene(second, event_id, role="journey", score=89).model_copy(
            update={"start_seconds": 4, "end_seconds": 8}
        ),
        _scene(finale, uuid4(), role="journey", score=88),
    ]
    settings = QuickMontageSettings(
        target_duration_seconds=20,
        max_video_clip_seconds=4,
        max_scenes_per_event=4,
        max_scenes_per_source=2,
        preserve_chronology=True,
    )

    selected = optimize_story_timeline_candidates(
        scenes,
        {asset.id: asset for asset in (first, second, finale)},
        settings,
    )

    assert [scene.asset_id for scene in selected] == [
        first.id,
        second.id,
        first.id,
        second.id,
        finale.id,
    ]
    assert [scene.start_seconds for scene in selected if scene.asset_id == first.id] == [0, 4]
    assert [scene.start_seconds for scene in selected if scene.asset_id == second.id] == [0, 4]


def _asset(index: int) -> MediaAsset:
    created_at = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=index)
    return MediaAsset(
        path=Path(f"source-{index}.mp4"),
        relative_path=Path(f"source-{index}.mp4"),
        media_type=MediaType.VIDEO,
        extension=".mp4",
        size_bytes=1,
        modified_at=created_at,
        modified_ns=index + 1,
        created_at=created_at,
        duration_seconds=60,
    )


def _scene(
    asset: MediaAsset,
    event_id: UUID,
    *,
    role: str,
    score: float,
) -> Scene:
    return Scene(
        asset_id=asset.id,
        start_seconds=0,
        end_seconds=4,
        importance_score=score,
        quality_score=80,
        metadata={
            "event_id": str(event_id),
            "ranking_score": score,
            "story_section_role": role,
        },
    )
