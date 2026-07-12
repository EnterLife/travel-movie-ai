from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageDraw

from travelmovieai.analysis.duplicates import detect_duplicate_scenes
from travelmovieai.domain.enums import MediaType
from travelmovieai.domain.models import (
    MediaAsset,
    MontageClip,
    MusicBeat,
    MusicPlan,
    QuickMontagePlan,
    QuickMontageSettings,
    Scene,
)
from travelmovieai.editing.timeline import (
    apply_music_directing,
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
    assets = [_asset(tmp_path / f"clip-{index}.mp4", created_at) for index in range(4)]
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


def test_semantic_threshold_relaxes_for_modest_but_consistent_material(
    tmp_path: Path,
) -> None:
    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    assets = [_asset(tmp_path / f"modest-{index}.mp4", created_at) for index in range(3)]
    scenes = [
        _scene(asset, uuid4(), score) for asset, score in zip(assets, [45, 43, 41], strict=True)
    ]
    settings = QuickMontageSettings(
        semantic_analysis=True,
        target_duration_seconds=6,
        max_video_clip_seconds=3,
        transition="none",
    )

    plan = build_semantic_montage_plan(assets, scenes, settings)

    assert len(plan.clips) == 2
    assert {clip.scene_id for clip in plan.clips}.issubset({scene.id for scene in scenes})


def test_semantic_threshold_rises_for_strong_archives(tmp_path: Path) -> None:
    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    assets = [_asset(tmp_path / f"scene-{index}.mp4", created_at) for index in range(4)]
    scenes = [
        _scene(asset, uuid4(), score) for asset, score in zip(assets, [95, 90, 86, 55], strict=True)
    ]
    settings = QuickMontageSettings(
        semantic_analysis=True,
        target_duration_seconds=6,
        max_video_clip_seconds=3,
        transition="none",
    )

    plan = build_semantic_montage_plan(assets, scenes, settings)
    report = build_selection_report(scenes, plan, settings)
    selected_ids = {clip.scene_id for clip in plan.clips}
    decisions = {decision.scene_id: decision for decision in report.decisions}

    assert scenes[-1].id not in selected_ids
    assert decisions[scenes[-1].id].reason.startswith("semantic score below adaptive")


def test_semantic_selection_limits_scenes_from_one_source_video(tmp_path: Path) -> None:
    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    source = _asset(tmp_path / "long-roll.mp4", created_at, duration=30)
    others = [_asset(tmp_path / f"other-{index}.mp4", created_at) for index in range(4)]
    scenes = [_scene(source, uuid4(), 95, start=index * 4) for index in range(4)]
    other_scenes = [_scene(asset, uuid4(), 78 - index) for index, asset in enumerate(others)]
    settings = QuickMontageSettings(
        semantic_analysis=True,
        target_duration_seconds=15,
        max_video_clip_seconds=3,
        max_scenes_per_source=2,
        transition="none",
    )

    plan = build_semantic_montage_plan([source, *others], [*scenes, *other_scenes], settings)

    source_clips = [clip for clip in plan.clips if clip.asset_id == source.id]
    assert len(source_clips) == 2
    assert {scene.id for scene in other_scenes} & {clip.scene_id for clip in plan.clips}


def test_semantic_selection_keeps_strict_source_limit_by_default(
    tmp_path: Path,
) -> None:
    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    dominant = _asset(tmp_path / "dominant-roll.mp4", created_at, duration=30)
    alternate = _asset(tmp_path / "alternate-roll.mp4", created_at, duration=30)
    dominant_scenes = [_scene(dominant, uuid4(), 99 - index, start=index * 4) for index in range(4)]
    alternate_scene = _scene(alternate, uuid4(), 94)
    settings = QuickMontageSettings(
        semantic_analysis=True,
        target_duration_seconds=12,
        max_video_clip_seconds=3,
        max_scenes_per_source=1,
        transition="none",
    )

    plan = build_semantic_montage_plan(
        [dominant, alternate],
        [*dominant_scenes, alternate_scene],
        settings,
    )

    source_counts: dict[object, int] = {}
    for clip in plan.clips:
        source_counts[clip.asset_id] = source_counts.get(clip.asset_id, 0) + 1
    assert source_counts[dominant.id] == 1
    assert source_counts[alternate.id] == 1


def test_semantic_selection_can_relax_source_limit_for_coverage(
    tmp_path: Path,
) -> None:
    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    dominant = _asset(tmp_path / "adaptive-roll.mp4", created_at, duration=30)
    alternate = _asset(tmp_path / "adaptive-alternate.mp4", created_at, duration=30)
    dominant_scenes = [_scene(dominant, uuid4(), 99 - index, start=index * 4) for index in range(4)]
    alternate_scene = _scene(alternate, uuid4(), 94)
    settings = QuickMontageSettings(
        semantic_analysis=True,
        target_duration_seconds=12,
        max_video_clip_seconds=3,
        max_scenes_per_source=1,
        strict_source_diversity=False,
        transition="none",
    )

    plan = build_semantic_montage_plan(
        [dominant, alternate],
        [*dominant_scenes, alternate_scene],
        settings,
    )

    dominant_clips = [clip for clip in plan.clips if clip.asset_id == dominant.id]
    assert len(dominant_clips) > 1


def test_semantic_selection_relaxes_event_limit_for_duration_coverage(
    tmp_path: Path,
) -> None:
    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    event_id = uuid4()
    assets = [
        _asset(tmp_path / f"same-event-{index}.mp4", created_at, duration=8) for index in range(5)
    ]
    scenes = [
        _scene(asset, event_id, 90 - index, start=0, duration=6)
        for index, asset in enumerate(assets)
    ]
    settings = QuickMontageSettings(
        semantic_analysis=True,
        target_duration_seconds=12,
        max_video_clip_seconds=3,
        max_scenes_per_event=2,
        transition="none",
    )

    plan = build_semantic_montage_plan(assets, scenes, settings)

    assert len(plan.clips) == 4
    assert plan.total_duration_seconds == 12
    assert {clip.event_id for clip in plan.clips} == {event_id}


def test_semantic_selection_relaxes_source_limit_for_few_long_videos(
    tmp_path: Path,
) -> None:
    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    source = _asset(tmp_path / "single-long-roll.mp4", created_at, duration=60)
    scenes = [_scene(source, uuid4(), 95 - index, start=index * 4) for index in range(5)]
    settings = QuickMontageSettings(
        semantic_analysis=True,
        target_duration_seconds=12,
        max_video_clip_seconds=3,
        max_scenes_per_source=2,
        transition="none",
    )

    plan = build_semantic_montage_plan([source], scenes, settings)

    assert len(plan.clips) == 4
    assert {clip.asset_id for clip in plan.clips} == {source.id}


def test_semantic_selection_uses_best_visual_window_inside_long_scene(
    tmp_path: Path,
) -> None:
    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    source = _asset(tmp_path / "long-scene.mp4", created_at, duration=12)
    scene = _scene(
        source,
        uuid4(),
        95,
        duration=12,
        quality_metrics={
            "panel_quality_scores": [35, 58, 92],
            "best_panel_index": 2,
            "best_panel_position": 0.88,
        },
    )
    settings = QuickMontageSettings(
        semantic_analysis=True,
        target_duration_seconds=5,
        max_video_clip_seconds=3,
        transition="none",
    )

    plan = build_semantic_montage_plan([source], [scene], settings)

    assert plan.clips[0].source_start_seconds == 9
    assert "best visual window 3/3" in plan.clips[0].selection_reason


def test_semantic_selection_prefers_explicit_highlight_window(
    tmp_path: Path,
) -> None:
    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    source = _asset(tmp_path / "highlight.mp4", created_at, duration=14)
    scene = _scene(
        source,
        uuid4(),
        95,
        duration=14,
        quality_metrics={"panel_quality_scores": [95, 40, 35]},
        highlight_windows=[
            {
                "relative_start_seconds": 6,
                "relative_end_seconds": 9,
                "score": 99,
                "label": "best smile",
            }
        ],
    )
    settings = QuickMontageSettings(
        semantic_analysis=True,
        target_duration_seconds=5,
        max_video_clip_seconds=3,
        transition="none",
    )

    plan = build_semantic_montage_plan([source], [scene], settings)

    assert plan.clips[0].source_start_seconds == 6
    assert "highlight window: best smile" in plan.clips[0].selection_reason


def test_semantic_selection_uses_quality_candidate_windows(
    tmp_path: Path,
) -> None:
    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    source = _asset(tmp_path / "candidate-window.mp4", created_at, duration=12)
    scene = _scene(
        source,
        uuid4(),
        95,
        duration=12,
        quality_metrics={
            "candidate_windows": [
                {
                    "relative_position": 0.25,
                    "score": 45,
                    "source": "visual_quality",
                    "label": "soft opening",
                },
                {
                    "relative_position": 0.75,
                    "score": 88,
                    "source": "visual_quality",
                    "label": "clean viewpoint",
                },
            ]
        },
    )
    settings = QuickMontageSettings(
        semantic_analysis=True,
        target_duration_seconds=5,
        max_video_clip_seconds=3,
        transition="none",
    )

    plan = build_semantic_montage_plan([source], [scene], settings)

    assert plan.clips[0].source_start_seconds == 7.5
    assert "visual candidate: clean viewpoint" in plan.clips[0].selection_reason


def test_semantic_selection_protects_people_from_quality_only_window(
    tmp_path: Path,
) -> None:
    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    source = _asset(tmp_path / "people-moment.mp4", created_at, duration=12)
    scene = _scene(
        source,
        uuid4(),
        82,
        duration=12,
        people_count=3,
        people_groups=["family"],
        vision_score_factors={"people": 88},
        quality_metrics={
            "candidate_windows": [
                {
                    "relative_position": 0.9,
                    "score": 94,
                    "source": "visual_quality",
                    "label": "empty but sharp",
                }
            ]
        },
    )
    settings = QuickMontageSettings(
        semantic_analysis=True,
        target_duration_seconds=5,
        max_video_clip_seconds=3,
        transition="none",
    )

    plan = build_semantic_montage_plan([source], [scene], settings)

    assert plan.clips[0].source_start_seconds == 4.5
    assert "people-safe center" in plan.clips[0].selection_reason


def test_semantic_selection_accepts_top_level_relative_candidate_window(
    tmp_path: Path,
) -> None:
    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    source = _asset(tmp_path / "future-audio-window.mp4", created_at, duration=12)
    scene = _scene(
        source,
        uuid4(),
        95,
        duration=12,
        candidate_windows=[
            {
                "relative_position": 0.7,
                "score": 93,
                "label": "future multimodal peak",
            }
        ],
    )
    settings = QuickMontageSettings(
        semantic_analysis=True,
        target_duration_seconds=5,
        max_video_clip_seconds=3,
        transition="none",
    )

    plan = build_semantic_montage_plan([source], [scene], settings)

    assert round(plan.clips[0].source_start_seconds, 1) == 6.9
    assert "highlight window: future multimodal peak" in plan.clips[0].selection_reason


def test_semantic_selection_prefers_speech_safe_window_over_cutting_phrase(
    tmp_path: Path,
) -> None:
    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    source = _asset(tmp_path / "speech-moment.mp4", created_at, duration=10)
    scene = _scene(
        source,
        uuid4(),
        95,
        duration=10,
        candidate_windows=[
            {
                "relative_position": 0.45,
                "score": 98,
                "source": "visual_quality",
                "label": "sharp but mid sentence",
            }
        ],
        speech_segments=[
            {
                "start_seconds": 2.0,
                "end_seconds": 4.0,
                "text": "Look at this view.",
                "confidence": 0.9,
            }
        ],
    )
    settings = QuickMontageSettings(
        semantic_analysis=True,
        target_duration_seconds=5,
        max_video_clip_seconds=3,
        transition="none",
    )

    plan = build_semantic_montage_plan([source], [scene], settings)

    assert round(plan.clips[0].source_start_seconds, 2) == 1.5
    assert "speech-safe window: Look at this view." in plan.clips[0].selection_reason


def test_semantic_selection_uses_best_panel_position_without_panel_scores(
    tmp_path: Path,
) -> None:
    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    source = _asset(tmp_path / "panel-position.mp4", created_at, duration=12)
    scene = _scene(
        source,
        uuid4(),
        95,
        duration=12,
        quality_metrics={
            "quality_score": 77,
            "best_panel_index": 1,
            "best_panel_position": 0.5,
        },
    )
    settings = QuickMontageSettings(
        semantic_analysis=True,
        target_duration_seconds=5,
        max_video_clip_seconds=3,
        transition="none",
    )

    plan = build_semantic_montage_plan([source], [scene], settings)

    assert plan.clips[0].source_start_seconds == 4.5
    assert "best visual panel 2" in plan.clips[0].selection_reason


def test_semantic_timeline_uses_story_section_order(tmp_path: Path) -> None:
    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    early_file = _asset(tmp_path / "early-highlight.mp4", created_at, duration=6)
    late_file = _asset(tmp_path / "late-opening.mp4", created_at, duration=6)
    highlight = _scene(
        early_file,
        uuid4(),
        94,
        duration=6,
        story_section_index=2,
        story_section_role="highlight",
        story_role_order=2,
    )
    opening = _scene(
        late_file,
        uuid4(),
        92,
        duration=6,
        story_section_index=0,
        story_section_role="opening",
        story_role_order=0,
    )
    settings = QuickMontageSettings(
        semantic_analysis=True,
        preserve_chronology=False,
        target_duration_seconds=6,
        max_video_clip_seconds=3,
        transition="none",
    )

    plan = build_semantic_montage_plan([early_file, late_file], [highlight, opening], settings)

    assert [clip.scene_id for clip in plan.clips] == [opening.id, highlight.id]


def test_semantic_timeline_preserves_capture_chronology_by_default(tmp_path: Path) -> None:
    early_file = _asset(
        tmp_path / "early-highlight.mp4",
        datetime(2026, 1, 1, tzinfo=UTC),
        duration=6,
    )
    late_file = _asset(
        tmp_path / "late-opening.mp4",
        datetime(2026, 1, 2, tzinfo=UTC),
        duration=6,
    )
    highlight = _scene(
        early_file,
        uuid4(),
        94,
        duration=6,
        story_section_index=2,
        story_section_role="highlight",
        story_role_order=2,
    )
    opening = _scene(
        late_file,
        uuid4(),
        92,
        duration=6,
        story_section_index=0,
        story_section_role="opening",
        story_role_order=0,
    )
    settings = QuickMontageSettings(
        semantic_analysis=True,
        target_duration_seconds=6,
        max_video_clip_seconds=3,
        transition="none",
    )

    plan = build_semantic_montage_plan([early_file, late_file], [highlight, opening], settings)

    assert [clip.scene_id for clip in plan.clips] == [highlight.id, opening.id]


def test_semantic_timeline_preserves_scene_order_inside_source_video(tmp_path: Path) -> None:
    source = _asset(
        tmp_path / "single-source.mp4",
        datetime(2026, 1, 1, tzinfo=UTC),
        duration=20,
    )
    early_highlight = _scene(
        source,
        uuid4(),
        94,
        start=0,
        duration=4,
        story_section_index=2,
        story_section_role="highlight",
        story_role_order=2,
    )
    late_opening = _scene(
        source,
        uuid4(),
        92,
        start=10,
        duration=4,
        story_section_index=0,
        story_section_role="opening",
        story_role_order=0,
    )
    settings = QuickMontageSettings(
        semantic_analysis=True,
        target_duration_seconds=8,
        max_video_clip_seconds=3,
        transition="none",
    )

    plan = build_semantic_montage_plan(
        [source],
        [early_highlight, late_opening],
        settings,
    )

    assert [clip.scene_id for clip in plan.clips] == [early_highlight.id, late_opening.id]


def test_semantic_timeline_can_use_story_order_over_capture_chronology(
    tmp_path: Path,
) -> None:
    early_file = _asset(
        tmp_path / "early-highlight.mp4",
        datetime(2026, 1, 1, tzinfo=UTC),
        duration=6,
    )
    late_file = _asset(
        tmp_path / "late-opening.mp4",
        datetime(2026, 1, 2, tzinfo=UTC),
        duration=6,
    )
    highlight = _scene(
        early_file,
        uuid4(),
        94,
        duration=6,
        story_section_index=2,
        story_section_role="highlight",
        story_role_order=2,
    )
    opening = _scene(
        late_file,
        uuid4(),
        92,
        duration=6,
        story_section_index=0,
        story_section_role="opening",
        story_role_order=0,
    )
    settings = QuickMontageSettings(
        semantic_analysis=True,
        preserve_chronology=False,
        target_duration_seconds=6,
        max_video_clip_seconds=3,
        transition="none",
    )

    plan = build_semantic_montage_plan([early_file, late_file], [highlight, opening], settings)

    assert [clip.scene_id for clip in plan.clips] == [opening.id, highlight.id]


def test_story_timeline_uses_section_duration_budgets(tmp_path: Path) -> None:
    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    assets = [
        _asset(tmp_path / f"section-{index}.mp4", created_at, duration=6) for index in range(7)
    ]
    scenes = [
        _scene(
            assets[index],
            uuid4(),
            99 - index,
            duration=6,
            story_section_index=0,
            story_section_role="opening",
            story_role_order=0,
        )
        for index in range(4)
    ]
    scenes.extend(
        [
            _scene(
                assets[4],
                uuid4(),
                88,
                duration=6,
                story_section_index=1,
                story_section_role="journey",
                story_role_order=1,
            ),
            _scene(
                assets[5],
                uuid4(),
                96,
                duration=6,
                story_section_index=2,
                story_section_role="highlight",
                story_role_order=2,
            ),
            _scene(
                assets[6],
                uuid4(),
                86,
                duration=6,
                story_section_index=3,
                story_section_role="finale",
                story_role_order=3,
            ),
        ]
    )
    settings = QuickMontageSettings(
        semantic_analysis=True,
        target_duration_seconds=12,
        max_video_clip_seconds=3,
        transition="none",
    )

    plan = build_semantic_montage_plan(assets, scenes, settings)
    selected_roles = [
        next(scene for scene in scenes if scene.id == clip.scene_id).metadata["story_section_role"]
        for clip in plan.clips
    ]

    assert selected_roles == ["opening", "journey", "highlight", "finale"]


def test_story_timeline_avoids_adjacent_similar_scenes_when_possible(
    tmp_path: Path,
) -> None:
    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    assets = [
        _asset(tmp_path / f"diverse-{index}.mp4", created_at, duration=6) for index in range(4)
    ]
    scenes = [
        _scene(
            assets[0],
            uuid4(),
            95,
            duration=6,
            story_section_index=1,
            story_section_role="journey",
            story_role_order=1,
            location_type="beach",
            activity="walking",
            tags=["sea", "walk"],
        ),
        _scene(
            assets[1],
            uuid4(),
            94,
            duration=6,
            story_section_index=1,
            story_section_role="journey",
            story_role_order=1,
            location_type="beach",
            activity="walking",
            tags=["sea", "walk"],
        ),
        _scene(
            assets[2],
            uuid4(),
            90,
            duration=6,
            story_section_index=1,
            story_section_role="journey",
            story_role_order=1,
            location_type="mountain",
            activity="hiking",
            tags=["viewpoint"],
        ),
        _scene(
            assets[3],
            uuid4(),
            89,
            duration=6,
            story_section_index=1,
            story_section_role="journey",
            story_role_order=1,
            location_type="beach",
            activity="walking",
            tags=["sea"],
        ),
    ]
    settings = QuickMontageSettings(
        semantic_analysis=True,
        preserve_chronology=False,
        target_duration_seconds=9,
        max_video_clip_seconds=3,
        transition="none",
    )

    plan = build_semantic_montage_plan(assets, scenes, settings)
    by_id = {scene.id: scene for scene in scenes}
    first = by_id[plan.clips[0].scene_id]
    second = by_id[plan.clips[1].scene_id]

    assert first.metadata["location_type"] != second.metadata["location_type"]
    assert first.metadata["activity"] != second.metadata["activity"]


def test_story_pacing_shortens_highlights_in_longer_movies(tmp_path: Path) -> None:
    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    assets = [
        _asset(tmp_path / f"pacing-{index}.mp4", created_at, duration=10) for index in range(2)
    ]
    journey = _scene(
        assets[0],
        uuid4(),
        88,
        duration=10,
        story_section_role="journey",
        story_role_order=1,
    )
    highlight = _scene(
        assets[1],
        uuid4(),
        98,
        duration=10,
        story_section_role="highlight",
        story_role_order=2,
    )
    settings = QuickMontageSettings(
        semantic_analysis=True,
        target_duration_seconds=24,
        max_video_clip_seconds=6,
        transition="none",
    )

    plan = build_semantic_montage_plan(assets, [journey, highlight], settings)
    durations = {clip.scene_id: clip.duration_seconds for clip in plan.clips}

    assert durations[highlight.id] < durations[journey.id]


def test_story_pacing_uses_energy_and_speech_protection(tmp_path: Path) -> None:
    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    assets = [
        _asset(tmp_path / f"energy-{index}.mp4", created_at, duration=10) for index in range(3)
    ]
    energetic = _scene(
        assets[0],
        uuid4(),
        96,
        duration=10,
        emotion="exciting",
        quality_metrics={"motion_score": 90},
        story_section_role="journey",
        story_role_order=1,
    )
    speech = _scene(
        assets[1],
        uuid4(),
        95,
        duration=10,
        emotion="exciting",
        quality_metrics={"motion_score": 90},
        story_section_role="journey",
        story_role_order=1,
        speech_segments=[
            {
                "start_seconds": 1.0,
                "end_seconds": 4.0,
                "text": "This part needs room to breathe.",
            }
        ],
    )
    calm = _scene(
        assets[2],
        uuid4(),
        94,
        duration=10,
        activity="sightseeing",
        emotion="relaxing",
        quality_metrics={"motion_score": 5},
        story_section_role="journey",
        story_role_order=1,
    )
    settings = QuickMontageSettings(
        semantic_analysis=True,
        target_duration_seconds=30,
        max_video_clip_seconds=6,
        transition="none",
    )

    plan = build_semantic_montage_plan(assets, [energetic, speech, calm], settings)
    durations = {clip.scene_id: clip.duration_seconds for clip in plan.clips}
    reasons = {clip.scene_id: clip.selection_reason for clip in plan.clips}

    assert round(durations[energetic.id], 2) == 5.28
    assert round(durations[speech.id], 2) == 5.88
    assert durations[calm.id] == 6
    assert "pacing: high energy" in reasons[energetic.id]
    assert "pacing: speech hold" in reasons[speech.id]


def test_semantic_selection_backfills_after_directed_pacing_shortens_clips(
    tmp_path: Path,
) -> None:
    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    assets = [
        _asset(tmp_path / f"paced-{index}.mp4", created_at, duration=8) for index in range(20)
    ]
    event_ids = [uuid4() for _ in range(4)]
    scenes = [
        _scene(
            asset,
            event_ids[index % len(event_ids)],
            95 - index * 0.1,
            duration=8,
            emotion="exciting",
            quality_metrics={"motion_score": 90},
            story_section_role="journey",
            story_role_order=1,
        )
        for index, asset in enumerate(assets)
    ]
    settings = QuickMontageSettings(
        semantic_analysis=True,
        target_duration_seconds=90,
        max_video_clip_seconds=6,
        max_scenes_per_event=4,
        transition="cinematic",
        transition_duration_seconds=0.45,
    )

    plan = build_semantic_montage_plan(assets, scenes, settings)

    assert abs(plan.total_duration_seconds - 90) <= 0.05
    assert len(plan.clips) > settings.max_scenes_per_event * len(event_ids)


def test_semantic_timeline_directs_cinematic_transitions_by_event(tmp_path: Path) -> None:
    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    assets = [
        _asset(tmp_path / f"transition-{index}.mp4", created_at, duration=8) for index in range(3)
    ]
    first_event = uuid4()
    second_event = uuid4()
    opening = _scene(
        assets[0],
        first_event,
        92,
        duration=8,
        story_section_role="opening",
        story_role_order=0,
        activity="viewing",
    )
    journey = _scene(
        assets[1],
        first_event,
        90,
        duration=8,
        story_section_role="journey",
        story_role_order=1,
        activity="walking",
    )
    finale = _scene(
        assets[2],
        second_event,
        91,
        duration=8,
        story_section_role="finale",
        story_role_order=3,
        activity="viewing",
    )
    settings = QuickMontageSettings(
        semantic_analysis=True,
        target_duration_seconds=18,
        max_video_clip_seconds=5,
        transition="cinematic",
        transition_duration_seconds=0.4,
    )

    plan = build_semantic_montage_plan(assets, [opening, journey, finale], settings)
    transitions = {clip.scene_id: clip.transition for clip in plan.clips}

    assert transitions[opening.id] is None
    assert transitions[journey.id] == "dissolve"
    assert transitions[finale.id] == "fade"
    assert round(plan.total_duration_seconds, 3) == 14.2


def test_music_directing_moves_cut_to_strong_beat_without_changing_total_duration(
    tmp_path: Path,
) -> None:
    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    assets = [_asset(tmp_path / f"beat-{index}.mp4", created_at, duration=8) for index in range(3)]
    scenes = [_scene(asset, uuid4(), 90 - index, duration=8) for index, asset in enumerate(assets)]
    settings = QuickMontageSettings(
        semantic_analysis=True,
        target_duration_seconds=9.2,
        max_video_clip_seconds=4,
        transition="none",
    )
    clips = [
        MontageClip(
            asset_id=asset.id,
            scene_id=scene.id,
            source_path=asset.path,
            relative_path=asset.relative_path,
            media_type=MediaType.VIDEO,
            duration_seconds=duration,
            semantic_score=90,
            selection_reason="vision 90",
        )
        for asset, scene, duration in zip(assets, scenes, [3.2, 3.0, 3.0], strict=True)
    ]
    plan = QuickMontagePlan(
        created_at=datetime.now(UTC),
        settings=settings,
        clips=clips,
        total_duration_seconds=9.2,
        music_plan=MusicPlan(
            mode="generated",
            bpm=80,
            duration_seconds=9.2,
            beat_grid=[
                MusicBeat(time_seconds=3.0, beat_index=4, bar_index=1, strength=0.9),
                MusicBeat(time_seconds=6.2, beat_index=8, bar_index=2, strength=0.3),
            ],
        ),
        selection_mode="semantic",
    )

    directed = apply_music_directing(plan, scenes)

    assert directed.total_duration_seconds == plan.total_duration_seconds
    assert directed.clips[0].duration_seconds == 3.0
    assert directed.clips[1].duration_seconds == 3.2
    assert "music beat start" in directed.clips[1].selection_reason


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
    duration: float = 3,
    **metadata: object,
) -> Scene:
    return Scene(
        asset_id=asset.id,
        start_seconds=start,
        end_seconds=start + duration,
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
