from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from travelmovieai.domain.enums import MediaType
from travelmovieai.domain.models import (
    MontageClip,
    QuickMontagePlan,
    QuickMontageSettings,
    Scene,
)
from travelmovieai.editing.quality_report import build_montage_quality_report


def test_montage_quality_report_flags_timeline_risks(tmp_path: Path) -> None:
    asset_id = uuid4()
    event_id = uuid4()
    scene = Scene(
        asset_id=asset_id,
        start_seconds=0,
        end_seconds=3,
        quality_score=28,
        importance_score=45,
        metadata={
            "event_id": str(event_id),
            "quality_metrics": {
                "brightness": 10,
                "sharpness": 18,
                "rejection_reasons": ["too_dark", "blurred"],
            },
        },
    )
    settings = QuickMontageSettings(
        target_duration_seconds=10,
        max_video_clip_seconds=3,
        transition="none",
        music_enabled=False,
    )
    clips = [
        MontageClip(
            asset_id=asset_id,
            scene_id=scene.id,
            source_path=tmp_path / f"clip-{index}.mp4",
            relative_path=Path(f"clip-{index}.mp4"),
            media_type=MediaType.VIDEO,
            duration_seconds=1.5,
            semantic_score=45,
            event_id=event_id,
            selection_reason="vision 45; center of scene",
        )
        for index in range(3)
    ]
    plan = QuickMontagePlan(
        created_at=datetime.now(UTC),
        settings=settings,
        clips=clips,
        total_duration_seconds=4.5,
        selection_mode="semantic",
    )

    report = build_montage_quality_report(plan, [scene])
    codes = {issue.code for issue in report.issues}

    assert report.score < 100
    assert report.duration_ratio == 0.45
    assert report.window_selection["center"] == 3
    assert "short_timeline" in codes
    assert "low_semantic_score" in codes
    assert "low_visual_quality" in codes
    assert "music_disabled" in codes
    assert "selected_dark_scene" in codes
    assert "selected_blurred_scene" in codes
