import json
import shutil
import subprocess
import wave
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from travelmovieai.core.exceptions import MontageError
from travelmovieai.domain.enums import MediaType
from travelmovieai.domain.models import (
    MontageClip,
    MontageQualityReport,
    MusicBeat,
    MusicCueSection,
    MusicPlan,
    QuickMontagePlan,
    QuickMontageSettings,
    Scene,
)
from travelmovieai.editing.quality_report import (
    _rendered_audio_rms,
    build_montage_quality_report,
    enforce_montage_quality,
    enrich_montage_quality_report_with_render,
)


def test_render_audio_end_check_uses_multiple_windows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sampled_starts: list[float] = []

    def fake_rms(
        output_path: Path,
        *,
        start_seconds: float,
        duration_seconds: float,
        ffmpeg_binary: str,
        timeout_seconds: float,
    ) -> float:
        del output_path, duration_seconds, ffmpeg_binary, timeout_seconds
        sampled_starts.append(start_seconds)
        return 25 if start_seconds > 27 else 400

    monkeypatch.setattr("travelmovieai.editing.quality_report._audio_rms", fake_rms)

    values = _rendered_audio_rms(
        tmp_path / "movie.mp4",
        duration_seconds=30,
        has_audio=True,
        ffmpeg_binary="ffmpeg",
        timeout_seconds=30,
    )

    assert len(sampled_starts) == 5
    assert values["end"] == 400


def test_quality_gate_rejects_critical_report() -> None:
    plan = QuickMontagePlan(
        created_at=datetime.now(UTC),
        settings=QuickMontageSettings(),
    )
    report = build_montage_quality_report(plan, [])

    with pytest.raises(MontageError, match="quality gate"):
        enforce_montage_quality(report)


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


def test_montage_quality_report_records_music_quality_metadata(
    tmp_path: Path,
) -> None:
    music = tmp_path / "music.wav"
    with wave.open(str(music), "wb") as audio:
        audio.setnchannels(2)
        audio.setsampwidth(2)
        audio.setframerate(8000)
        audio.writeframes((b"\x10\x27\x10\x27") * 8000)
    settings = QuickMontageSettings(target_duration_seconds=5, transition="none")
    clip = MontageClip(
        asset_id=uuid4(),
        source_path=tmp_path / "clip.mp4",
        relative_path=Path("clip.mp4"),
        media_type=MediaType.VIDEO,
        duration_seconds=4,
    )
    plan = QuickMontagePlan(
        created_at=datetime.now(UTC),
        settings=settings,
        clips=[clip],
        total_duration_seconds=4,
        music_plan=MusicPlan(
            mode="generated",
            source_path=music,
            duration_seconds=4,
            cue_sections=[
                MusicCueSection(
                    role="journey",
                    start_seconds=0,
                    end_seconds=5,
                    bpm=76,
                    intensity=0.45,
                )
            ],
            beat_grid=[
                MusicBeat(
                    time_seconds=0,
                    beat_index=0,
                    bar_index=0,
                    strength=0.72,
                )
            ],
        ),
    )

    report = build_montage_quality_report(plan, [])

    assert report.music_cue_section_count == 1
    assert report.music_beat_count == 1
    assert report.music_loudness_rms is not None
    assert report.music_loudness_rms > 0
    assert report.music_peak_ratio is not None
    assert report.music_clipping_ratio == 0


def test_montage_quality_report_flags_unsynced_music_cuts(tmp_path: Path) -> None:
    settings = QuickMontageSettings(target_duration_seconds=9, transition="none")
    clips = [
        MontageClip(
            asset_id=uuid4(),
            source_path=tmp_path / f"clip-{index}.mp4",
            relative_path=Path(f"clip-{index}.mp4"),
            media_type=MediaType.VIDEO,
            duration_seconds=duration,
            selection_reason="vision 80",
        )
        for index, duration in enumerate([3.2, 3.0, 2.8])
    ]
    plan = QuickMontagePlan(
        created_at=datetime.now(UTC),
        settings=settings,
        clips=clips,
        total_duration_seconds=9,
        music_plan=MusicPlan(
            mode="generated",
            duration_seconds=9,
            beat_grid=[
                MusicBeat(time_seconds=3.0, beat_index=4, bar_index=1, strength=0.9),
                MusicBeat(time_seconds=6.0, beat_index=8, bar_index=2, strength=0.9),
            ],
        ),
    )

    report = build_montage_quality_report(plan, [])

    assert "unsynced_music_cuts" in {issue.code for issue in report.issues}


def test_montage_quality_report_accounts_for_transition_overlaps_in_music_sync(
    tmp_path: Path,
) -> None:
    settings = QuickMontageSettings(
        target_duration_seconds=11,
        transition="fade",
        transition_duration_seconds=0.5,
    )
    clips = [
        MontageClip(
            asset_id=uuid4(),
            source_path=tmp_path / f"clip-{index}.mp4",
            relative_path=Path(f"clip-{index}.mp4"),
            media_type=MediaType.VIDEO,
            duration_seconds=4,
            selection_reason="vision 80",
        )
        for index in range(3)
    ]
    plan = QuickMontagePlan(
        created_at=datetime.now(UTC),
        settings=settings,
        clips=clips,
        total_duration_seconds=11,
        music_plan=MusicPlan(
            mode="generated",
            duration_seconds=11,
            beat_grid=[
                MusicBeat(time_seconds=3.5, beat_index=4, bar_index=1, strength=0.9),
                MusicBeat(time_seconds=7.0, beat_index=8, bar_index=2, strength=0.9),
            ],
        ),
    )

    report = build_montage_quality_report(plan, [])

    assert "unsynced_music_cuts" not in {issue.code for issue in report.issues}


def test_montage_quality_report_flags_speech_boundary_cuts(tmp_path: Path) -> None:
    asset_id = uuid4()
    scene = Scene(
        asset_id=asset_id,
        start_seconds=10,
        end_seconds=18,
        quality_score=80,
        importance_score=80,
        metadata={
            "speech_segments": [
                {
                    "start_seconds": 1.0,
                    "end_seconds": 4.0,
                    "text": "This should not be cut.",
                }
            ]
        },
    )
    plan = QuickMontagePlan(
        created_at=datetime.now(UTC),
        settings=QuickMontageSettings(target_duration_seconds=5, transition="none"),
        clips=[
            MontageClip(
                asset_id=asset_id,
                scene_id=scene.id,
                source_path=tmp_path / "speech.mp4",
                relative_path=Path("speech.mp4"),
                media_type=MediaType.VIDEO,
                source_start_seconds=12.0,
                duration_seconds=2.0,
                selection_reason="vision 80",
            )
        ],
        total_duration_seconds=2,
        selection_mode="semantic",
    )

    report = build_montage_quality_report(plan, [scene])

    assert "speech_boundary_cut" in {issue.code for issue in report.issues}


def test_montage_quality_report_flags_excessive_center_cuts(tmp_path: Path) -> None:
    settings = QuickMontageSettings(target_duration_seconds=12, transition="none")
    clips = [
        MontageClip(
            asset_id=uuid4(),
            source_path=tmp_path / f"center-{index}.mp4",
            relative_path=Path(f"center-{index}.mp4"),
            media_type=MediaType.VIDEO,
            duration_seconds=3,
            selection_reason="vision 80; center of scene",
        )
        for index in range(4)
    ]
    plan = QuickMontagePlan(
        created_at=datetime.now(UTC),
        settings=settings,
        clips=clips,
        total_duration_seconds=12,
        selection_mode="semantic",
    )

    report = build_montage_quality_report(plan, [])

    assert "excessive_center_cuts" in {issue.code for issue in report.issues}


def test_render_quality_report_reports_ffprobe_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "final.mp4"
    output.write_bytes(b"fake")
    timeouts: list[float] = []

    def fake_run(command: list[str], **kwargs: object) -> object:
        timeout = kwargs["timeout"]
        assert isinstance(timeout, int | float)
        timeouts.append(float(timeout))
        raise subprocess.TimeoutExpired(cmd=command, timeout=timeout)

    monkeypatch.setattr("travelmovieai.editing.quality_report.subprocess.run", fake_run)

    with pytest.raises(MontageError, match="FFprobe timed out after 0.25s"):
        enrich_montage_quality_report_with_render(
            _render_report(),
            output,
            timeout_seconds=0.25,
        )

    assert timeouts == [0.25]


def test_render_quality_report_treats_sample_timeouts_as_unavailable_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "final.mp4"
    output.write_bytes(b"fake")
    calls = 0

    def fake_run(command: list[str], **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        timeout = kwargs["timeout"]
        if calls == 1:
            return type(
                "Completed",
                (),
                {
                    "returncode": 0,
                    "stdout": json.dumps(
                        {
                            "format": {"duration": "3.0"},
                            "streams": [
                                {"codec_type": "video"},
                                {"codec_type": "audio"},
                            ],
                        }
                    ),
                    "stderr": "",
                },
            )()
        raise subprocess.TimeoutExpired(cmd=command, timeout=timeout)

    monkeypatch.setattr("travelmovieai.editing.quality_report.subprocess.run", fake_run)

    report = enrich_montage_quality_report_with_render(
        _render_report(),
        output,
        timeout_seconds=0.25,
    )

    assert report.rendered_has_video is True
    assert report.rendered_has_audio is True
    assert report.rendered_audio_rms == {}
    assert report.rendered_video_luma == {}
    assert "render_audio_rms_unavailable" in {issue.code for issue in report.issues}


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is not installed")
@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="FFprobe is not installed")
def test_render_quality_report_ignores_intentional_music_fade_out(
    tmp_path: Path,
) -> None:
    output = tmp_path / "faded-ending.mp4"
    _generate_faded_ending_movie(output)
    report = _render_report().model_copy(
        update={
            "planned_duration_seconds": 12,
            "target_duration_seconds": 12,
            "music_mode": "generated",
        }
    )

    enriched = enrich_montage_quality_report_with_render(report, output)

    assert enriched.rendered_audio_rms["end"] >= 50
    assert "render_audio_fades_out_early" not in {issue.code for issue in enriched.issues}


def _generate_faded_ending_movie(path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x240:rate=24:duration=12",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=12",
            "-filter:a",
            "volume=0.03,afade=t=out:st=10.5:d=1.5",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ],
        check=True,
    )


def _render_report() -> MontageQualityReport:
    return MontageQualityReport(
        created_at=datetime.now(UTC),
        score=100,
        target_duration_seconds=3,
        planned_duration_seconds=3,
        duration_ratio=1,
        clip_count=1,
        selected_scene_count=0,
        selected_event_count=0,
        total_event_count=0,
        event_coverage_ratio=1,
        source_count=1,
        dominant_source_ratio=1,
    )
