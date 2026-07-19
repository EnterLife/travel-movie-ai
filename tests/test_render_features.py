import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from travelmovieai.core.exceptions import MontageError
from travelmovieai.domain.enums import MediaType
from travelmovieai.domain.models import (
    MediaAsset,
    MontageClip,
    MusicCueSection,
    MusicPlan,
    QuickMontagePlan,
    QuickMontageSettings,
    Scene,
)
from travelmovieai.editing.renderer import (
    QuickMontageRenderer,
    _active_narration_path,
    _build_filter_graph,
    _build_segment_video_graph,
    _truncate_overlay_text,
)
from travelmovieai.editing.timeline import build_semantic_montage_plan
from travelmovieai.infrastructure.ffmpeg import parse_probe_payload
from travelmovieai.story import music


def test_render_feature_settings_are_opt_in_and_validate_safe_area() -> None:
    settings = QuickMontageSettings()

    assert settings.photo_motion == "none"
    assert settings.framing_mode == "fit"
    assert settings.vertical_video_layout == "fit"
    assert settings.color_normalization is False
    assert settings.hdr_to_sdr is False
    assert settings.event_titles_enabled is False
    assert settings.scene_subtitles_enabled is False
    assert settings.music_bpm_analysis is False
    assert settings.music_volume_envelope is False
    with pytest.raises(ValueError, match="overlay_safe_margin"):
        QuickMontageSettings(overlay_safe_margin=0.01)
    with pytest.raises(ValueError, match="photo_zoom_ratio"):
        QuickMontageSettings(photo_zoom_ratio=1.5)


def test_probe_retains_rotation_dimensions_and_hdr_metadata() -> None:
    result = parse_probe_payload(
        {
            "format": {"duration": "4.2"},
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "hevc",
                    "width": 1080,
                    "height": 1920,
                    "color_space": "bt2020nc",
                    "color_transfer": "smpte2084",
                    "color_primaries": "bt2020",
                    "side_data_list": [{"rotation": -90}],
                }
            ],
        }
    )

    stream = result.metadata["streams"][0]
    assert stream["rotation_degrees"] == 270
    assert stream["width"] == 1080
    assert stream["height"] == 1920
    assert stream["color_transfer"] == "smpte2084"


def test_semantic_plan_carries_face_focus_and_render_metadata(tmp_path: Path) -> None:
    asset_id = uuid4()
    event_id = uuid4()
    asset = MediaAsset(
        id=asset_id,
        path=tmp_path / "портретное видео.mp4",
        relative_path=Path("портретное видео.mp4"),
        media_type=MediaType.VIDEO,
        extension=".mp4",
        size_bytes=10,
        modified_at=datetime.now(UTC),
        modified_ns=1,
        duration_seconds=8,
        width=1080,
        height=1920,
        probe_metadata={
            "streams": [
                {
                    "codec_type": "video",
                    "width": 1080,
                    "height": 1920,
                    "rotation_degrees": 90,
                    "color_transfer": "arib-std-b67",
                }
            ]
        },
    )
    scene = Scene(
        asset_id=asset_id,
        start_seconds=0,
        end_seconds=6,
        caption="Люди у моря",
        importance_score=90,
        metadata={
            "selection_override": "include",
            "ranking_score": 95,
            "event_id": str(event_id),
            "event_title": "Вечер в Сочи",
            "face_boxes": [{"x": 0.1, "y": 0.2, "width": 0.2, "height": 0.4, "confidence": 0.9}],
            "quality_metrics": {
                "brightness": 20,
                "exposure_score": 40,
                "contrast": 30,
                "saturation": 80,
            },
        },
    )

    plan = build_semantic_montage_plan(
        [asset],
        [scene],
        QuickMontageSettings(target_duration_seconds=5),
    )

    clip = plan.clips[0]
    assert clip.event_title == "Вечер в Сочи"
    assert clip.source_width == 1080
    assert clip.source_height == 1920
    assert clip.rotation_degrees == 90
    assert clip.color_transfer == "arib-std-b67"
    assert clip.focus_source == "face"
    assert clip.focus_x == pytest.approx(0.2)
    assert clip.focus_y == pytest.approx(0.4)
    assert clip.brightness_adjustment == pytest.approx(0.05)
    assert clip.contrast_multiplier == pytest.approx(1.08)
    assert clip.saturation_multiplier == pytest.approx(0.88)


def test_smart_crop_falls_back_to_fit_without_valid_focus(tmp_path: Path) -> None:
    clip = _clip(tmp_path / "wide.mp4", source_width=1920, source_height=1080)
    plan = _plan(clip, QuickMontageSettings(framing_mode="smart"))

    graph = _build_segment_video_graph(
        clip,
        plan,
        trim_start=0,
        show_event_title=False,
        show_credits=False,
    )

    assert "force_original_aspect_ratio=decrease" in graph
    assert "pad=1280:720" in graph
    assert "force_original_aspect_ratio=increase" not in graph


def test_video_filters_cover_rotation_vertical_hdr_and_color(tmp_path: Path) -> None:
    clip = _clip(
        tmp_path / "vertical.mp4",
        source_width=1080,
        source_height=1920,
        rotation_degrees=180,
        color_transfer="smpte2084",
    )
    plan = _plan(
        clip,
        QuickMontageSettings(
            vertical_video_layout="blur",
            hdr_to_sdr=True,
            color_normalization=True,
        ),
    )

    graph = _build_segment_video_graph(
        clip,
        plan,
        trim_start=0,
        show_event_title=False,
        show_credits=False,
    )

    assert "hflip,vflip" in graph
    assert "zscale=transfer=linear:npl=100" in graph
    assert "tonemap=mobius" in graph
    assert "eq=contrast=1.0000:brightness=0.0000:saturation=1.0000" in graph
    assert "split=2[vbackgroundsource][vforegroundsource]" in graph
    assert "boxblur=20:1" in graph


def test_photo_ken_burns_and_unicode_overlays_are_escaped(tmp_path: Path) -> None:
    clip = _clip(
        tmp_path / "фото семьи.jpg",
        media_type=MediaType.PHOTO,
        caption="Море: 100% — семья, солнце",
        event_title="День семьи's",
        focus_x=0.7,
        focus_y=0.4,
    )
    settings = QuickMontageSettings(
        photo_motion="ken_burns",
        event_titles_enabled=True,
        scene_subtitles_enabled=True,
        credits_text="Снято в Сочи: команда's 100%",
        overlay_safe_margin=0.08,
    )
    plan = _plan(clip, settings)

    graph = _build_segment_video_graph(
        clip,
        plan,
        trim_start=0,
        show_event_title=True,
        show_credits=True,
    )

    assert "zoompan=z='min(zoom+" in graph
    assert "iw*0.7000" in graph
    assert graph.count("drawtext=") == 3
    assert "Море\\: 100\\%" in graph
    assert "семья\\, солнце" in graph
    assert "команда'\\''s" in graph
    assert "w*0.080" in graph


def test_long_overlay_text_uses_ascii_ellipsis_and_a_word_boundary() -> None:
    settings = QuickMontageSettings(overlay_max_characters=24)

    truncated = _truncate_overlay_text(
        "A caption with several words that cannot fit",
        settings,
        font_height_divisor=24,
    )

    assert truncated == "A caption with..."
    assert len(truncated) <= settings.overlay_max_characters
    assert "…" not in truncated


def test_overlay_text_at_the_limit_is_not_modified() -> None:
    settings = QuickMontageSettings(overlay_max_characters=20)

    text = "Exactly twenty chars"

    assert len(text) == 20
    assert _truncate_overlay_text(text, settings, font_height_divisor=24) == text


def test_overlay_text_normalizes_font_fragile_punctuation() -> None:
    settings = QuickMontageSettings()

    normalized = _truncate_overlay_text(
        "TravelMovieAI • Sochi — ‘2026’…",
        settings,
        font_height_divisor=20,
    )

    assert normalized == "TravelMovieAI - Sochi - '2026'..."


def test_renderer_uses_unicode_argv_and_disables_ffmpeg_autorotate(tmp_path: Path) -> None:
    captured: list[list[str]] = []

    class CapturingRenderer(QuickMontageRenderer):
        def _run(self, command: list[str], message: str) -> None:
            captured.append(command)

    clip = _clip(tmp_path / "поворот 90°.mp4", rotation_degrees=90)
    plan = _plan(clip, QuickMontageSettings())

    CapturingRenderer()._render_segment(clip, plan, tmp_path / "готовый сегмент.mp4")

    command = captured[0]
    assert str(clip.source_path) in command
    assert command[command.index("-noautorotate") + 1] == "-i"
    graph = command[command.index("-filter_complex") + 1]
    assert "transpose=cclock" in graph


def test_audio_graph_mixes_narration_and_automatic_music_envelope(tmp_path: Path) -> None:
    clip = _clip(tmp_path / "clip.mp4")
    settings = QuickMontageSettings(
        narration_enabled=True,
        narration_volume=1.25,
        background_volume_during_narration=0.25,
        source_audio_volume=0.4,
        music_volume=0.8,
        music_volume_envelope=True,
    )
    music_plan = MusicPlan(
        mode="manual",
        source_path=tmp_path / "music.wav",
        bpm=120,
        cue_sections=[
            MusicCueSection(
                role="intro",
                start_seconds=0,
                end_seconds=2,
                bpm=120,
                intensity=0.4,
            )
        ],
    )
    plan = QuickMontagePlan(
        created_at=datetime.now(UTC),
        settings=settings,
        clips=[clip],
        total_duration_seconds=2,
        music_path=tmp_path / "music.wav",
        music_plan=music_plan,
        narration_path=tmp_path / "narration.wav",
    )

    graph = _build_filter_graph(plan, transition_duration=0)

    assert "volume=0.400[sourceaudio]" in graph
    assert "volume='0.800*(if(between(t,0.000,2.000),0.790,0.700))':eval=frame" in graph
    assert "[2:a]aresample=48000" in graph
    assert "volume=1.250,asplit=2[narrationsc][narrationmix]" in graph
    assert "ratio=15.25" in graph
    assert "[duckedbackground][narrationmix]amix=inputs=2" in graph


def test_disabled_or_missing_narration_is_handled_safely(tmp_path: Path) -> None:
    clip = _clip(tmp_path / "clip.mp4")
    missing = tmp_path / "missing narration.wav"
    disabled = _plan(
        clip,
        QuickMontageSettings(narration_enabled=False),
        narration_path=missing,
    )
    assert _active_narration_path(disabled) is None

    enabled = disabled.model_copy(update={"settings": QuickMontageSettings(narration_enabled=True)})
    with pytest.raises(MontageError, match="Narration file does not exist"):
        QuickMontageRenderer().render(enabled, tmp_path / "out.mp4", tmp_path / "render")


def test_manual_music_bpm_analysis_rebuilds_beat_grid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    soundtrack = tmp_path / "ритм.wav"
    soundtrack.write_bytes(b"local")
    clip = _clip(tmp_path / "clip.mp4")
    montage = _plan(clip, QuickMontageSettings())
    settings = QuickMontageSettings(
        music_mode="manual",
        music_path=soundtrack,
        music_bpm_analysis=True,
    )
    monkeypatch.setattr(
        music,
        "_analyze_music_bpm",
        lambda path, *, ffmpeg_binary="ffmpeg": 123,
    )

    result = music.build_music_plan(
        [],
        [],
        settings,
        tmp_path,
        tmp_path / "generated.wav",
        montage,
    )

    assert result.bpm == 123
    assert result.beat_grid[1].time_seconds == pytest.approx(60 / 123, abs=0.001)
    assert all(section.bpm == 123 for section in result.cue_sections)


def test_bpm_analysis_falls_back_when_ffmpeg_cannot_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    def fail_decode(command: list[str], **kwargs: object) -> SimpleNamespace:
        commands.append(command)
        return SimpleNamespace(returncode=1, stdout=b"")

    monkeypatch.setattr(
        music.subprocess,
        "run",
        fail_decode,
    )

    assert (
        music._analyze_music_bpm(
            tmp_path / "broken.mp3",
            ffmpeg_binary="custom-ffmpeg",
        )
        is None
    )
    assert commands[0][0] == "custom-ffmpeg"


def test_bpm_estimator_detects_regular_click_track() -> None:
    sample_rate = 11_025
    samples = np.zeros(sample_rate * 12, dtype=np.float64)
    beat_interval = sample_rate // 2
    for start in range(0, len(samples), beat_interval):
        end = min(len(samples), start + 80)
        samples[start:end] = np.linspace(1.0, 0.0, end - start)

    assert music._estimate_bpm(samples, sample_rate=sample_rate) == pytest.approx(120, abs=3)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is not installed")
@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="FFprobe is not installed")
def test_renderer_creates_movie_with_unicode_titles_ken_burns_and_narration(
    tmp_path: Path,
) -> None:
    photo = tmp_path / "фото Сочи.jpg"
    music_path = tmp_path / "музыка.wav"
    narration_path = tmp_path / "рассказ.wav"
    output = tmp_path / "готовый фильм.mp4"
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
            "testsrc2=size=640x480:rate=1",
            "-frames:v",
            "1",
            "-update",
            "1",
            str(photo),
        ],
        check=True,
    )
    for path, frequency in ((music_path, 220), (narration_path, 660)):
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
                f"sine=frequency={frequency}:sample_rate=48000:duration=2",
                str(path),
            ],
            check=True,
        )
    clip = _clip(
        photo,
        media_type=MediaType.PHOTO,
        caption="Море: солнце, 100%",
        event_title="День семьи's",
        focus_x=0.7,
        focus_y=0.4,
    )
    settings = QuickMontageSettings(
        width=320,
        height=240,
        fps=24,
        render_device="cpu",
        transition="none",
        photo_motion="ken_burns",
        event_titles_enabled=True,
        scene_subtitles_enabled=True,
        credits_text="Сочи: команда's 100%",
        narration_enabled=True,
        music_volume_envelope=True,
    )
    plan = QuickMontagePlan(
        created_at=datetime.now(UTC),
        settings=settings,
        clips=[clip],
        total_duration_seconds=2,
        music_path=music_path,
        music_plan=MusicPlan(
            mode="manual",
            source_path=music_path,
            bpm=120,
            cue_sections=[
                MusicCueSection(
                    role="intro",
                    start_seconds=0,
                    end_seconds=2,
                    bpm=120,
                    intensity=0.5,
                )
            ],
        ),
        narration_path=narration_path,
    )

    QuickMontageRenderer().render(plan, output, tmp_path / "render")

    assert output.is_file()
    assert output.stat().st_size > 1000
    frame_hashes = []
    for position in (0.1, 1.7):
        decoded = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                str(position),
                "-i",
                str(output),
                "-frames:v",
                "1",
                "-f",
                "md5",
                "pipe:1",
            ],
            capture_output=True,
            check=True,
        )
        frame_hashes.append(decoded.stdout)
    assert frame_hashes[0] != frame_hashes[1]


def _clip(
    path: Path,
    *,
    media_type: MediaType = MediaType.VIDEO,
    **updates: object,
) -> MontageClip:
    payload: dict[str, object] = {
        "asset_id": uuid4(),
        "source_path": path,
        "relative_path": Path(path.name),
        "media_type": media_type,
        "duration_seconds": 2,
        "has_audio": False,
    }
    payload.update(updates)
    return MontageClip.model_validate(payload)


def _plan(
    clip: MontageClip,
    settings: QuickMontageSettings,
    *,
    narration_path: Path | None = None,
) -> QuickMontagePlan:
    return QuickMontagePlan(
        created_at=datetime.now(UTC),
        settings=settings,
        clips=[clip],
        total_duration_seconds=clip.duration_seconds,
        narration_path=narration_path,
    )
