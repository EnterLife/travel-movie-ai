import json
import math
import shutil
import struct
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from travelmovieai.application.service import TravelMovieService
from travelmovieai.core.config import Settings
from travelmovieai.core.exceptions import MontageError
from travelmovieai.domain.enums import MediaType
from travelmovieai.domain.models import (
    MediaAsset,
    MontageClip,
    QuickMontagePlan,
    QuickMontageSettings,
    SceneUnderstanding,
)
from travelmovieai.editing.renderer import (
    QuickMontageRenderer,
    _build_filter_graph,
    _transition_duration,
)
from travelmovieai.editing.timeline import build_quick_montage_plan


class FakeVisionProvider:
    name = "fake-vision"
    model = "fake-model"

    def __init__(self) -> None:
        self.calls = 0

    def analyze(self, image_path: Path, style: object) -> SceneUnderstanding:
        self.calls += 1
        return SceneUnderstanding(
            caption=f"Travel scene {self.calls}",
            detailed_description=f"Detailed travel scene {self.calls}.",
            location_type="city" if self.calls == 1 else "beach",
            activity="walking",
            emotion="joyful",
            people_count=2,
            people_groups=["group"],
            landmarks=[],
            vision_score=90 if self.calls == 1 else 70,
            score_factors={
                "uniqueness": 80,
                "people": 70,
                "emotion": 80,
                "visual_quality": 50,
                "landmark": 20,
                "unusual_event": 30,
            },
            story_relevance="Useful travel moment.",
            tags=["travel", f"scene-{self.calls}"],
        )


def test_quick_montage_plan_orders_assets_and_respects_duration(tmp_path: Path) -> None:
    early = _asset(
        tmp_path / "early.mp4",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        duration=20,
    )
    photo = _asset(
        tmp_path / "photo.jpg",
        created_at=datetime(2026, 1, 2, tzinfo=UTC),
        media_type=MediaType.PHOTO,
    )
    late = _asset(
        tmp_path / "late.mp4",
        created_at=datetime(2026, 1, 3, tzinfo=UTC),
        duration=20,
    )

    plan = build_quick_montage_plan(
        [late, photo, early],
        QuickMontageSettings(
            target_duration_seconds=10,
            max_video_clip_seconds=5,
            photo_duration_seconds=3,
        ),
    )

    assert [clip.relative_path.name for clip in plan.clips] == [
        "early.mp4",
        "photo.jpg",
        "late.mp4",
    ]
    assert plan.total_duration_seconds == 10
    assert plan.clips[0].source_start_seconds == 7.5
    assert plan.clips[-1].duration_seconds == 2


def test_renderer_uses_cut_only_graph_even_when_transition_is_requested(tmp_path: Path) -> None:
    settings = QuickMontageSettings(transition="fade", transition_duration_seconds=0.4)
    plan = QuickMontagePlan(
        created_at=datetime.now(UTC),
        settings=settings,
        clips=[
            MontageClip(
                asset_id=uuid4(),
                source_path=tmp_path / "first.mp4",
                relative_path=Path("first.mp4"),
                media_type=MediaType.VIDEO,
                duration_seconds=3,
            ),
            MontageClip(
                asset_id=uuid4(),
                source_path=tmp_path / "second.mp4",
                relative_path=Path("second.mp4"),
                media_type=MediaType.VIDEO,
                duration_seconds=3,
                transition="slideright",
            ),
        ],
        total_duration_seconds=5.6,
    )

    transition_duration = _transition_duration(plan)
    graph = _build_filter_graph(plan, transition_duration=transition_duration)

    assert transition_duration == 0
    assert "xfade=" not in graph
    assert "acrossfade=" not in graph
    assert "concat=n=2:v=1:a=0" in graph


def test_renderer_ignores_soft_transition_preset(tmp_path: Path) -> None:
    settings = QuickMontageSettings(transition="soft", transition_duration_seconds=0.35)
    plan = QuickMontagePlan(
        created_at=datetime.now(UTC),
        settings=settings,
        clips=[
            MontageClip(
                asset_id=uuid4(),
                source_path=tmp_path / "first.mp4",
                relative_path=Path("first.mp4"),
                media_type=MediaType.VIDEO,
                duration_seconds=3,
            ),
            MontageClip(
                asset_id=uuid4(),
                source_path=tmp_path / "second.mp4",
                relative_path=Path("second.mp4"),
                media_type=MediaType.VIDEO,
                duration_seconds=3,
            ),
        ],
        total_duration_seconds=5.65,
    )

    graph = _build_filter_graph(plan, transition_duration=_transition_duration(plan))

    assert "xfade=" not in graph
    assert "concat=n=2:v=1:a=0" in graph


def test_renderer_uses_preroll_and_trim_for_video_segments(tmp_path: Path) -> None:
    captured: list[list[str]] = []

    class CapturingRenderer(QuickMontageRenderer):
        def _run(self, command: list[str], message: str) -> None:
            captured.append(command)

    settings = QuickMontageSettings(transition="none", music_enabled=False)
    clip = MontageClip(
        asset_id=uuid4(),
        source_path=tmp_path / "source.mp4",
        relative_path=Path("source.mp4"),
        media_type=MediaType.VIDEO,
        source_start_seconds=5.2,
        duration_seconds=2,
        has_audio=True,
    )
    plan = QuickMontagePlan(
        created_at=datetime.now(UTC),
        settings=settings,
        clips=[clip],
        total_duration_seconds=2,
    )

    CapturingRenderer()._render_segment(clip, plan, tmp_path / "segment.mp4")

    command = captured[0]
    filter_graph = command[command.index("-filter_complex") + 1]
    assert command[command.index("-ss") + 1] == "4.200"
    assert command[command.index("-t") + 1] == "3.250"
    assert "trim=start=1.000:duration=2.000" in filter_graph
    assert "atrim=start=1.000:duration=2.000" in filter_graph


def test_renderer_reports_ffmpeg_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[float] = []

    def fake_run(*args: object, **kwargs: object) -> object:
        timeout = kwargs["timeout"]
        assert isinstance(timeout, int | float)
        calls.append(float(timeout))
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=timeout)

    monkeypatch.setattr("travelmovieai.editing.renderer.subprocess.run", fake_run)

    with pytest.raises(MontageError, match="timed out after 0.25s"):
        QuickMontageRenderer(timeout_seconds=0.25)._run(["ffmpeg"], "Could not render")

    assert calls == [0.25]


def test_renderer_rejects_missing_soundtrack_before_ffmpeg(tmp_path: Path) -> None:
    missing_music = tmp_path / "missing.wav"
    plan = QuickMontagePlan(
        created_at=datetime.now(UTC),
        settings=QuickMontageSettings(),
        clips=[
            MontageClip(
                asset_id=uuid4(),
                source_path=tmp_path / "clip.mp4",
                relative_path=Path("clip.mp4"),
                media_type=MediaType.VIDEO,
                duration_seconds=2,
            )
        ],
        total_duration_seconds=2,
        music_path=missing_music,
    )

    with pytest.raises(MontageError, match="Soundtrack file does not exist"):
        QuickMontageRenderer().render(plan, tmp_path / "out.mp4", tmp_path)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is not installed")
def test_service_creates_playable_quick_montage(tmp_path: Path) -> None:
    media = tmp_path / "Моя поездка"
    media.mkdir()
    video = media / "clip with audio.mp4"
    photo = media / "photo.jpg"
    _generate_video(video)
    _generate_photo(photo)
    workspace = tmp_path / "workspace"

    result = TravelMovieService(Settings()).create_quick_montage(
        input_path=media,
        workspace=workspace,
        settings=QuickMontageSettings(
            target_duration_seconds=5,
            max_video_clip_seconds=1,
            photo_duration_seconds=1,
            width=320,
            height=240,
            fps=24,
            music_engine="procedural",
        ),
    )

    assert result.output_path.is_file()
    assert result.output_path.stat().st_size > 0
    assert result.clip_count == 2
    assert result.timeline_path.is_file()
    quality_report = json.loads(
        (workspace / "artifacts" / "montage_quality_report.json").read_text(encoding="utf-8")
    )
    assert quality_report["clip_count"] == result.clip_count
    assert quality_report["planned_duration_seconds"] == pytest.approx(result.duration_seconds)
    assert quality_report["rendered_path"] == str(result.output_path)
    assert quality_report["rendered_has_video"] is True
    assert quality_report["rendered_has_audio"] is True
    assert quality_report["rendered_duration_seconds"] == pytest.approx(
        result.duration_seconds,
        abs=0.2,
    )
    assert set(quality_report["rendered_audio_rms"]) == {"start", "middle", "end"}
    assert quality_report["rendered_audio_rms"]["middle"] > 10
    assert set(quality_report["rendered_video_luma"]) == {"start", "middle", "end"}
    timeline = json.loads(result.timeline_path.read_text(encoding="utf-8"))
    music_plan = timeline["music_plan"]
    assert music_plan["generated"] is True
    assert music_plan["duration_seconds"] == pytest.approx(result.duration_seconds)
    assert music_plan["arrangement_version"] == "adaptive-lounge-v5"
    assert music_plan["accents"][0]["kind"] == "intro"
    assert music_plan["accents"][-1]["kind"] == "finale"

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(result.output_path),
        ],
        capture_output=True,
        check=True,
        text=True,
    )
    payload = json.loads(probe.stdout)
    stream_types = {stream["codec_type"] for stream in payload["streams"]}
    assert stream_types == {"video", "audio"}
    assert float(payload["format"]["duration"]) == pytest.approx(
        result.duration_seconds,
        abs=0.15,
    )


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is not installed")
def test_service_creates_cached_semantic_montage_with_music(tmp_path: Path) -> None:
    media = tmp_path / "AI поездка"
    media.mkdir()
    _generate_video(media / "city.mp4")
    _generate_photo(media / "beach.jpg")
    _generate_music(media / "cinematic theme.wav")
    workspace = tmp_path / "workspace"
    provider = FakeVisionProvider()
    service = TravelMovieService(
        Settings(),
        vision_provider_factory=lambda _: provider,
    )
    settings = QuickMontageSettings(
        target_duration_seconds=5,
        max_video_clip_seconds=1,
        photo_duration_seconds=1,
        width=320,
        height=240,
        fps=24,
        semantic_analysis=True,
        music_mode="library",
        transition="dissolve",
        transition_duration_seconds=0.25,
    )

    first = service.create_quick_montage(
        input_path=media,
        workspace=workspace,
        settings=settings,
    )
    calls_after_first_run = provider.calls
    second = service.create_quick_montage(
        input_path=media,
        workspace=workspace,
        settings=settings,
    )

    assert first.selection_mode == "semantic"
    assert first.output_path.is_file()
    assert calls_after_first_run >= 2
    assert provider.calls == calls_after_first_run
    assert second.output_path.is_file()
    vision = json.loads(
        (workspace / "artifacts" / "vision_analysis.json").read_text(encoding="utf-8")
    )
    timeline = json.loads(first.timeline_path.read_text(encoding="utf-8"))
    assert len(vision["scenes"]) >= 2
    assert (workspace / "artifacts" / "events.json").is_file()
    assert (workspace / "artifacts" / "scene_descriptions.json").is_file()
    quality_report = json.loads(
        (workspace / "artifacts" / "montage_quality_report.json").read_text(encoding="utf-8")
    )
    assert quality_report["selected_scene_count"] == first.clip_count
    assert quality_report["music_mode"] == "library"
    assert quality_report["rendered_has_video"] is True
    assert quality_report["rendered_has_audio"] is True
    assert timeline["selection_mode"] == "semantic"
    assert timeline["music_path"].endswith("cinematic theme.wav")
    assert all(clip["semantic_score"] is not None for clip in timeline["clips"])


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is not installed")
@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="FFprobe is not installed")
def test_renderer_keeps_short_music_audible_until_the_end(tmp_path: Path) -> None:
    photo = tmp_path / "quiet-photo.jpg"
    music = tmp_path / "short-theme.wav"
    output = tmp_path / "movie.mp4"
    _generate_photo(photo)
    _generate_music(music, duration=0.25)
    settings = QuickMontageSettings(
        width=320,
        height=240,
        fps=24,
        transition="none",
        music_volume=0.5,
    )
    clip = MontageClip(
        asset_id=uuid4(),
        source_path=photo,
        relative_path=Path("quiet-photo.jpg"),
        media_type=MediaType.PHOTO,
        duration_seconds=2,
        has_audio=False,
    )
    plan = QuickMontagePlan(
        created_at=datetime.now(UTC),
        settings=settings,
        clips=[clip],
        total_duration_seconds=2,
        music_path=music,
    )

    QuickMontageRenderer().render(plan, output, tmp_path / "render")

    assert _audio_rms(output, start_seconds=1.45, duration_seconds=0.35) > 100


def _asset(
    path: Path,
    *,
    created_at: datetime,
    duration: float | None = None,
    media_type: MediaType = MediaType.VIDEO,
) -> MediaAsset:
    return MediaAsset(
        path=path,
        relative_path=Path(path.name),
        media_type=media_type,
        extension=path.suffix,
        size_bytes=1,
        modified_at=created_at,
        modified_ns=1,
        created_at=created_at,
        duration_seconds=duration,
    )


def _generate_video(path: Path) -> None:
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
            "testsrc2=size=320x240:rate=24:duration=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=2",
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


def _generate_photo(path: Path) -> None:
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
            "color=c=orange:s=320x240",
            "-frames:v",
            "1",
            "-update",
            "1",
            str(path),
        ],
        check=True,
    )


def _generate_music(path: Path, duration: float = 4) -> None:
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
            f"sine=frequency=220:sample_rate=48000:duration={duration}",
            str(path),
        ],
        check=True,
    )


def _audio_rms(path: Path, *, start_seconds: float, duration_seconds: float) -> float:
    completed = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start_seconds:.3f}",
            "-t",
            f"{duration_seconds:.3f}",
            "-i",
            str(path),
            "-vn",
            "-f",
            "s16le",
            "-ac",
            "1",
            "-ar",
            "8000",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
    )
    sample_count = len(completed.stdout) // 2
    if sample_count == 0:
        return 0.0
    samples = struct.unpack(f"<{sample_count}h", completed.stdout[: sample_count * 2])
    return math.sqrt(sum(sample * sample for sample in samples) / sample_count)
