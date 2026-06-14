import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from travelmovieai.application.service import TravelMovieService
from travelmovieai.core.config import Settings
from travelmovieai.domain.enums import MediaType
from travelmovieai.domain.models import (
    MediaAsset,
    QuickMontageSettings,
    SceneUnderstanding,
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
    assert plan.clips[-1].duration_seconds == 3


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
    timeline = json.loads(result.timeline_path.read_text(encoding="utf-8"))
    music_plan = timeline["music_plan"]
    assert music_plan["generated"] is True
    assert music_plan["duration_seconds"] == pytest.approx(result.duration_seconds)
    assert music_plan["arrangement_version"] == "adaptive-lounge-v2"
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
    assert timeline["selection_mode"] == "semantic"
    assert timeline["music_path"].endswith("cinematic theme.wav")
    assert all(clip["semantic_score"] is not None for clip in timeline["clips"])


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


def _generate_music(path: Path) -> None:
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
            "sine=frequency=220:sample_rate=48000:duration=4",
            str(path),
        ],
        check=True,
    )
