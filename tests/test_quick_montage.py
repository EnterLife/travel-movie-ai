import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from travelmovieai.application.service import TravelMovieService
from travelmovieai.core.config import Settings
from travelmovieai.domain.enums import MediaType
from travelmovieai.domain.models import MediaAsset, QuickMontageSettings
from travelmovieai.editing.timeline import build_quick_montage_plan


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
        ),
    )

    assert result.output_path.is_file()
    assert result.output_path.stat().st_size > 0
    assert result.clip_count == 2
    assert result.timeline_path.is_file()

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
    assert float(payload["format"]["duration"]) == pytest.approx(2, abs=0.15)


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
