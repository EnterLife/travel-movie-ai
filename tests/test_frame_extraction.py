import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest
from PIL import Image

from travelmovieai.analysis.scenes import RepresentativeFrameExtractor
from travelmovieai.domain.enums import MediaType
from travelmovieai.domain.models import MediaAsset, Scene


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is not installed")
def test_contact_sheet_supports_limited_range_yuv(tmp_path: Path) -> None:
    video = tmp_path / "limited-range.mp4"
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
            "testsrc2=size=640x360:rate=24:duration=2",
            "-vf",
            "format=yuv420p,setrange=limited",
            "-color_range",
            "tv",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
    )
    asset = MediaAsset(
        path=video,
        relative_path=Path("DJI sample.mp4"),
        media_type=MediaType.VIDEO,
        extension=".mp4",
        size_bytes=video.stat().st_size,
        modified_at=datetime.now(UTC),
        modified_ns=video.stat().st_mtime_ns,
        duration_seconds=2,
        width=640,
        height=360,
        fps=24,
    )
    scene = Scene(asset_id=asset.id, start_seconds=0, end_seconds=2)

    extractor = RepresentativeFrameExtractor(use_cuda_decode=True)
    output = extractor.extract(
        scene,
        asset,
        tmp_path / "frames",
    )

    assert output.suffix == ".png"
    assert "NVDEC=" in extractor.backend_summary
    assert "CPU fallback=" in extractor.backend_summary
    with Image.open(output) as image:
        assert image.size == (1440, 270)
        assert image.mode == "RGB"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is not installed")
def test_contact_sheet_clamps_samples_to_video_stream_duration(tmp_path: Path) -> None:
    video = tmp_path / "short-video-long-container.mp4"
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
            "testsrc2=size=640x360:rate=24:duration=2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
    )
    asset = MediaAsset(
        path=video,
        relative_path=Path("DJI final scene.mp4"),
        media_type=MediaType.VIDEO,
        extension=".mp4",
        size_bytes=video.stat().st_size,
        modified_at=datetime.now(UTC),
        modified_ns=video.stat().st_mtime_ns,
        duration_seconds=2.5,
        width=640,
        height=360,
        fps=24,
        probe_metadata={"video_duration_seconds": 2.0},
    )
    scene = Scene(asset_id=asset.id, start_seconds=1.5, end_seconds=2.5)

    output = RepresentativeFrameExtractor().extract(
        scene,
        asset,
        tmp_path / "frames",
    )

    assert output.is_file()
    with Image.open(output) as image:
        assert image.size == (1440, 270)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is not installed")
def test_deep_contact_sheet_extracts_nine_frames(tmp_path: Path) -> None:
    video = tmp_path / "deep-contact-sheet.mp4"
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
            "testsrc2=size=640x360:rate=24:duration=3",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
    )
    asset = MediaAsset(
        path=video,
        relative_path=Path("deep sample.mp4"),
        media_type=MediaType.VIDEO,
        extension=".mp4",
        size_bytes=video.stat().st_size,
        modified_at=datetime.now(UTC),
        modified_ns=video.stat().st_mtime_ns,
        duration_seconds=3,
        width=640,
        height=360,
        fps=24,
    )
    scene = Scene(asset_id=asset.id, start_seconds=0, end_seconds=3)

    output = RepresentativeFrameExtractor(frame_sample_count=9).extract(
        scene,
        asset,
        tmp_path / "frames",
    )

    assert output.name.endswith("-contact-v4-9.png")
    with Image.open(output) as image:
        assert image.size == (1440, 810)
