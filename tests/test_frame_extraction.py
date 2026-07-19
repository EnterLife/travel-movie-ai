import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from threading import Barrier
from uuid import uuid4

import pytest
from PIL import Image

from travelmovieai.analysis.scenes import RepresentativeFrameExtractor, scene_cache_key
from travelmovieai.core.exceptions import MontageError
from travelmovieai.domain.enums import MediaType
from travelmovieai.domain.models import MediaAsset, QuickMontageSettings, Scene
from travelmovieai.pipeline.stages.frame_sampling import _extract_frames


def test_photo_scene_cache_invalidates_when_duration_changes(tmp_path: Path) -> None:
    asset = MediaAsset(
        path=tmp_path / "photo.jpg",
        relative_path=Path("photo.jpg"),
        media_type=MediaType.PHOTO,
        extension=".jpg",
        size_bytes=1,
        modified_at=datetime.now(UTC),
        modified_ns=1,
    )

    short_key = scene_cache_key(asset, QuickMontageSettings(photo_duration_seconds=2))
    long_key = scene_cache_key(asset, QuickMontageSettings(photo_duration_seconds=5))

    assert short_key != long_key
    video = asset.model_copy(update={"media_type": MediaType.VIDEO, "extension": ".mp4"})
    assert scene_cache_key(
        video,
        QuickMontageSettings(photo_duration_seconds=2),
    ) == scene_cache_key(video, QuickMontageSettings(photo_duration_seconds=5))


def test_parallel_frame_extraction_does_not_start_queued_work_after_cancel(
    tmp_path: Path,
) -> None:
    barrier = Barrier(2)
    calls: list[Path] = []
    assets = []
    scenes = []
    for index in range(5):
        asset = MediaAsset(
            id=uuid4(),
            path=tmp_path / f"clip-{index}.mp4",
            relative_path=Path(f"clip-{index}.mp4"),
            media_type=MediaType.VIDEO,
            extension=".mp4",
            size_bytes=1,
            modified_at=datetime.now(UTC),
            modified_ns=index + 1,
            duration_seconds=2,
        )
        assets.append(asset)
        scenes.append(Scene(asset_id=asset.id, start_seconds=0, end_seconds=2))

    class BlockingExtractor:
        def extract(self, scene: Scene, asset: MediaAsset, frames_dir: Path) -> Path:
            del scene
            calls.append(asset.relative_path)
            barrier.wait(timeout=5)
            output = frames_dir / f"{asset.id}.png"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"png")
            return output

    def cancel_after_first(current: int, total: int, message: str) -> None:
        del total, message
        if current >= 1:
            raise MontageError("cancel frame queue")

    with pytest.raises(MontageError, match="cancel frame queue"):
        _extract_frames(
            scenes,
            {asset.id: asset for asset in assets},
            BlockingExtractor(),
            tmp_path / "frames",
            workers=2,
            progress=cancel_after_first,
        )

    assert len(calls) == 2


def test_frame_extraction_times_out_hung_ffmpeg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[float] = []

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        timeout = kwargs["timeout"]
        assert isinstance(timeout, (int, float))
        calls.append(float(timeout))
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=timeout)

    monkeypatch.setattr("travelmovieai.analysis.scenes.subprocess.run", fake_run)
    asset = MediaAsset(
        path=tmp_path / "clip.mp4",
        relative_path=Path("DJI sample.mp4"),
        media_type=MediaType.VIDEO,
        extension=".mp4",
        size_bytes=1,
        modified_at=datetime.now(UTC),
        modified_ns=1,
        duration_seconds=3,
        width=640,
        height=360,
        fps=24,
        probe_metadata={"video_duration_seconds": 3.0},
    )
    scene = Scene(asset_id=asset.id, start_seconds=0, end_seconds=3)

    extractor = RepresentativeFrameExtractor(
        use_cuda_decode=True,
        timeout_seconds=0.25,
    )
    with pytest.raises(MontageError, match="timed out after 0.25s.*DJI sample.mp4"):
        extractor.extract(scene, asset, tmp_path / "frames")

    assert calls == [0.25, 0.25]


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
