import hashlib
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from threading import Barrier
from uuid import uuid4

import pytest
from PIL import Image

from travelmovieai.analysis.scenes import (
    CONTACT_SHEET_SCHEMA_VERSION,
    RepresentativeFrameExtractor,
    contact_sheet_file_valid,
    scene_cache_key,
)
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


def test_scene_cache_key_normalizes_equivalent_numeric_settings(tmp_path: Path) -> None:
    asset = MediaAsset(
        path=tmp_path / "clip.mp4",
        relative_path=Path("clip.mp4"),
        media_type=MediaType.VIDEO,
        extension=".mp4",
        size_bytes=1,
        modified_at=datetime.now(UTC),
        modified_ns=1,
    )
    defaults = QuickMontageSettings()
    explicit_floats = QuickMontageSettings(
        scene_threshold=float(defaults.scene_threshold),
        min_scene_duration_seconds=float(defaults.min_scene_duration_seconds),
        max_scene_duration_seconds=float(defaults.max_scene_duration_seconds),
    )

    assert scene_cache_key(asset, defaults) == scene_cache_key(asset, explicit_floats)


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


def test_frame_extractor_does_not_reuse_untracked_nonempty_png(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset = MediaAsset(
        path=tmp_path / "clip.mp4",
        relative_path=Path("clip.mp4"),
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
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    cached_path = frames_dir / f"{scene.id}-contact-v4-3.png"
    Image.new("RGB", (1440, 270), (255, 0, 0)).save(cached_path)
    calls = 0

    def fake_run(
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        del kwargs
        nonlocal calls
        calls += 1
        Image.new("RGB", (1440, 270), (0, 255, 0)).save(Path(command[-1]))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("travelmovieai.analysis.scenes.subprocess.run", fake_run)

    output = RepresentativeFrameExtractor().extract(scene, asset, frames_dir)

    assert calls == 1
    with Image.open(output) as regenerated:
        assert regenerated.getpixel((0, 0)) == (0, 255, 0)


def test_cached_contact_sheet_rejects_wrong_geometry_with_matching_digest(
    tmp_path: Path,
) -> None:
    contact_sheet = tmp_path / "wrong-geometry.png"
    Image.new("RGB", (480, 270), (0, 0, 0)).save(contact_sheet)
    metadata: dict[str, object] = {
        "schema_version": CONTACT_SHEET_SCHEMA_VERSION,
        "sample_count": 3,
        "sample_positions": [0.12, 0.5, 0.88],
        "sample_timestamps_seconds": [0.12, 0.5, 0.88],
        "columns": 3,
        "rows": 1,
        "content_sha256": hashlib.sha256(contact_sheet.read_bytes()).hexdigest(),
    }

    assert not contact_sheet_file_valid(
        contact_sheet,
        metadata,
        expected_sample_count=3,
    )


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

    extractor = RepresentativeFrameExtractor(frame_sample_count=9)
    output = extractor.extract(
        scene,
        asset,
        tmp_path / "frames",
    )
    metadata = extractor.sampling_metadata(scene, asset, output)

    assert output.name.endswith("-contact-v4-9.png")
    with Image.open(output) as image:
        assert image.size == (1440, 810)
    assert metadata["sample_count"] == 9
    positions = metadata["sample_positions"]
    timestamps = metadata["sample_timestamps_seconds"]
    assert isinstance(positions, list)
    assert isinstance(timestamps, list)
    assert positions == sorted(positions)
    assert positions == pytest.approx([timestamp / 3 for timestamp in timestamps])
    assert len(str(metadata["content_sha256"])) == 64
