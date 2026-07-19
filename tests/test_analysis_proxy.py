import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from threading import Barrier

import pytest

from travelmovieai.analysis.scenes import scene_cache_key
from travelmovieai.application.context import ProjectContext
from travelmovieai.core.config import Settings
from travelmovieai.core.exceptions import MontageError
from travelmovieai.domain.enums import MediaType
from travelmovieai.domain.models import MediaAsset, QuickMontageSettings, Scene
from travelmovieai.infrastructure.database import MediaAssetRepository
from travelmovieai.infrastructure.ffmpeg import FFprobeClient, ProbeResult
from travelmovieai.media.proxy import AnalysisMedia, AnalysisProxyManager, decide_analysis_proxy
from travelmovieai.pipeline.stages import frame_sampling, scene_detection
from travelmovieai.pipeline.stages.frame_sampling import (
    FrameSamplingStage,
    _prepare_analysis_assets,
)
from travelmovieai.pipeline.stages.scene_detection import SceneDetectionStage


class FakeProbe:
    def __init__(self, result: ProbeResult) -> None:
        self.result = result
        self.paths: list[Path] = []

    def probe(self, path: Path) -> ProbeResult:
        self.paths.append(path)
        return self.result


def test_proxy_decision_uses_ffprobe_when_scan_dimensions_are_missing(
    tmp_path: Path,
) -> None:
    asset = _video_asset(tmp_path / "исходник с пробелом.mp4", width=None, height=None)
    asset.path.write_bytes(b"source")
    probe = FakeProbe(ProbeResult(width=7680, height=4320, duration_seconds=10))
    manager = AnalysisProxyManager(tmp_path / "cache", probe=probe)

    decision = manager.decide(asset)

    assert decision.required is True
    assert decision.reason == "oversized"
    assert probe.paths == [asset.path]


def test_proxy_decision_skips_hd_and_non_video_sources(tmp_path: Path) -> None:
    video = _video_asset(tmp_path / "hd.mp4", width=1920, height=1080)
    photo = video.model_copy(update={"media_type": MediaType.PHOTO, "extension": ".jpg"})

    hd = decide_analysis_proxy(video, mode="auto", max_dimension=1920)
    image = decide_analysis_proxy(photo, mode="always", max_dimension=1920)

    assert (hd.required, hd.reason) == (False, "within-limit")
    assert (image.required, image.reason) == (False, "not-video")


def test_proxy_generation_is_atomic_unicode_safe_and_reuses_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset = _video_asset(tmp_path / "ролик с пробелом.mp4")
    probe = FakeProbe(ProbeResult(width=1920, height=1080, duration_seconds=10))
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        assert str(asset.path) in command
        Path(command[-1]).write_bytes(b"proxy-video")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("travelmovieai.media.proxy.subprocess.run", fake_run)
    manager = AnalysisProxyManager(tmp_path / "workspace" / "cache" / "proxies", probe=probe)

    first = manager.resolve(asset)
    second = manager.resolve(asset)
    proxy_stat = first.analysis_path.stat()
    os.utime(
        first.analysis_path,
        ns=(proxy_stat.st_atime_ns, proxy_stat.st_mtime_ns + 1_000_000_000),
    )
    changed_cache_file = manager.resolve(asset)

    assert first.proxied is True
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.analysis_path == first.analysis_path
    assert first.source_path == asset.path
    assert first.analysis_path.read_bytes() == b"proxy-video"
    assert changed_cache_file.cache_hit is False
    assert len(calls) == 2
    assert not list(manager.cache_dir.glob(".*.tmp.mp4"))


def test_proxy_cache_invalidates_when_source_or_proxy_settings_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset = _video_asset(tmp_path / "source.mp4")
    probe = FakeProbe(ProbeResult(width=1920, height=1080, duration_seconds=10))
    calls = 0

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        Path(command[-1]).write_bytes(f"proxy-{calls}".encode())
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("travelmovieai.media.proxy.subprocess.run", fake_run)
    manager = AnalysisProxyManager(tmp_path / "cache", probe=probe)

    first = manager.resolve(asset)
    changed = asset.model_copy(update={"size_bytes": 2, "modified_ns": 2})
    second = manager.resolve(changed)
    changed_settings = AnalysisProxyManager(
        manager.cache_dir,
        video_bitrate_mbps=8,
        probe=probe,
    ).resolve(changed)

    assert calls == 3
    assert first.analysis_path != second.analysis_path
    assert second.analysis_path != changed_settings.analysis_path
    assert first.cache_key != second.cache_key
    assert second.cache_key != changed_settings.cache_key


def test_proxy_failure_removes_partial_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset = _video_asset(tmp_path / "broken.mp4")

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        Path(command[-1]).write_bytes(b"partial")
        return subprocess.CompletedProcess(
            command,
            7,
            "",
            f"encoder failed for {asset.path} into {command[-1]}",
        )

    monkeypatch.setattr("travelmovieai.media.proxy.subprocess.run", fake_run)
    manager = AnalysisProxyManager(
        tmp_path / "cache",
        probe=FakeProbe(ProbeResult(width=1920, height=1080, duration_seconds=10)),
    )

    with pytest.raises(MontageError, match="Could not create an analysis proxy") as captured:
        manager.resolve(asset)

    assert str(tmp_path) not in str(captured.value)
    assert not list(manager.cache_dir.glob("*.mp4"))
    assert not list(manager.cache_dir.glob("*.json"))


def test_proxy_manifest_failure_removes_replaced_media(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset = _video_asset(tmp_path / "source.mp4")

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        Path(command[-1]).write_bytes(b"complete proxy")
        return subprocess.CompletedProcess(command, 0, "", "")

    def fail_manifest(*_: object, **__: object) -> None:
        raise OSError("disk failure")

    monkeypatch.setattr("travelmovieai.media.proxy.subprocess.run", fake_run)
    monkeypatch.setattr("travelmovieai.media.proxy.write_json_atomic", fail_manifest)
    manager = AnalysisProxyManager(
        tmp_path / "cache",
        probe=FakeProbe(ProbeResult(width=1920, height=1080, duration_seconds=10)),
    )

    with pytest.raises(MontageError, match="Could not finalize"):
        manager.resolve(asset)

    assert not list(manager.cache_dir.glob("*.mp4"))
    assert not list(manager.cache_dir.glob("*.json"))


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg and FFprobe are not installed",
)
def test_real_ffmpeg_proxy_downscales_tiny_4k_source_without_modifying_it(
    tmp_path: Path,
) -> None:
    source = tmp_path / "исходник 4k.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=3840x2160:r=24:d=0.25",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
    )
    source_bytes = source.read_bytes()
    source_stat = source.stat()
    asset = MediaAsset(
        path=source,
        relative_path=Path(source.name),
        media_type=MediaType.VIDEO,
        extension=".mp4",
        size_bytes=source_stat.st_size,
        modified_at=datetime.fromtimestamp(source_stat.st_mtime, UTC),
        modified_ns=source_stat.st_mtime_ns,
        duration_seconds=0.25,
        width=3840,
        height=2160,
        fps=24,
    )
    manager = AnalysisProxyManager(
        tmp_path / "workspace" / "cache" / "proxies",
        max_dimension=1280,
        video_bitrate_mbps=1,
    )

    resolved = manager.resolve(asset)
    cached = manager.resolve(asset)
    proxy_probe = FFprobeClient().probe(resolved.analysis_path)

    assert resolved.proxied is True
    assert cached.cache_hit is True
    assert (proxy_probe.width, proxy_probe.height) == (1280, 720)
    assert source.read_bytes() == source_bytes
    assert resolved.analysis_path != source


def test_frame_sampling_uses_proxy_transiently_and_preserves_source_asset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "исходники" / "4k clip.mp4"
    proxy_path = tmp_path / "workspace" / "cache" / "proxies" / "proxy.mp4"
    asset = _video_asset(source_path)
    scene = Scene(asset_id=asset.id, start_seconds=0, end_seconds=5)
    context = ProjectContext(
        input_path=source_path.parent,
        workspace=tmp_path / "workspace",
        settings=Settings(),
    )
    context.prepare()
    proxy_path.parent.mkdir(parents=True)
    proxy_path.write_bytes(b"proxy")
    repository = MediaAssetRepository(context.database_path)
    repository.initialize()
    repository.synchronize([asset], datetime.now(UTC))
    repository.synchronize_scenes([scene])
    received_paths: list[Path] = []

    class FakeProxyManager:
        def __init__(self, *_: object, **__: object) -> None:
            pass

        def cache_identity(self) -> str:
            return "c" * 64

        def resolve(self, source: MediaAsset) -> AnalysisMedia:
            return AnalysisMedia(
                asset_id=source.id,
                source_path=source.path,
                analysis_path=proxy_path,
                width=1920,
                height=1080,
                duration_seconds=source.duration_seconds,
                proxied=True,
                cache_key="a" * 64,
            )

    class FakeExtractor:
        def __init__(self, *_: object, **__: object) -> None:
            pass

        def extract(self, source_scene: Scene, source: MediaAsset, frames_dir: Path) -> Path:
            received_paths.append(source.path)
            output = frames_dir / f"{source_scene.id}.png"
            output.write_bytes(b"png")
            return output

    monkeypatch.setattr(frame_sampling, "AnalysisProxyManager", FakeProxyManager)
    monkeypatch.setattr(frame_sampling, "RepresentativeFrameExtractor", FakeExtractor)
    monkeypatch.setattr(
        frame_sampling,
        "detect_resource_profile",
        lambda *args, **kwargs: type(
            "Profile",
            (),
            {"nvenc": False, "frame_workers": 1},
        )(),
    )

    result = FrameSamplingStage().run(context)

    stored_asset = repository.list_assets()[0]
    assert received_paths == [proxy_path]
    assert stored_asset.path == source_path
    assert "proxies=1 generated/0 cached" in result.message


def test_scene_cache_invalidates_when_proxy_configuration_changes(tmp_path: Path) -> None:
    source_path = tmp_path / "source" / "clip.mp4"
    asset = _video_asset(source_path, width=1280, height=720)
    context = ProjectContext(
        input_path=source_path.parent,
        workspace=tmp_path / "workspace",
        settings=Settings(),
        montage_settings=QuickMontageSettings(),
    )
    context.prepare()
    repository = MediaAssetRepository(context.database_path)
    repository.initialize()
    repository.synchronize([asset], datetime.now(UTC))
    calls = 0

    class FakeProxyManager:
        def __init__(self, identity: str) -> None:
            self.identity = identity

        def cache_identity(self) -> str:
            return self.identity

        def resolve(self, source: MediaAsset) -> AnalysisMedia:
            return AnalysisMedia(
                asset_id=source.id,
                source_path=source.path,
                analysis_path=source.path,
                width=source.width,
                height=source.height,
                duration_seconds=source.duration_seconds,
            )

    class FakeDetector:
        def detect(
            self,
            detected_asset: MediaAsset,
            settings: QuickMontageSettings,
        ) -> tuple[list[Scene], bool]:
            nonlocal calls
            calls += 1
            return [
                Scene(
                    asset_id=detected_asset.id,
                    start_seconds=0,
                    end_seconds=5,
                    metadata={"cache_key": scene_cache_key(detected_asset, settings)},
                )
            ], False

    first = SceneDetectionStage(
        detector=FakeDetector(),
        proxy_manager=FakeProxyManager("a" * 64),
    ).run(context)
    cached = SceneDetectionStage(
        detector=FakeDetector(),
        proxy_manager=FakeProxyManager("a" * 64),
    ).run(context)
    changed = SceneDetectionStage(
        detector=FakeDetector(),
        proxy_manager=FakeProxyManager("b" * 64),
    ).run(context)

    assert first.skipped is False
    assert cached.skipped is True
    assert changed.skipped is False
    assert calls == 2


def test_scene_detection_uses_proxy_before_decoding_original_4k_media(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "исходники" / "4k clip.mp4"
    proxy_path = tmp_path / "workspace" / "cache" / "proxies" / "proxy.mp4"
    asset = _video_asset(source_path)
    context = ProjectContext(
        input_path=source_path.parent,
        workspace=tmp_path / "workspace",
        settings=Settings(),
        montage_settings=QuickMontageSettings(),
    )
    context.prepare()
    proxy_path.parent.mkdir(parents=True, exist_ok=True)
    proxy_path.write_bytes(b"proxy")
    repository = MediaAssetRepository(context.database_path)
    repository.initialize()
    repository.synchronize([asset], datetime.now(UTC))
    received_paths: list[Path] = []

    class FakeProxyManager:
        def __init__(self, *_: object, **__: object) -> None:
            pass

        def cache_identity(self) -> str:
            return "d" * 64

        def resolve(self, source: MediaAsset) -> AnalysisMedia:
            return AnalysisMedia(
                asset_id=source.id,
                source_path=source.path,
                analysis_path=proxy_path,
                width=1920,
                height=1080,
                duration_seconds=source.duration_seconds,
                proxied=True,
                cache_hit=False,
                cache_key="b" * 64,
            )

    class FakeDetector:
        def detect(
            self,
            detected_asset: MediaAsset,
            settings: QuickMontageSettings,
        ) -> tuple[list[Scene], bool]:
            del settings
            received_paths.append(detected_asset.path)
            return [Scene(asset_id=detected_asset.id, start_seconds=0, end_seconds=5)], False

    monkeypatch.setattr(scene_detection, "AnalysisProxyManager", FakeProxyManager)
    result = SceneDetectionStage(detector=FakeDetector()).run(context)

    assert received_paths == [proxy_path]
    assert repository.list_assets()[0].path == source_path
    assert "proxies=1 generated/0 cached" in result.message


def test_parallel_proxy_preparation_does_not_start_queued_work_after_cancel(
    tmp_path: Path,
) -> None:
    barrier = Barrier(2)
    calls: list[Path] = []
    assets = [
        _video_asset(tmp_path / f"clip-{index}.mp4", width=1280, height=720) for index in range(5)
    ]

    class BlockingManager(AnalysisProxyManager):
        def __init__(self) -> None:
            pass

        def resolve(self, asset: MediaAsset) -> AnalysisMedia:
            calls.append(asset.relative_path)
            barrier.wait(timeout=5)
            return AnalysisMedia(
                asset_id=asset.id,
                source_path=asset.path,
                analysis_path=asset.path,
                width=asset.width,
                height=asset.height,
                duration_seconds=asset.duration_seconds,
            )

    def cancel_after_first(current: int, total: int, message: str) -> None:
        del total, message
        if current >= 1:
            raise MontageError("cancel proxy queue")

    with pytest.raises(MontageError, match="cancel proxy queue"):
        _prepare_analysis_assets(
            {asset.id: asset for asset in assets},
            BlockingManager(),
            workers=2,
            progress=cancel_after_first,
        )

    assert len(calls) == 2


def _video_asset(
    path: Path,
    *,
    width: int | None = 3840,
    height: int | None = 2160,
) -> MediaAsset:
    return MediaAsset(
        path=path,
        relative_path=Path(path.name),
        media_type=MediaType.VIDEO,
        extension=".mp4",
        size_bytes=1,
        modified_at=datetime(2026, 1, 1, tzinfo=UTC),
        modified_ns=1,
        duration_seconds=10,
        width=width,
        height=height,
        fps=30,
    )
