from datetime import UTC, datetime
from pathlib import Path
from threading import Barrier, Lock
from types import SimpleNamespace

import pytest

from travelmovieai.analysis import scenes as scene_analysis
from travelmovieai.application.context import ProjectContext
from travelmovieai.core.config import Settings
from travelmovieai.core.exceptions import MontageError
from travelmovieai.domain.enums import MediaType, StageStatus
from travelmovieai.domain.models import (
    MediaAsset,
    QuickMontageSettings,
    Scene,
    SceneDetectionReport,
)
from travelmovieai.infrastructure.database import MediaAssetRepository
from travelmovieai.media.proxy import AnalysisMedia
from travelmovieai.pipeline.stages.scene_detection import SceneDetectionStage


class _DirectProxyManager:
    def cache_identity(self) -> str:
        return "a" * 64

    def resolve(self, asset: MediaAsset) -> AnalysisMedia:
        return AnalysisMedia(
            asset_id=asset.id,
            source_path=asset.path,
            analysis_path=asset.path,
            width=asset.width,
            height=asset.height,
            duration_seconds=asset.duration_seconds,
        )


def test_no_cut_clip_is_detected_without_uniform_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset = _video_asset(tmp_path / "continuous.mp4")
    context = _context_with_assets(tmp_path, [asset])
    captured: dict[str, object] = {}

    class Timecode:
        def __init__(self, seconds: float) -> None:
            self.seconds = seconds

        def get_seconds(self) -> float:
            return self.seconds

    def detect(
        path: str,
        detector: object,
        *,
        show_progress: bool,
        start_in_scene: bool,
    ) -> list[tuple[Timecode, Timecode]]:
        captured.update(
            path=path,
            detector=detector,
            show_progress=show_progress,
            start_in_scene=start_in_scene,
        )
        return [(Timecode(0), Timecode(10))] if start_in_scene else []

    fake_scenedetect = SimpleNamespace(
        ContentDetector=lambda **kwargs: kwargs,
        detect=detect,
    )
    monkeypatch.setattr(
        scene_analysis.importlib,
        "import_module",
        lambda name: fake_scenedetect if name == "scenedetect" else None,
    )

    result = SceneDetectionStage(
        proxy_manager=_DirectProxyManager(),
        workers=1,
    ).run(context)
    report = SceneDetectionReport.model_validate_json(
        (context.artifacts_dir / "scenes.json").read_text(encoding="utf-8")
    )

    assert captured["path"] == str(asset.path)
    assert captured["show_progress"] is False
    assert captured["start_in_scene"] is True
    assert result.status is StageStatus.COMPLETED
    assert result.execution.fallback_count == 0
    assert result.execution.fallback_provider is None
    assert report.fallback_count == 0
    assert [scene.metadata["detector"] for scene in report.scenes] == ["pyscenedetect"]


def test_scene_detection_is_parallel_deterministic_and_cacheable(tmp_path: Path) -> None:
    assets = [_video_asset(tmp_path / f"clip-{index}.mp4") for index in (3, 1, 2, 0)]
    context = _context_with_assets(tmp_path, assets)
    barrier = Barrier(2)
    calls: list[Path] = []
    lock = Lock()

    class ParallelDetector:
        def detect(
            self,
            asset: MediaAsset,
            settings: QuickMontageSettings,
        ) -> tuple[list[Scene], bool]:
            del settings
            with lock:
                calls.append(asset.relative_path)
            barrier.wait(timeout=5)
            return [Scene(asset_id=asset.id, start_seconds=0, end_seconds=5)], False

    first = SceneDetectionStage(
        detector=ParallelDetector(),
        proxy_manager=_DirectProxyManager(),
        workers=2,
    ).run(context)

    assert first.status is StageStatus.COMPLETED
    assert set(calls) == {asset.relative_path for asset in assets}
    detected = SceneDetectionReport.model_validate_json(
        (context.artifacts_dir / "scenes.json").read_text(encoding="utf-8")
    ).scenes
    by_id = {asset.id: asset.relative_path.as_posix() for asset in assets}
    assert [by_id[scene.asset_id] for scene in detected] == sorted(by_id.values())
    assert len(list((context.artifacts_dir / "scene_detection_shards").glob("*.json"))) == 4

    class FailIfCalled:
        def detect(
            self,
            asset: MediaAsset,
            settings: QuickMontageSettings,
        ) -> tuple[list[Scene], bool]:
            del asset, settings
            raise AssertionError("valid cached scenes must bypass the detector")

    cached = SceneDetectionStage(
        detector=FailIfCalled(),
        proxy_manager=_DirectProxyManager(),
        workers=2,
    ).run(context)
    assert cached.status is StageStatus.CACHED
    assert cached.cache_hit is True
    assert cached.execution.fallback_count == 0


def test_scene_detection_resumes_from_asset_checkpoint_after_failure(tmp_path: Path) -> None:
    assets = [_video_asset(tmp_path / name) for name in ("a.mp4", "b.mp4")]
    context = _context_with_assets(tmp_path, assets)
    first_calls: list[str] = []

    class FailingSecondDetector:
        def detect(
            self,
            asset: MediaAsset,
            settings: QuickMontageSettings,
        ) -> tuple[list[Scene], bool]:
            del settings
            first_calls.append(asset.relative_path.name)
            if asset.relative_path.name == "b.mp4":
                raise MontageError("synthetic scene failure")
            return [Scene(asset_id=asset.id, start_seconds=0, end_seconds=5)], False

    with pytest.raises(MontageError, match="synthetic scene failure"):
        SceneDetectionStage(
            detector=FailingSecondDetector(),
            proxy_manager=_DirectProxyManager(),
            workers=1,
        ).run(context)

    shard_dir = context.artifacts_dir / "scene_detection_shards"
    assert (shard_dir / f"{assets[0].id}.json").is_file()
    resumed_calls: list[str] = []

    class ResumedDetector:
        def detect(
            self,
            asset: MediaAsset,
            settings: QuickMontageSettings,
        ) -> tuple[list[Scene], bool]:
            del settings
            resumed_calls.append(asset.relative_path.name)
            return [Scene(asset_id=asset.id, start_seconds=0, end_seconds=5)], True

    resumed = SceneDetectionStage(
        detector=ResumedDetector(),
        proxy_manager=_DirectProxyManager(),
        workers=1,
    ).run(context)

    assert first_calls == ["a.mp4", "b.mp4"]
    assert resumed_calls == ["b.mp4"]
    assert resumed.status is StageStatus.DEGRADED
    assert resumed.cache_hit is False
    assert resumed.execution.fallback_count == 1
    assert resumed.execution.fallback_provider == "uniform"

    recovered_calls: list[str] = []

    class RecoveredDetector:
        def detect(
            self,
            asset: MediaAsset,
            settings: QuickMontageSettings,
        ) -> tuple[list[Scene], bool]:
            del settings
            recovered_calls.append(asset.relative_path.name)
            return [Scene(asset_id=asset.id, start_seconds=0, end_seconds=5)], False

    recovered = SceneDetectionStage(
        detector=RecoveredDetector(),
        proxy_manager=_DirectProxyManager(),
        workers=1,
    ).run(context)

    assert recovered_calls == ["b.mp4"]
    assert recovered.status is StageStatus.COMPLETED
    assert recovered.cache_hit is False
    assert recovered.skipped is False
    assert recovered.execution.fallback_count == 0
    assert recovered.execution.fallback_provider is None

    class FailIfCalled:
        def detect(
            self,
            asset: MediaAsset,
            settings: QuickMontageSettings,
        ) -> tuple[list[Scene], bool]:
            del asset, settings
            raise AssertionError("recovered scenes must be cacheable")

    cached_recovery = SceneDetectionStage(
        detector=FailIfCalled(),
        proxy_manager=_DirectProxyManager(),
        workers=1,
    ).run(context)

    assert cached_recovery.status is StageStatus.CACHED
    assert cached_recovery.cache_hit is True
    assert cached_recovery.execution.fallback_count == 0


def test_scene_detection_checkpoints_successful_batch_peer_before_failure(
    tmp_path: Path,
) -> None:
    assets = [_video_asset(tmp_path / name) for name in ("a.mp4", "b.mp4")]
    context = _context_with_assets(tmp_path, assets)
    barrier = Barrier(2)

    class OneFailureDetector:
        def detect(
            self,
            asset: MediaAsset,
            settings: QuickMontageSettings,
        ) -> tuple[list[Scene], bool]:
            del settings
            barrier.wait(timeout=5)
            if asset.relative_path.name == "b.mp4":
                raise MontageError("parallel peer failed")
            return [Scene(asset_id=asset.id, start_seconds=0, end_seconds=5)], False

    with pytest.raises(MontageError, match="parallel peer failed"):
        SceneDetectionStage(
            detector=OneFailureDetector(),
            proxy_manager=_DirectProxyManager(),
            workers=2,
        ).run(context)

    assert (context.artifacts_dir / "scene_detection_shards" / f"{assets[0].id}.json").is_file()


def test_scene_detection_cancellation_does_not_start_queued_assets(tmp_path: Path) -> None:
    assets = [_video_asset(tmp_path / f"clip-{index}.mp4") for index in range(5)]
    barrier = Barrier(2)
    calls: list[str] = []
    lock = Lock()

    def cancel_after_first(current: int, total: int, message: str) -> None:
        del total, message
        if current >= 1:
            raise MontageError("cancel scene queue")

    context = _context_with_assets(tmp_path, assets, progress=cancel_after_first)

    class BlockingDetector:
        def detect(
            self,
            asset: MediaAsset,
            settings: QuickMontageSettings,
        ) -> tuple[list[Scene], bool]:
            del settings
            with lock:
                calls.append(asset.relative_path.name)
            barrier.wait(timeout=5)
            return [Scene(asset_id=asset.id, start_seconds=0, end_seconds=5)], False

    with pytest.raises(MontageError, match="cancel scene queue"):
        SceneDetectionStage(
            detector=BlockingDetector(),
            proxy_manager=_DirectProxyManager(),
            workers=2,
        ).run(context)

    assert len(calls) == 2


def _context_with_assets(
    tmp_path: Path,
    assets: list[MediaAsset],
    *,
    progress=None,
) -> ProjectContext:
    context = ProjectContext(
        input_path=tmp_path,
        workspace=tmp_path / "workspace",
        settings=Settings(),
        progress=progress,
    )
    context.prepare()
    with MediaAssetRepository(context.database_path) as repository:
        repository.initialize()
        repository.synchronize(assets, datetime.now(UTC))
    return context


def _video_asset(path: Path) -> MediaAsset:
    return MediaAsset(
        path=path,
        relative_path=Path(path.name),
        media_type=MediaType.VIDEO,
        extension=".mp4",
        size_bytes=1,
        modified_at=datetime(2026, 1, 1, tzinfo=UTC),
        modified_ns=1,
        duration_seconds=10,
        width=1920,
        height=1080,
        fps=30,
    )
