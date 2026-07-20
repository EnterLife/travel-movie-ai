import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest

from travelmovieai.analysis import quality as quality_module
from travelmovieai.analysis import scenes as scene_analysis
from travelmovieai.analysis.quality import QualityBackend, resolve_quality_backend
from travelmovieai.application.context import ProjectContext
from travelmovieai.core.config import Settings
from travelmovieai.core.exceptions import PipelineStageError
from travelmovieai.domain.enums import MediaType, StageStatus
from travelmovieai.domain.models import (
    MediaAsset,
    QualityAnalysisReport,
    QuickMontageSettings,
    Scene,
    SpeechAnalysisReport,
)
from travelmovieai.infrastructure.artifacts import artifact_fingerprint
from travelmovieai.infrastructure.database import MediaAssetRepository
from travelmovieai.infrastructure.whisper import FasterWhisperProvider
from travelmovieai.pipeline.stages import quality_analysis, speech_analysis
from travelmovieai.pipeline.stages.quality_analysis import QualityAnalysisStage
from travelmovieai.pipeline.stages.scene_detection import SceneDetectionStage
from travelmovieai.pipeline.stages.speech_analysis import SpeechAnalysisStage
from travelmovieai.story.events import detect_events


def test_detected_scene_ids_are_stable_for_identical_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset = _asset(tmp_path / "clip.mp4", duration_seconds=9)
    settings = QuickMontageSettings(
        min_scene_duration_seconds=2,
        max_scene_duration_seconds=4,
    )

    def missing_scenedetect(name: str) -> object:
        assert name == "scenedetect"
        raise ImportError

    monkeypatch.setattr(scene_analysis.importlib, "import_module", missing_scenedetect)

    first, _ = scene_analysis.SceneDetector().detect(asset, settings)
    second, _ = scene_analysis.SceneDetector().detect(asset, settings)
    rescanned, _ = scene_analysis.SceneDetector().detect(
        asset.model_copy(update={"id": UUID(int=999)}),
        settings,
    )
    changed, _ = scene_analysis.SceneDetector().detect(
        asset.model_copy(update={"size_bytes": asset.size_bytes + 1}),
        settings,
    )

    assert [scene.id for scene in first] == [scene.id for scene in second]
    assert [scene.id for scene in first] == [scene.id for scene in rescanned]
    assert all(scene.id.version == 5 for scene in first)
    assert [scene.id for scene in changed] != [scene.id for scene in first]
    assert [(scene.start_seconds, scene.end_seconds) for scene in first] == [
        (0, 4),
        (4, 9),
    ]


def test_scene_detection_replaces_legacy_random_id_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = QuickMontageSettings(
        min_scene_duration_seconds=2,
        max_scene_duration_seconds=4,
    )
    context = _context(tmp_path, Settings())
    asset = _asset(tmp_path / "legacy.mp4", duration_seconds=8)
    legacy_id = UUID("ca36e7a8-d529-4b18-a243-8917b98d35f0")
    legacy_scene = Scene(
        id=legacy_id,
        asset_id=asset.id,
        start_seconds=0,
        end_seconds=4,
        metadata={"cache_key": _legacy_scene_cache_key(asset, settings)},
    )
    _seed(context, asset, legacy_scene)

    def missing_scenedetect(name: str) -> object:
        assert name == "scenedetect"
        raise ImportError

    monkeypatch.setattr(scene_analysis.importlib, "import_module", missing_scenedetect)

    result = SceneDetectionStage(settings=settings).run(context)
    stored = MediaAssetRepository(context.database_path).list_scenes()

    assert result.status is StageStatus.DEGRADED
    assert result.execution.fallback_count == 2
    assert result.execution.fallback_provider == "uniform"
    assert legacy_id not in {scene.id for scene in stored}
    assert all(scene.id.version == 5 for scene in stored)


def test_event_ids_are_stable_and_change_with_membership(tmp_path: Path) -> None:
    asset = _asset(tmp_path / "clip.mp4", duration_seconds=10)
    scenes = [
        Scene(
            id=UUID(int=1),
            asset_id=asset.id,
            start_seconds=0,
            end_seconds=4,
            metadata={"location_type": "city", "activity": "walking"},
        ),
        Scene(
            id=UUID(int=2),
            asset_id=asset.id,
            start_seconds=4,
            end_seconds=8,
            metadata={"location_type": "city", "activity": "walking"},
        ),
    ]

    first, _ = detect_events(scenes, [asset])
    second, _ = detect_events(scenes, [asset])
    changed, _ = detect_events(
        [scenes[0], scenes[1].model_copy(update={"id": UUID(int=3)})],
        [asset],
    )

    assert first.events[0].id == second.events[0].id
    assert first.events[0].id.version == 5
    assert changed.events[0].id != first.events[0].id


def test_artifact_fingerprint_canonicalizes_equivalent_numbers() -> None:
    integer = artifact_fingerprint({"threshold": 27, "nested": [0, -0.0]})
    floating_point = artifact_fingerprint({"nested": [0.0, 0], "threshold": 27.0})

    assert integer == floating_point
    assert artifact_fingerprint(True) != artifact_fingerprint(1)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_artifact_fingerprint_rejects_non_finite_numbers(invalid: float) -> None:
    with pytest.raises(ValueError, match="NaN or infinity"):
        artifact_fingerprint(invalid)


def test_artifact_fingerprint_rejects_ambiguous_object_values() -> None:
    with pytest.raises(TypeError, match="do not support"):
        artifact_fingerprint(object())


def test_quality_backend_respects_explicit_cpu_and_rejects_unknown_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported: list[str] = []

    def fake_import(name: str) -> object:
        imported.append(name)
        assert name == "cv2"
        return SimpleNamespace(__version__="4.10-test")

    monkeypatch.setattr(quality_module.importlib, "import_module", fake_import)

    backend = resolve_quality_backend("cpu")

    assert backend == QualityBackend("opencv", "cpu", "4.10-test")
    assert imported == ["cv2"]
    with pytest.raises(ValueError, match="Unsupported quality analysis device"):
        resolve_quality_backend("quantum")


def test_whisper_cache_only_mode_never_enables_downloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    class MissingWhisperModel:
        def __init__(self, model: str, **kwargs: object) -> None:
            assert model == "medium"
            calls.append(kwargs)
            raise OSError("model is not cached")

    monkeypatch.setattr(
        "travelmovieai.infrastructure.whisper.importlib.import_module",
        lambda name: SimpleNamespace(WhisperModel=MissingWhisperModel),
    )
    cache_dir = tmp_path / "models" / "faster-whisper"
    provider = FasterWhisperProvider(
        "medium",
        "cpu",
        cache_dir=cache_dir,
        allow_download=False,
    )

    with pytest.raises(PipelineStageError, match="cache-only mode.*allow_model_download=true"):
        provider._ensure_loaded()

    assert calls == [
        {
            "device": "cpu",
            "compute_type": "int8",
            "local_files_only": True,
            "download_root": str(cache_dir.resolve()),
        }
    ]


def test_whisper_online_mode_uses_configured_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    class LoadedWhisperModel:
        def __init__(self, model: str, **kwargs: object) -> None:
            assert model == "medium"
            calls.append(kwargs)

    monkeypatch.setattr(
        "travelmovieai.infrastructure.whisper.importlib.import_module",
        lambda name: SimpleNamespace(WhisperModel=LoadedWhisperModel),
    )
    cache_dir = tmp_path / "models" / "faster-whisper"
    provider = FasterWhisperProvider(
        "medium",
        "cpu",
        cache_dir=cache_dir,
        allow_download=True,
    )

    provider._ensure_loaded()

    assert calls[0]["local_files_only"] is False
    assert calls[0]["download_root"] == str(cache_dir.resolve())


def test_speech_stage_propagates_offline_model_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        model_cache=tmp_path / "models",
        allow_model_download=False,
        device="cpu",
    )
    context = _context(tmp_path, settings)
    asset = _asset(tmp_path / "speech.mp4", duration_seconds=5, has_audio=True)
    scene = Scene(asset_id=asset.id, start_seconds=0, end_seconds=4)
    _seed(context, asset, scene)
    captured: dict[str, object] = {}

    class FakeProvider:
        name = "fake-whisper"

        def __init__(
            self,
            model: str,
            device: str,
            *,
            cache_dir: Path,
            allow_download: bool,
        ) -> None:
            self.model = model
            captured.update(
                model=model,
                device=device,
                cache_dir=cache_dir,
                allow_download=allow_download,
            )

        def release(self) -> None:
            pass

    def fake_analyze(
        scenes: list[Scene],
        *args: object,
        **kwargs: object,
    ) -> SpeechAnalysisReport:
        updated = scenes[0].model_copy(
            update={
                "transcript": "cached locally",
                "metadata": {**scenes[0].metadata, "speech_cache_key": "test"},
            }
        )
        return SpeechAnalysisReport(
            created_at=datetime.now(UTC),
            provider="fake-whisper",
            model="medium",
            scenes=[updated],
            transcribed_count=1,
        )

    monkeypatch.setattr(speech_analysis, "FasterWhisperProvider", FakeProvider)
    monkeypatch.setattr(speech_analysis, "analyze_speech", fake_analyze)

    SpeechAnalysisStage().run(context)

    assert captured == {
        "model": "medium",
        "device": "cpu",
        "cache_dir": (tmp_path / "models" / "faster-whisper").resolve(),
        "allow_download": False,
    }


def test_quality_cache_invalidates_when_backend_or_device_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path, Settings(device="cpu"))
    asset = _asset(tmp_path / "quality.mp4", duration_seconds=5)
    keyframe = tmp_path / "quality.png"
    keyframe.write_bytes(b"synthetic")
    scene = Scene(
        asset_id=asset.id,
        start_seconds=0,
        end_seconds=4,
        keyframe_path=keyframe,
    )
    _seed(context, asset, scene)
    backend = [QualityBackend("opencv", "cpu", "4.10")]
    calls = 0

    monkeypatch.setattr(quality_analysis, "resolve_quality_backend", lambda device: backend[0])
    monkeypatch.setattr(quality_analysis, "create_quality_analyzer", lambda value: object())
    monkeypatch.setattr(
        quality_analysis,
        "detect_resource_profile",
        lambda *args, **kwargs: SimpleNamespace(analysis_workers=2),
    )

    def fake_analyze(
        scenes: list[Scene],
        *,
        analyzer: object,
        workers: int,
        progress: object | None = None,
    ) -> QualityAnalysisReport:
        nonlocal calls
        calls += 1
        assert workers == 2
        assert progress is None
        updated = scenes[0].model_copy(
            update={
                "quality_score": 75,
                "metadata": {
                    **scenes[0].metadata,
                    "quality_metrics": {"quality_score": 75, "backend": backend[0].name},
                },
            }
        )
        return QualityAnalysisReport(created_at=datetime.now(UTC), scenes=[updated])

    monkeypatch.setattr(quality_analysis, "analyze_scene_quality", fake_analyze)

    QualityAnalysisStage().run(context)
    QualityAnalysisStage().run(context)
    context.settings.device = "cuda"
    backend[0] = QualityBackend("torch-cuda", "cuda", "2.5")
    QualityAnalysisStage().run(context)

    assert calls == 2


def _context(tmp_path: Path, settings: Settings) -> ProjectContext:
    context = ProjectContext(
        input_path=tmp_path / "media",
        workspace=tmp_path / "workspace",
        settings=settings,
        montage_settings=QuickMontageSettings(speech_analysis=True),
    )
    context.prepare()
    return context


def _asset(
    path: Path,
    *,
    duration_seconds: float,
    has_audio: bool = False,
) -> MediaAsset:
    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    streams: list[dict[str, object]] = [{"codec_type": "video"}]
    if has_audio:
        streams.append({"codec_type": "audio"})
    return MediaAsset(
        path=path,
        relative_path=Path(path.name),
        media_type=MediaType.VIDEO,
        extension=".mp4",
        size_bytes=1,
        modified_at=created_at,
        modified_ns=1,
        created_at=created_at,
        duration_seconds=duration_seconds,
        probe_metadata={"streams": streams},
    )


def _seed(context: ProjectContext, asset: MediaAsset, scene: Scene) -> None:
    repository = MediaAssetRepository(context.database_path)
    repository.initialize()
    repository.synchronize([asset], datetime.now(UTC))
    repository.synchronize_scenes([scene])


def _legacy_scene_cache_key(asset: MediaAsset, settings: QuickMontageSettings) -> str:
    payload: dict[str, object] = {
        "asset_id": str(asset.id),
        "size": asset.size_bytes,
        "modified_ns": asset.modified_ns,
        "threshold": settings.scene_threshold,
        "min": settings.min_scene_duration_seconds,
        "max": settings.max_scene_duration_seconds,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
