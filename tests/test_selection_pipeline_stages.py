import hashlib
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from PIL import Image

from travelmovieai.analysis.speech import speech_cache_key
from travelmovieai.application.context import ProjectContext
from travelmovieai.core.config import Settings
from travelmovieai.domain.enums import (
    ActivityType,
    LocationType,
    MediaType,
    PipelineStage,
    StageStatus,
)
from travelmovieai.domain.models import (
    AudioAnalysisReport,
    AudioSceneAnalysis,
    Event,
    FrameSamplingReport,
    MediaAsset,
    MontageClip,
    MontageQualityReport,
    MusicPlan,
    NarrationReport,
    QualityAnalysisReport,
    QuickMontagePlan,
    QuickMontageSettings,
    Scene,
    SceneDetectionReport,
    SceneSelectionReport,
    SpeechAnalysisReport,
    StageCacheManifest,
    Storyboard,
    VisionAnalysisReport,
)
from travelmovieai.infrastructure.database import MediaAssetRepository
from travelmovieai.pipeline.stages import (
    audio_analysis,
    frame_sampling,
    music_selection,
    quality_analysis,
    rendering,
    speech_analysis,
    timeline_builder,
    vision_analysis,
)
from travelmovieai.pipeline.stages.audio_analysis import AudioAnalysisStage
from travelmovieai.pipeline.stages.duplicate_detection import DuplicateDetectionStage
from travelmovieai.pipeline.stages.frame_sampling import FrameSamplingStage
from travelmovieai.pipeline.stages.music_selection import MusicSelectionStage
from travelmovieai.pipeline.stages.narration import NarrationStage
from travelmovieai.pipeline.stages.quality_analysis import QualityAnalysisStage
from travelmovieai.pipeline.stages.rendering import RenderingStage
from travelmovieai.pipeline.stages.scene_ranking import SceneRankingStage
from travelmovieai.pipeline.stages.speech_analysis import SpeechAnalysisStage
from travelmovieai.pipeline.stages.storyboard import StoryBuilderStage
from travelmovieai.pipeline.stages.timeline_builder import TimelineBuilderStage
from travelmovieai.pipeline.stages.vision_analysis import VisionAnalysisStage
from travelmovieai.story.music import MusicPlanExecution


def test_scene_ranking_stage_persists_ranked_scenes(tmp_path: Path) -> None:
    context = _context(tmp_path)
    asset = _asset(tmp_path / "clip.mp4")
    weaker = _scene(asset, score=45, start=0)
    stronger = _scene(asset, score=90, start=4)
    _seed_project(context, [asset], [weaker, stronger])

    result = SceneRankingStage().run(context)
    cached = SceneRankingStage().run(context)
    report = SceneDetectionReport.model_validate_json(
        (context.artifacts_dir / "ranked_scenes.json").read_text(encoding="utf-8")
    )
    stored = {
        scene.id: scene for scene in MediaAssetRepository(context.database_path).list_scenes()
    }

    assert result.stage is PipelineStage.SCENE_RANKING
    assert result.skipped is False
    assert cached.skipped is True
    assert [scene.id for scene in report.scenes] == [stronger.id, weaker.id]
    assert (
        stored[stronger.id].metadata["ranking_score"] > stored[weaker.id].metadata["ranking_score"]
    )
    assert "ranking_factors" in stored[stronger.id].metadata


def test_timeline_builder_stage_writes_plan_and_selection_report(tmp_path: Path) -> None:
    context = _context(tmp_path)
    assets = [
        _asset(tmp_path / "opening.mp4"),
        _asset(tmp_path / "highlight.mp4"),
    ]
    scenes = [
        _scene(
            assets[0],
            score=85,
            start=0,
            story_section_index=0,
            story_section_role="opening",
            story_role_order=0,
        ),
        _scene(
            assets[1],
            score=92,
            start=0,
            story_section_index=2,
            story_section_role="highlight",
            story_role_order=2,
        ),
    ]
    _seed_project(context, assets, scenes)
    SceneRankingStage().run(context)

    result = TimelineBuilderStage().run(context)
    plan = QuickMontagePlan.model_validate_json(
        (context.artifacts_dir / "quick_timeline.json").read_text(encoding="utf-8")
    )
    selection = SceneSelectionReport.model_validate_json(
        (context.artifacts_dir / "selection_decisions.json").read_text(encoding="utf-8")
    )

    assert result.stage is PipelineStage.TIMELINE_BUILDER
    assert result.skipped is False
    assert plan.selection_mode == "semantic"
    assert plan.settings.preserve_chronology is True
    assert {clip.scene_id for clip in plan.clips} == {scenes[0].id, scenes[1].id}
    assert {decision.scene_id for decision in selection.decisions} == {scene.id for scene in scenes}


def test_frame_sampling_stage_reuses_cached_contact_sheets(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context = _context(tmp_path)
    asset = _asset(tmp_path / "frame-source.mp4")
    scene = _scene(asset, score=80, start=0)
    _seed_project(context, [asset], [scene])
    calls = 0

    class FakeExtractor:
        backend_summary = "fake"

        def __init__(self, *args: object, **kwargs: object) -> None:
            self.frame_sample_count = int(kwargs["frame_sample_count"])

        def extract(self, source_scene: Scene, media_asset: MediaAsset, frames_dir: Path) -> Path:
            del media_asset
            nonlocal calls
            calls += 1
            output = frames_dir / f"{source_scene.id}.png"
            output.parent.mkdir(parents=True, exist_ok=True)
            Image.new(
                "RGB",
                (1440, ((self.frame_sample_count + 2) // 3) * 270),
                (0, calls, 0),
            ).save(output)
            return output

        def sampling_metadata(
            self,
            source_scene: Scene,
            media_asset: MediaAsset,
            image_path: Path,
        ) -> dict[str, object]:
            del source_scene, media_asset
            count = self.frame_sample_count
            return {
                "schema_version": frame_sampling.CONTACT_SHEET_SCHEMA_VERSION,
                "sample_count": count,
                "sample_positions": [index / count for index in range(count)],
                "sample_timestamps_seconds": [float(index) for index in range(count)],
                "columns": min(3, count),
                "rows": (count + 2) // 3,
                "content_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            }

    monkeypatch.setattr(frame_sampling, "RepresentativeFrameExtractor", FakeExtractor)
    monkeypatch.setattr(
        frame_sampling,
        "detect_resource_profile",
        lambda *args, **kwargs: type("Profile", (), {"nvenc": False, "frame_workers": 2})(),
    )

    first = FrameSamplingStage().run(context)
    second = FrameSamplingStage().run(context)

    assert first.skipped is False
    assert second.skipped is True
    assert first.status is StageStatus.COMPLETED
    assert second.status is StageStatus.CACHED
    assert calls == 1
    assert (context.artifacts_dir / "frame_sampling.cache.json").is_file()


def test_frame_sampling_cache_restores_missing_database_frame_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path)
    asset = _asset(tmp_path / "frame-source.mp4")
    scene = _scene(asset, score=80, start=0)
    _seed_project(context, [asset], [scene])
    calls = 0

    class FakeExtractor:
        backend_summary = "fake"

        def __init__(self, *args: object, **kwargs: object) -> None:
            self.frame_sample_count = int(kwargs["frame_sample_count"])

        def extract(self, source_scene: Scene, media_asset: MediaAsset, frames_dir: Path) -> Path:
            del media_asset
            nonlocal calls
            calls += 1
            output = frames_dir / f"{source_scene.id}.png"
            output.parent.mkdir(parents=True, exist_ok=True)
            Image.new(
                "RGB",
                (1440, ((self.frame_sample_count + 2) // 3) * 270),
                (0, calls, 0),
            ).save(output)
            return output

        def sampling_metadata(
            self,
            source_scene: Scene,
            media_asset: MediaAsset,
            image_path: Path,
        ) -> dict[str, object]:
            del source_scene, media_asset
            count = self.frame_sample_count
            return {
                "schema_version": frame_sampling.CONTACT_SHEET_SCHEMA_VERSION,
                "sample_count": count,
                "sample_positions": [index / count for index in range(count)],
                "sample_timestamps_seconds": [float(index) for index in range(count)],
                "columns": min(3, count),
                "rows": (count + 2) // 3,
                "content_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            }

    monkeypatch.setattr(frame_sampling, "RepresentativeFrameExtractor", FakeExtractor)
    monkeypatch.setattr(
        frame_sampling,
        "detect_resource_profile",
        lambda *args, **kwargs: type("Profile", (), {"nvenc": False, "frame_workers": 2})(),
    )

    FrameSamplingStage().run(context)
    with MediaAssetRepository(context.database_path) as repository:
        stored = repository.list_scenes()[0]
        contact_sheet = stored.metadata["contact_sheet"]
        damaged_metadata = {
            key: value for key, value in stored.metadata.items() if key != "contact_sheet"
        }
        damaged_metadata["downstream_marker"] = "preserve"
        repository.synchronize_scenes(
            [
                stored.model_copy(
                    update={
                        "keyframe_path": None,
                        "caption": "Later-stage caption",
                        "metadata": damaged_metadata,
                    }
                )
            ]
        )

    cached = FrameSamplingStage().run(context)
    with MediaAssetRepository(context.database_path) as repository:
        restored = repository.list_scenes()[0]

    assert cached.status is StageStatus.CACHED
    assert calls == 1
    assert restored.keyframe_path is not None
    assert restored.keyframe_path.is_file()
    assert restored.metadata["contact_sheet"] == contact_sheet
    assert restored.metadata["downstream_marker"] == "preserve"
    assert restored.caption == "Later-stage caption"


@pytest.mark.parametrize("tamper_mode", ["corrupt", "substitute"])
def test_frame_sampling_stage_reextracts_tampered_contact_sheet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper_mode: str,
) -> None:
    context = _context(tmp_path)
    asset = _asset(tmp_path / "frame-source.mp4")
    scene = _scene(asset, score=80, start=0)
    _seed_project(context, [asset], [scene])
    calls = 0

    class FakeExtractor:
        backend_summary = "fake"

        def __init__(self, *args: object, **kwargs: object) -> None:
            self.frame_sample_count = int(kwargs["frame_sample_count"])

        def extract(self, source_scene: Scene, media_asset: MediaAsset, frames_dir: Path) -> Path:
            del media_asset
            nonlocal calls
            calls += 1
            output = frames_dir / f"{source_scene.id}.png"
            output.parent.mkdir(parents=True, exist_ok=True)
            Image.new(
                "RGB",
                (1440, ((self.frame_sample_count + 2) // 3) * 270),
                (0, calls, 0),
            ).save(output)
            return output

        def sampling_metadata(
            self,
            source_scene: Scene,
            media_asset: MediaAsset,
            image_path: Path,
        ) -> dict[str, object]:
            del source_scene, media_asset
            count = self.frame_sample_count
            return {
                "schema_version": frame_sampling.CONTACT_SHEET_SCHEMA_VERSION,
                "sample_count": count,
                "sample_positions": [index / count for index in range(count)],
                "sample_timestamps_seconds": [float(index) for index in range(count)],
                "columns": min(3, count),
                "rows": (count + 2) // 3,
                "content_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            }

    monkeypatch.setattr(frame_sampling, "RepresentativeFrameExtractor", FakeExtractor)
    monkeypatch.setattr(
        frame_sampling,
        "detect_resource_profile",
        lambda *args, **kwargs: type("Profile", (), {"nvenc": False, "frame_workers": 2})(),
    )

    FrameSamplingStage().run(context)
    report = FrameSamplingReport.model_validate_json(
        (context.artifacts_dir / "frame_sampling.json").read_text(encoding="utf-8")
    )
    contact_sheet_path = report.scenes[0].keyframe_path
    assert contact_sheet_path is not None
    if tamper_mode == "corrupt":
        contact_sheet_path.write_bytes(b"not a decodable image")
    else:
        with Image.open(contact_sheet_path) as cached_image:
            size = cached_image.size
        Image.new("RGB", size, (255, 0, 0)).save(contact_sheet_path)

    rerun = FrameSamplingStage().run(context)

    assert rerun.status is StageStatus.COMPLETED
    assert calls == 2
    with Image.open(contact_sheet_path) as regenerated:
        assert regenerated.getpixel((0, 0)) == (0, 2, 0)


def test_frame_sampling_stage_bounds_parallel_nvdec_decode(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context = _context(tmp_path)
    asset = _asset(tmp_path / "cuda-source.mp4")
    scene = _scene(asset, score=80, start=0)
    _seed_project(context, [asset], [scene])
    captured: dict[str, object] = {}

    class FakeExtractor:
        backend_summary = "fake"

        def __init__(self, *args: object, use_cuda_decode: bool, **kwargs: object) -> None:
            captured["use_cuda_decode"] = use_cuda_decode

    def fake_extract_frames(
        source_scenes: list[Scene],
        assets: object,
        extractor: object,
        frames_dir: Path,
        workers: int,
        *,
        progress: object | None = None,
    ) -> tuple[list[Scene], int, int]:
        del progress
        captured["workers"] = workers
        frame_path = frames_dir / "cuda-source.png"
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        frame_path.write_bytes(b"png")
        return [source_scenes[0].model_copy(update={"keyframe_path": frame_path})], 1, 0

    monkeypatch.setattr(frame_sampling, "RepresentativeFrameExtractor", FakeExtractor)
    monkeypatch.setattr(frame_sampling, "_extract_frames", fake_extract_frames)
    monkeypatch.setattr(
        frame_sampling,
        "detect_resource_profile",
        lambda *args, **kwargs: type(
            "Profile",
            (),
            {"nvenc": True, "frame_workers": 8, "resource_mode": "performance"},
        )(),
    )

    result = FrameSamplingStage().run(context)

    assert captured == {"use_cuda_decode": True, "workers": 2}
    assert "decode=NVDEC" in result.message


def test_frame_sampling_stage_keeps_cpu_decode_parallel(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context = _context(tmp_path, settings=Settings(device="cpu"))
    asset = _asset(tmp_path / "cpu-source.mp4")
    scene = _scene(asset, score=80, start=0)
    _seed_project(context, [asset], [scene])
    captured: dict[str, object] = {}

    class FakeExtractor:
        backend_summary = "fake"

        def __init__(self, *args: object, use_cuda_decode: bool, **kwargs: object) -> None:
            captured["use_cuda_decode"] = use_cuda_decode

    def fake_extract_frames(
        source_scenes: list[Scene],
        assets: object,
        extractor: object,
        frames_dir: Path,
        workers: int,
        *,
        progress: object | None = None,
    ) -> tuple[list[Scene], int, int]:
        del progress
        captured["workers"] = workers
        frame_path = frames_dir / "cpu-source.png"
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        frame_path.write_bytes(b"png")
        return [source_scenes[0].model_copy(update={"keyframe_path": frame_path})], 1, 0

    monkeypatch.setattr(frame_sampling, "RepresentativeFrameExtractor", FakeExtractor)
    monkeypatch.setattr(frame_sampling, "_extract_frames", fake_extract_frames)
    monkeypatch.setattr(
        frame_sampling,
        "detect_resource_profile",
        lambda *args, **kwargs: type("Profile", (), {"nvenc": True, "frame_workers": 8})(),
    )

    result = FrameSamplingStage().run(context)

    assert captured == {"use_cuda_decode": False, "workers": 8}
    assert "decode=CPU" in result.message


def test_quality_analysis_stage_reuses_cached_metrics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context = _context(tmp_path)
    asset = _asset(tmp_path / "quality-source.mp4")
    keyframe = tmp_path / "quality.png"
    keyframe.write_bytes(b"png")
    scene = _scene(asset, score=80, start=0).model_copy(update={"keyframe_path": keyframe})
    _seed_project(context, [asset], [scene])
    calls = 0

    def fake_analyze(
        scenes: list[Scene],
        *,
        analyzer: object | None = None,
        workers: int = 1,
        progress: object | None = None,
    ) -> QualityAnalysisReport:
        del analyzer, progress
        nonlocal calls
        calls += 1
        assert workers == 3
        updated = [
            item.model_copy(
                update={
                    "quality_score": 77,
                    "metadata": {
                        **item.metadata,
                        "quality_metrics": {"quality_score": 77, "backend": "fake"},
                    },
                }
            )
            for item in scenes
        ]
        return QualityAnalysisReport(created_at=datetime.now(UTC), scenes=updated)

    monkeypatch.setattr(quality_analysis, "analyze_scene_quality", fake_analyze)
    monkeypatch.setattr(
        quality_analysis,
        "detect_resource_profile",
        lambda *args, **kwargs: type("Profile", (), {"analysis_workers": 3})(),
    )

    first = QualityAnalysisStage().run(context)
    second = QualityAnalysisStage().run(context)

    assert first.skipped is False
    assert second.skipped is True
    assert first.status is StageStatus.COMPLETED
    assert second.status is StageStatus.CACHED
    assert calls == 1
    assert (context.artifacts_dir / "quality_analysis.cache.json").is_file()


def test_vision_analysis_stage_reuses_cached_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context = _context(tmp_path)
    asset = _asset(tmp_path / "vision-source.mp4")
    keyframe = tmp_path / "keyframe.png"
    keyframe.write_bytes(b"png")
    scene = _scene(asset, score=80, start=0).model_copy(update={"keyframe_path": keyframe})
    _seed_project(context, [asset], [scene])
    calls = 0
    provider_devices: list[str] = []

    def fake_profile(*args: object, **kwargs: object) -> object:
        return type(
            "Profile",
            (),
            {
                "gpu_memory_mb": None,
                "memory_mb": None,
                "model_batch_size": 1,
                "device": "cpu",
            },
        )()

    releases = 0

    class FakeVisionProvider:
        name = "fake-vision"
        model = "fake-model"

        def release(self) -> None:
            nonlocal releases
            releases += 1

    def fake_provider(*args: object, **kwargs: object) -> object:
        provider_devices.append(str(kwargs["device"]))
        return FakeVisionProvider()

    def fake_analyze(
        scenes: list[Scene],
        provider: object,
        style: object,
        **kwargs: object,
    ) -> VisionAnalysisReport:
        nonlocal calls
        calls += 1
        updated = [
            item.model_copy(
                update={
                    "caption": "Cached view",
                    "importance_score": 88,
                    "metadata": {
                        **item.metadata,
                        "vision_cache_key": "fake",
                        "vision_provider": "fake-vision",
                    },
                }
            )
            for item in scenes
        ]
        return VisionAnalysisReport(
            created_at=datetime.now(UTC),
            provider="fake-vision",
            model="fake-model",
            prompt_version="test",
            scenes=updated,
            analyzed_count=len(updated),
        )

    monkeypatch.setattr(vision_analysis, "detect_resource_profile", fake_profile)
    monkeypatch.setattr(vision_analysis, "build_vision_provider", fake_provider)
    monkeypatch.setattr(vision_analysis, "analyze_scenes", fake_analyze)

    first = VisionAnalysisStage().run(context)
    repository = MediaAssetRepository(context.database_path)
    stored = repository.list_scenes()[0]
    tampered_metadata = {
        **stored.metadata,
        "speech_cache_key": "speech",
    }
    tampered_metadata.pop("vision_cache_key")
    tampered_metadata.pop("vision_provider")
    repository.synchronize_scenes(
        [
            stored.model_copy(
                update={
                    "caption": None,
                    "importance_score": None,
                    "transcript": "Later speech result",
                    "metadata": tampered_metadata,
                }
            )
        ]
    )
    context_with_larger_pool = ProjectContext(
        input_path=context.input_path,
        workspace=context.workspace,
        settings=context.settings.model_copy(update={"vision_model_pool_size": 2}),
        output_path=context.output_path,
        style=context.style,
        montage_settings=context.montage_settings,
    )
    second = VisionAnalysisStage().run(context_with_larger_pool)

    assert first.skipped is False
    assert second.skipped is True
    assert first.status is StageStatus.COMPLETED
    assert second.status is StageStatus.CACHED
    assert calls == 1
    assert provider_devices == ["cpu"]
    assert first.execution.provider == "fake-vision"
    assert first.execution.model == "fake-model"
    assert second.execution.provider == "fake-vision"
    assert second.execution.model == "fake-model"
    assert second.execution.fallback_count == 0
    assert second.execution.fallback_provider is None
    assert releases == 1
    restored = repository.list_scenes()[0]
    assert restored.transcript == "Later speech result"
    assert restored.caption == "Cached view"
    assert restored.importance_score == 88
    assert restored.metadata["vision_cache_key"] == "fake"
    assert restored.metadata["vision_provider"] == "fake-vision"
    assert restored.metadata["speech_cache_key"] == "speech"
    assert (context.artifacts_dir / "vision_analysis.cache.json").is_file()


def test_vision_stage_surfaces_non_sticky_scene_fallback_as_degraded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path)
    asset = _asset(tmp_path / "vision-fallback.mp4")
    keyframe = tmp_path / "vision-fallback.png"
    keyframe.write_bytes(b"frame")
    scene = _scene(asset, score=80, start=0).model_copy(update={"keyframe_path": keyframe})
    _seed_project(context, [asset], [scene])

    class FakeVisionProvider:
        name = "fake-vision"
        model = "fake-model"

        def release(self) -> None:
            return None

    def fake_analyze(
        scenes: list[Scene],
        provider: object,
        style: object,
        **kwargs: object,
    ) -> VisionAnalysisReport:
        del provider, style, kwargs
        degraded = scenes[0].model_copy(
            update={
                "caption": "Retry required",
                "importance_score": 25,
                "metadata": {**scenes[0].metadata, "vision_status": "degraded"},
            }
        )
        return VisionAnalysisReport(
            created_at=datetime.now(UTC),
            provider="fake-vision",
            model="fake-model",
            prompt_version="test",
            scenes=[degraded],
            degraded_count=1,
            retry_count=2,
        )

    monkeypatch.setattr(vision_analysis, "analyze_scenes", fake_analyze)
    monkeypatch.setattr(
        vision_analysis,
        "build_vision_provider",
        lambda **_: FakeVisionProvider(),
    )

    result = VisionAnalysisStage().run(context)

    assert result.status is StageStatus.DEGRADED
    assert result.cache_hit is False
    assert result.skipped is False
    assert result.execution.fallback_count == 1
    assert result.execution.retry_count == 2
    assert result.execution.provider == "fake-vision"
    assert result.execution.fallback_provider == "deterministic"
    assert result.execution.model == "fake-model"


def test_speech_analysis_stage_reuses_cached_transcripts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context = _context(tmp_path)
    asset = _asset(tmp_path / "speech-source.mp4").model_copy(
        update={"probe_metadata": {"streams": [{"codec_type": "audio"}]}}
    )
    scene = _scene(asset, score=80, start=0)
    _seed_project(context, [asset], [scene])
    calls = 0

    def fake_analyze(
        scenes: list[Scene],
        assets: list[MediaAsset],
        provider: object,
        ffmpeg_binary: str,
        audio_dir: Path,
        *,
        timeout_seconds: float = 120,
        progress: object | None = None,
    ) -> SpeechAnalysisReport:
        del assets, provider, ffmpeg_binary, audio_dir, progress
        nonlocal calls
        calls += 1
        assert timeout_seconds == context.settings.frame_extraction_timeout_seconds
        updated = [
            item.model_copy(
                update={
                    "transcript": "Welcome.",
                    "metadata": {**item.metadata, "speech_cache_key": "fake"},
                }
            )
            for item in scenes
        ]
        return SpeechAnalysisReport(
            created_at=datetime.now(UTC),
            provider="fake-whisper",
            model="medium",
            scenes=updated,
            transcribed_count=len(updated),
        )

    monkeypatch.setattr(speech_analysis, "analyze_speech", fake_analyze)

    first = SpeechAnalysisStage().run(context)
    second = SpeechAnalysisStage().run(context)

    assert first.skipped is False
    assert second.skipped is True
    assert first.status is StageStatus.COMPLETED
    assert second.status is StageStatus.CACHED
    assert calls == 1
    assert first.execution.provider == "fake-whisper"
    assert first.execution.model == "medium"
    assert second.execution.provider == "faster-whisper"
    assert (context.artifacts_dir / "speech_analysis.cache.json").is_file()


def test_speech_analysis_stage_resumes_checkpoint_after_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path)
    asset = _asset(tmp_path / "speech-source.mp4").model_copy(
        update={"probe_metadata": {"streams": [{"codec_type": "audio"}]}}
    )
    scene = _scene(asset, score=80, start=0)
    _seed_project(context, [asset], [scene])
    calls = 0

    def fake_analyze(
        scenes: list[Scene],
        assets: list[MediaAsset],
        provider: object,
        ffmpeg_binary: str,
        audio_dir: Path,
        *,
        timeout_seconds: float = 120,
        progress: object | None = None,
        checkpoint=None,
    ) -> SpeechAnalysisReport:
        del ffmpeg_binary, audio_dir, timeout_seconds, progress
        nonlocal calls
        calls += 1
        if calls == 1:
            assert callable(checkpoint)
            model = str(provider.model)
            updated = scenes[0].model_copy(
                update={
                    "transcript": "Saved partial transcript.",
                    "metadata": {
                        **scenes[0].metadata,
                        "speech_cache_key": speech_cache_key(scenes[0], assets[0], model),
                    },
                }
            )
            checkpoint(updated)
            raise RuntimeError("later whisper scene failed")
        assert scenes[0].transcript == "Saved partial transcript."
        return SpeechAnalysisReport(
            created_at=datetime.now(UTC),
            provider="fake-whisper",
            model=str(provider.model),
            scenes=scenes,
            cached_count=1,
        )

    monkeypatch.setattr(speech_analysis, "analyze_speech", fake_analyze)

    with pytest.raises(RuntimeError, match="later whisper scene failed"):
        SpeechAnalysisStage().run(context)
    resumed = SpeechAnalysisStage().run(context)

    assert calls == 2
    assert resumed.status is StageStatus.CACHED
    assert MediaAssetRepository(context.database_path).list_scenes()[0].transcript == (
        "Saved partial transcript."
    )


def test_speech_analysis_stage_respects_disabled_montage_setting(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context = ProjectContext(
        input_path=tmp_path / "media",
        workspace=tmp_path / "workspace",
        settings=Settings(),
        montage_settings=QuickMontageSettings(
            semantic_analysis=True,
            speech_analysis=False,
        ),
    )
    context.prepare()
    asset = _asset(tmp_path / "speech-source.mp4").model_copy(
        update={"probe_metadata": {"streams": [{"codec_type": "audio"}]}}
    )
    scene = _scene(asset, score=80, start=0).model_copy(
        update={
            "transcript": "stale transcript",
            "metadata": {
                "speech_cache_key": "stale",
                "speech_provider": "stale",
                "speech_model": "stale",
                "speech_language": "en",
                "speech_confidence": 0.9,
                "speech_segments": [],
                "unrelated": "preserved",
            },
        }
    )
    _seed_project(context, [asset], [scene])
    for name in ("speech_analysis.json", "speech_analysis.cache.json"):
        (context.artifacts_dir / name).write_text("stale", encoding="utf-8")

    def fail_analyze(*args: object, **kwargs: object) -> SpeechAnalysisReport:
        raise AssertionError("speech analysis should be skipped")

    monkeypatch.setattr(speech_analysis, "analyze_speech", fail_analyze)

    result = SpeechAnalysisStage().run(context)
    stored = MediaAssetRepository(context.database_path).list_scenes()[0]

    assert result.stage is PipelineStage.SPEECH_ANALYSIS
    assert result.skipped is True
    assert result.status is StageStatus.DISABLED
    assert "disabled" in result.message
    assert stored.transcript is None
    assert stored.metadata == {"unrelated": "preserved"}
    assert not (context.artifacts_dir / "speech_analysis.json").exists()
    assert not (context.artifacts_dir / "speech_analysis.cache.json").exists()


def test_audio_analysis_stage_reuses_cached_scene_audio_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context = _context(tmp_path)
    asset = _asset(tmp_path / "audio-source.mp4").model_copy(
        update={"probe_metadata": {"streams": [{"codec_type": "audio"}]}}
    )
    scene = _scene(asset, score=80, start=0).model_copy(update={"transcript": "Welcome."})
    _seed_project(context, [asset], [scene])
    calls = 0

    def fake_analyze(
        scenes: list[Scene],
        assets: list[MediaAsset],
        ffmpeg_binary: str,
        *,
        timeout_seconds: float = 120,
        progress: object | None = None,
    ) -> AudioAnalysisReport:
        del assets, ffmpeg_binary, progress
        nonlocal calls
        calls += 1
        assert timeout_seconds == context.settings.frame_extraction_timeout_seconds
        analysis = AudioSceneAnalysis(
            scene_id=scenes[0].id,
            has_audio=True,
            primary_label="speech",
            labels=["speech"],
            speech_likelihood=0.9,
            ambience_score=65,
        )
        updated = [
            scenes[0].model_copy(
                update={
                    "metadata": {
                        **scenes[0].metadata,
                        "audio_analysis": analysis.model_dump(mode="json"),
                        "audio_context": analysis.labels,
                        "audio_features": {
                            "primary_label": analysis.primary_label,
                            "speech_likelihood": analysis.speech_likelihood,
                            "noise_score": analysis.noise_score,
                            "ambience_score": analysis.ambience_score,
                        },
                    }
                }
            )
        ]
        return AudioAnalysisReport(
            created_at=datetime.now(UTC),
            scenes=updated,
            analyses=[analysis],
            analyzed_count=1,
        )

    monkeypatch.setattr(audio_analysis, "analyze_audio", fake_analyze)

    first = AudioAnalysisStage().run(context)
    second = AudioAnalysisStage().run(context)

    assert first.skipped is False
    assert second.skipped is True
    assert first.status is StageStatus.COMPLETED
    assert second.status is StageStatus.CACHED
    assert calls == 1
    assert (context.artifacts_dir / "audio_analysis.cache.json").is_file()


def test_quality_analysis_disabled_clears_only_owned_scene_state(tmp_path: Path) -> None:
    context = ProjectContext(
        input_path=tmp_path / "media",
        workspace=tmp_path / "workspace",
        settings=Settings(),
        montage_settings=QuickMontageSettings(
            semantic_analysis=True,
            quality_analysis=False,
        ),
    )
    context.prepare()
    asset = _asset(tmp_path / "quality-source.mp4")
    scene = _scene(asset, score=83, start=0).model_copy(
        update={
            "metadata": {
                "quality_metrics": {"quality_score": 83},
                "technical_rejection_reasons": ["blur"],
                "unrelated": "preserved",
            }
        }
    )
    _seed_project(context, [asset], [scene])
    for name in ("quality_analysis.json", "quality_analysis.cache.json"):
        (context.artifacts_dir / name).write_text("stale", encoding="utf-8")

    result = QualityAnalysisStage().run(context)
    stored = MediaAssetRepository(context.database_path).list_scenes()[0]

    assert result.status is StageStatus.DISABLED
    assert stored.quality_score is None
    assert stored.metadata == {"unrelated": "preserved"}
    assert not (context.artifacts_dir / "quality_analysis.json").exists()
    assert not (context.artifacts_dir / "quality_analysis.cache.json").exists()


def test_quality_analysis_no_frames_clears_stale_owned_scene_state(tmp_path: Path) -> None:
    context = _context(tmp_path)
    asset = _asset(tmp_path / "quality-source.mp4")
    scene = _scene(asset, score=83, start=0).model_copy(
        update={
            "keyframe_path": None,
            "metadata": {
                "quality_metrics": {"quality_score": 83},
                "technical_rejection_reasons": ["blur"],
                "unrelated": "preserved",
            },
        }
    )
    _seed_project(context, [asset], [scene])
    (context.artifacts_dir / "quality_analysis.json").write_text("stale", encoding="utf-8")

    result = QualityAnalysisStage().run(context)
    stored = MediaAssetRepository(context.database_path).list_scenes()[0]

    assert result.status is StageStatus.NO_INPUT
    assert stored.quality_score is None
    assert stored.metadata == {"unrelated": "preserved"}
    assert not (context.artifacts_dir / "quality_analysis.json").exists()


def test_audio_analysis_disabled_preserves_non_owned_candidate_windows(tmp_path: Path) -> None:
    context = ProjectContext(
        input_path=tmp_path / "media",
        workspace=tmp_path / "workspace",
        settings=Settings(),
        montage_settings=QuickMontageSettings(
            semantic_analysis=True,
            audio_analysis=False,
        ),
    )
    context.prepare()
    asset = _asset(tmp_path / "audio-source.mp4")
    scene = _scene(asset, score=80, start=0).model_copy(
        update={
            "metadata": {
                "audio_analysis": {"primary_label": "speech"},
                "audio_context": ["speech"],
                "audio_features": {"speech_likelihood": 0.9},
                "candidate_windows": [
                    {"source": "audio_analysis", "start_ratio": 0.1},
                    {"source": "manual", "start_ratio": 0.4},
                ],
                "unrelated": "preserved",
            }
        }
    )
    _seed_project(context, [asset], [scene])

    result = AudioAnalysisStage().run(context)
    stored = MediaAssetRepository(context.database_path).list_scenes()[0]

    assert result.status is StageStatus.DISABLED
    assert stored.metadata == {
        "candidate_windows": [{"source": "manual", "start_ratio": 0.4}],
        "unrelated": "preserved",
    }


def test_duplicate_detection_disabled_clears_owned_scene_state(tmp_path: Path) -> None:
    context = ProjectContext(
        input_path=tmp_path / "media",
        workspace=tmp_path / "workspace",
        settings=Settings(),
        montage_settings=QuickMontageSettings(
            semantic_analysis=True,
            duplicate_detection=False,
        ),
    )
    context.prepare()
    asset = _asset(tmp_path / "duplicate-source.mp4")
    scene = _scene(asset, score=80, start=0).model_copy(
        update={
            "metadata": {
                "perceptual_hash": "0123456789abcdef",
                "duplicate_of": None,
                "duplicate_similarity": 1.0,
                "duplicate_status": "keeper",
                "unrelated": "preserved",
            }
        }
    )
    _seed_project(context, [asset], [scene])

    result = DuplicateDetectionStage().run(context)
    stored = MediaAssetRepository(context.database_path).list_scenes()[0]

    assert result.status is StageStatus.DISABLED
    assert stored.metadata == {"unrelated": "preserved"}


@pytest.mark.parametrize(
    ("stage", "artifact_names"),
    [
        (
            QualityAnalysisStage(),
            ("quality_analysis.json", "quality_analysis.cache.json"),
        ),
        (
            SpeechAnalysisStage(),
            ("speech_analysis.json", "speech_analysis.cache.json"),
        ),
        (
            AudioAnalysisStage(),
            ("audio_analysis.json", "audio_analysis.cache.json"),
        ),
        (
            DuplicateDetectionStage(),
            ("duplicates.json", "duplicates.cache.json"),
        ),
    ],
)
def test_optional_analysis_no_input_removes_stale_artifacts(
    tmp_path: Path,
    stage: object,
    artifact_names: tuple[str, str],
) -> None:
    context = _context(tmp_path)
    for name in artifact_names:
        (context.artifacts_dir / name).write_text("stale", encoding="utf-8")

    result = stage.run(context)

    assert result.status is StageStatus.NO_INPUT
    assert result.skipped is True
    assert all(not (context.artifacts_dir / name).exists() for name in artifact_names)


def test_timeline_builder_stage_reuses_cached_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context = _context(tmp_path)
    asset = _asset(tmp_path / "cached-opening.mp4")
    scene = _scene(asset, score=90, start=0)
    _seed_project(context, [asset], [scene])

    first = TimelineBuilderStage().run(context)

    def fail_build(*args: object, **kwargs: object) -> QuickMontagePlan:
        raise AssertionError("timeline should be reused")

    monkeypatch.setattr(timeline_builder, "build_semantic_montage_plan", fail_build)
    second = TimelineBuilderStage().run(context)

    assert first.skipped is False
    assert second.skipped is True
    assert (context.artifacts_dir / "quick_timeline.cache.json").is_file()


def test_timeline_builder_stage_rebuilds_corrupt_cached_artifact(tmp_path: Path) -> None:
    context = _context(tmp_path)
    asset = _asset(tmp_path / "corrupt-cache.mp4")
    scene = _scene(asset, score=90, start=0)
    _seed_project(context, [asset], [scene])
    TimelineBuilderStage().run(context)
    timeline_path = context.artifacts_dir / "quick_timeline.json"
    timeline_path.write_text("not json", encoding="utf-8")

    result = TimelineBuilderStage().run(context)
    plan = QuickMontagePlan.model_validate_json(timeline_path.read_text(encoding="utf-8"))

    assert result.skipped is False
    assert plan.clips[0].scene_id == scene.id


def test_timeline_builder_stage_rejects_pre_safe_transition_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context = _context(tmp_path)
    asset = _asset(tmp_path / "legacy-transition.mp4")
    scene = _scene(asset, score=90, start=0)
    _seed_project(context, [asset], [scene])
    TimelineBuilderStage().run(context)
    cache_path = context.artifacts_dir / "quick_timeline.cache.json"
    manifest = StageCacheManifest.model_validate_json(cache_path.read_text(encoding="utf-8"))
    timeline_builder.write_json_atomic(
        cache_path,
        manifest.model_copy(update={"artifact_schema_version": "timeline-builder-v3"}),
    )
    build = timeline_builder.build_semantic_montage_plan
    calls = 0

    def counting_build(
        assets: list[MediaAsset],
        scenes: list[Scene],
        settings: QuickMontageSettings,
        music_plan: MusicPlan | None = None,
    ) -> QuickMontagePlan:
        nonlocal calls
        calls += 1
        return build(assets, scenes, settings, music_plan)

    monkeypatch.setattr(timeline_builder, "build_semantic_montage_plan", counting_build)

    result = TimelineBuilderStage().run(context)

    assert result.skipped is False
    assert calls == 1


def test_timeline_builder_stage_invalidates_cache_when_scenes_change(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context = _context(tmp_path)
    asset = _asset(tmp_path / "changed-scenes.mp4")
    first_scene = _scene(asset, score=80, start=0)
    _seed_project(context, [asset], [first_scene])
    TimelineBuilderStage().run(context)
    second_scene = _scene(asset, score=95, start=6)
    _seed_project(context, [asset], [first_scene, second_scene])
    calls = 0

    def fake_build(
        assets: list[MediaAsset],
        scenes: list[Scene],
        settings: QuickMontageSettings,
        music_plan: MusicPlan | None = None,
    ) -> QuickMontagePlan:
        nonlocal calls
        calls += 1
        assert music_plan is None
        return _timeline_plan(assets[0], scenes[-1])

    monkeypatch.setattr(timeline_builder, "build_semantic_montage_plan", fake_build)

    result = TimelineBuilderStage().run(context)
    plan = QuickMontagePlan.model_validate_json(
        (context.artifacts_dir / "quick_timeline.json").read_text(encoding="utf-8")
    )

    assert result.skipped is False
    assert calls == 1
    assert plan.clips[0].scene_id == second_scene.id


def test_music_selection_stage_writes_music_plan_without_existing_timeline(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context = _context(tmp_path)
    asset = _asset(tmp_path / "clip.mp4")
    scene = _scene(asset, score=90, start=0)
    _seed_project(context, [asset], [scene])

    def fake_build_music_plan(*args: object, **kwargs: object) -> MusicPlan:
        montage_plan = args[5]
        assert isinstance(montage_plan, QuickMontagePlan)
        soundtrack = context.artifacts_dir / "theme.wav"
        soundtrack.write_bytes(b"fake wav")
        return MusicPlan(
            mode="generated",
            source_path=soundtrack,
            duration_seconds=montage_plan.total_duration_seconds,
            reasoning="fake music",
        )

    monkeypatch.setattr(music_selection, "build_music_plan", fake_build_music_plan)

    result = MusicSelectionStage().run(context)
    music_plan = MusicPlan.model_validate_json(
        (context.artifacts_dir / "music_plan.json").read_text(encoding="utf-8")
    )

    assert result.stage is PipelineStage.MUSIC_SELECTION
    assert result.skipped is False
    assert music_plan.mode == "generated"
    assert not (context.artifacts_dir / "quick_timeline.json").exists()


def test_music_selection_stage_passes_local_ai_generator(
    tmp_path: Path,
    monkeypatch,
) -> None:
    heartbeats: list[tuple[int, int, str]] = []
    context = replace(
        _context(tmp_path),
        progress=lambda current, total, message: heartbeats.append((current, total, message)),
    )
    asset = _asset(tmp_path / "clip.mp4")
    scene = _scene(asset, score=90, start=0)
    _seed_project(context, [asset], [scene])
    captured: dict[str, object] = {}

    class FakeGenerator:
        name = "ace-step"
        model = "fake-ace-step"

        def __init__(self, model: str, **kwargs: object) -> None:
            captured["model"] = model
            captured["options"] = kwargs

    def fake_build_music_plan(*args: object, **kwargs: object) -> MusicPlan:
        captured["generator"] = kwargs.get("neural_generator")
        soundtrack = context.artifacts_dir / "theme.wav"
        soundtrack.write_bytes(b"fake wav")
        return MusicPlan(
            mode="generated",
            source_path=soundtrack,
            duration_seconds=6,
            generator="ace-step",
            model="fake-ace-step",
        )

    monkeypatch.setattr(music_selection, "AceStepMusicGenerator", FakeGenerator)
    monkeypatch.setattr(
        music_selection,
        "detect_resource_profile",
        lambda *args, **kwargs: type("Profile", (), {"gpu_memory_mb": 6144})(),
    )
    monkeypatch.setattr(music_selection, "build_music_plan", fake_build_music_plan)

    result = MusicSelectionStage().run(context)

    assert isinstance(captured["generator"], FakeGenerator)
    assert captured["model"] == "ACE-Step/acestep-v15-turbo"
    options = captured["options"]
    assert isinstance(options, dict)
    cancel_requested = options["cancel_requested"]
    assert callable(cancel_requested)
    assert cancel_requested() is False
    assert heartbeats[-1] == (1, 4, "ACE-Step: generation is still running")
    assert result.status is StageStatus.COMPLETED
    assert result.execution.provider == "ace-step"
    assert result.execution.model == "fake-ace-step"
    assert result.execution.fallback_count == 0


def test_music_selection_stage_rejects_missing_generated_soundtrack(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context = _context(tmp_path)
    asset = _asset(tmp_path / "clip.mp4")
    scene = _scene(asset, score=90, start=0)
    _seed_project(context, [asset], [scene])

    def fake_build_music_plan(*args: object, **kwargs: object) -> MusicPlan:
        montage_plan = args[5]
        assert isinstance(montage_plan, QuickMontagePlan)
        return MusicPlan(
            mode="generated",
            source_path=context.artifacts_dir / "missing-theme.wav",
            duration_seconds=montage_plan.total_duration_seconds,
            reasoning="fake music",
        )

    monkeypatch.setattr(music_selection, "build_music_plan", fake_build_music_plan)

    with pytest.raises(music_selection.MontageError, match="without an available soundtrack"):
        MusicSelectionStage().run(context)

    assert not (context.artifacts_dir / "music_plan.json").exists()


def test_music_selection_stage_reuses_current_cache_and_rejects_legacy_schema(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context = _context(tmp_path)
    asset = _asset(tmp_path / "clip.mp4")
    scene = _scene(asset, score=90, start=0)
    _seed_project(context, [asset], [scene])
    calls = 0

    def fake_build_music_plan(*args: object, **kwargs: object) -> MusicPlan:
        nonlocal calls
        calls += 1
        montage_plan = args[5]
        assert isinstance(montage_plan, QuickMontagePlan)
        soundtrack = context.artifacts_dir / "cached-theme.wav"
        soundtrack.write_bytes(b"fake wav")
        return MusicPlan(
            mode="generated",
            source_path=soundtrack,
            duration_seconds=montage_plan.total_duration_seconds,
            generator="ace-step",
            model="cached-model",
        )

    monkeypatch.setattr(music_selection, "build_music_plan", fake_build_music_plan)

    first = MusicSelectionStage().run(context)
    second = MusicSelectionStage().run(context)

    assert first.skipped is False
    assert second.skipped is True
    assert first.status is StageStatus.COMPLETED
    assert first.execution.provider == "ace-step"
    assert first.execution.model == "cached-model"
    assert second.status is StageStatus.CACHED
    assert second.cache_hit is True
    assert second.execution.provider == "ace-step"
    assert second.execution.model == "cached-model"
    assert calls == 1

    (context.artifacts_dir / "cached-theme.wav").write_bytes(b"replaced soundtrack")
    replaced = MusicSelectionStage().run(context)

    assert replaced.status is StageStatus.COMPLETED
    assert replaced.cache_hit is False
    assert calls == 2
    cache_path = context.artifacts_dir / "music_plan.cache.json"
    manifest = StageCacheManifest.model_validate_json(cache_path.read_text(encoding="utf-8"))
    music_selection.write_json_atomic(
        cache_path,
        manifest.model_copy(update={"artifact_schema_version": "music-selection-v1"}),
    )

    legacy = MusicSelectionStage().run(context)

    assert legacy.skipped is False
    assert calls == 3


def test_music_selection_stage_reports_internal_composition_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context = _context(tmp_path)
    asset = _asset(tmp_path / "music-internal-cache.mp4")
    scene = _scene(asset, score=90, start=0)
    _seed_project(context, [asset], [scene])

    def fake_build_music_plan(*args: object, **kwargs: object) -> MusicPlan:
        execution = kwargs.get("execution")
        assert isinstance(execution, MusicPlanExecution)
        execution.cache_hit = True
        soundtrack = context.artifacts_dir / "reused-theme.wav"
        soundtrack.write_bytes(b"cached wav")
        return MusicPlan(
            mode="generated",
            source_path=soundtrack,
            duration_seconds=6,
            generator="ace-step",
            model="reused-model",
        )

    monkeypatch.setattr(music_selection, "build_music_plan", fake_build_music_plan)

    result = MusicSelectionStage().run(context)

    assert result.status is StageStatus.CACHED
    assert result.cache_hit is True
    assert result.skipped is True
    assert result.execution.provider == "ace-step"
    assert result.execution.model == "reused-model"
    assert result.execution.fallback_count == 0


def test_music_selection_stage_reports_non_sticky_procedural_fallback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context = _context(tmp_path)
    asset = _asset(tmp_path / "music-fallback.mp4")
    scene = _scene(asset, score=90, start=0)
    _seed_project(context, [asset], [scene])
    calls = 0

    def fake_build_music_plan(*args: object, **kwargs: object) -> MusicPlan:
        nonlocal calls
        calls += 1
        soundtrack = context.artifacts_dir / "fallback-theme.wav"
        soundtrack.write_bytes(b"procedural wav")
        return MusicPlan(
            mode="generated",
            source_path=soundtrack,
            duration_seconds=6,
            generator="procedural",
            fallback_used=True,
        )

    monkeypatch.setattr(music_selection, "build_music_plan", fake_build_music_plan)

    first = MusicSelectionStage().run(context)
    second = MusicSelectionStage().run(context)

    assert first.status is StageStatus.DEGRADED
    assert first.cache_hit is False
    assert first.execution.provider == "ace-step"
    assert first.execution.fallback_provider == "procedural"
    assert first.execution.fallback_count == 1
    assert second.status is StageStatus.DEGRADED
    assert second.cache_hit is False
    assert second.execution.provider == "ace-step"
    assert second.execution.fallback_provider == "procedural"
    assert second.execution.fallback_count == 1
    assert calls == 2
    assert not (context.artifacts_dir / "music_plan.cache.json").exists()


def test_music_selection_stage_rebuilds_after_disable_then_enable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context = _context(tmp_path)
    asset = _asset(tmp_path / "music-toggle.mp4")
    scene = _scene(asset, score=90, start=0)
    _seed_project(context, [asset], [scene])
    calls = 0

    def fake_build_music_plan(*args: object, **kwargs: object) -> MusicPlan:
        nonlocal calls
        calls += 1
        soundtrack = context.artifacts_dir / "toggle-theme.wav"
        soundtrack.write_bytes(b"fake wav")
        return MusicPlan(
            mode="generated",
            source_path=soundtrack,
            duration_seconds=6,
        )

    monkeypatch.setattr(music_selection, "build_music_plan", fake_build_music_plan)
    first = MusicSelectionStage().run(context)
    disabled_context = ProjectContext(
        input_path=context.input_path,
        workspace=context.workspace,
        settings=context.settings,
        montage_settings=QuickMontageSettings(music_enabled=False),
    )
    disabled = MusicSelectionStage().run(disabled_context)
    cache_path = context.artifacts_dir / "music_plan.cache.json"

    assert first.skipped is False
    assert disabled.skipped is True
    assert not cache_path.exists()

    enabled_again = MusicSelectionStage().run(context)

    assert enabled_again.skipped is False
    assert calls == 2


def test_music_selection_stage_skips_without_assets_or_scenes(tmp_path: Path) -> None:
    context = _context(tmp_path)

    result = MusicSelectionStage().run(context)

    assert result.stage is PipelineStage.MUSIC_SELECTION
    assert result.skipped is True
    assert not (context.artifacts_dir / "music_plan.json").exists()


def test_timeline_builder_stage_embeds_existing_music_plan(tmp_path: Path) -> None:
    context = _context(tmp_path)
    asset = _asset(tmp_path / "music-timeline.mp4")
    scene = _scene(asset, score=90, start=0)
    _seed_project(context, [asset], [scene])
    soundtrack = context.artifacts_dir / "theme.wav"
    soundtrack.write_bytes(b"fake wav")
    music_plan = MusicPlan(
        mode="generated",
        source_path=soundtrack,
        duration_seconds=6,
        reasoning="fake music",
    )
    (context.artifacts_dir / "music_plan.json").write_text(
        music_plan.model_dump_json(),
        encoding="utf-8",
    )

    result = TimelineBuilderStage().run(context)
    plan = QuickMontagePlan.model_validate_json(
        (context.artifacts_dir / "quick_timeline.json").read_text(encoding="utf-8")
    )

    assert result.stage is PipelineStage.TIMELINE_BUILDER
    assert result.skipped is False
    assert plan.music_path == music_plan.source_path
    assert plan.music_plan == music_plan


def test_timeline_builder_invalidates_cache_when_music_file_is_replaced(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    asset = _asset(tmp_path / "music-revision.mp4")
    scene = _scene(asset, score=90, start=0)
    _seed_project(context, [asset], [scene])
    soundtrack = context.artifacts_dir / "replaceable-theme.wav"
    soundtrack.write_bytes(b"first soundtrack")
    timeline_builder.write_json_atomic(
        context.artifacts_dir / "music_plan.json",
        MusicPlan(mode="manual", source_path=soundtrack, duration_seconds=6),
    )

    first = TimelineBuilderStage().run(context)
    cached = TimelineBuilderStage().run(context)
    soundtrack.write_bytes(b"a different soundtrack revision")
    changed = TimelineBuilderStage().run(context)

    assert first.skipped is False
    assert cached.skipped is True
    assert changed.skipped is False


def test_timeline_builder_stage_rejects_missing_music_source(tmp_path: Path) -> None:
    context = _context(tmp_path)
    asset = _asset(tmp_path / "missing-music.mp4")
    scene = _scene(asset, score=90, start=0)
    _seed_project(context, [asset], [scene])
    music_plan = MusicPlan(
        mode="generated",
        source_path=context.artifacts_dir / "missing-theme.wav",
        duration_seconds=6,
        reasoning="fake music",
    )
    (context.artifacts_dir / "music_plan.json").write_text(
        music_plan.model_dump_json(),
        encoding="utf-8",
    )

    with pytest.raises(timeline_builder.MontageError, match="missing soundtrack file"):
        TimelineBuilderStage().run(context)

    assert not (context.artifacts_dir / "quick_timeline.json").exists()


def test_rendering_stage_renders_timeline_and_writes_quality_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context = _context(tmp_path)
    asset = _asset(tmp_path / "clip.mp4")
    scene = _scene(asset, score=90, start=0)
    _seed_project(context, [asset], [scene])
    plan = _timeline_plan(asset, scene)
    (context.artifacts_dir / "quick_timeline.json").write_text(
        plan.model_dump_json(),
        encoding="utf-8",
    )
    output_path = context.artifacts_dir / "final.mp4"
    output_path.write_bytes(b"previous movie")
    render_targets: list[Path] = []

    class FakeRenderer:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def render(
            self,
            montage_plan: QuickMontagePlan,
            output_path: Path,
            work_dir: Path,
        ) -> str:
            assert montage_plan == plan
            assert work_dir == context.cache_dir
            assert output_path.parent == (context.artifacts_dir / "final.mp4").parent
            assert output_path != context.artifacts_dir / "final.mp4"
            assert output_path.name.startswith(".final.")
            assert (context.artifacts_dir / "final.mp4").read_bytes() == b"previous movie"
            render_targets.append(output_path)
            output_path.write_bytes(b"fake mp4")
            return "fake-encoder"

    def fake_detect_resource_profile(*args: object, **kwargs: object) -> object:
        return type("Profile", (), {"render_workers": 2, "ffmpeg_threads": 3})()

    def fake_enrich(
        report: MontageQualityReport,
        output_path: Path,
        **kwargs: object,
    ) -> MontageQualityReport:
        return report.model_copy(
            update={
                "rendered_path": output_path,
                "rendered_duration_seconds": report.planned_duration_seconds,
                "rendered_has_video": True,
                "rendered_has_audio": True,
            }
        )

    monkeypatch.setattr(rendering, "QuickMontageRenderer", FakeRenderer)
    monkeypatch.setattr(rendering, "detect_resource_profile", fake_detect_resource_profile)
    monkeypatch.setattr(rendering, "enrich_montage_quality_report_with_render", fake_enrich)

    result = RenderingStage().run(context)
    quality = MontageQualityReport.model_validate_json(
        (context.artifacts_dir / "montage_quality_report.json").read_text(encoding="utf-8")
    )

    assert result.stage is PipelineStage.RENDERING
    assert result.skipped is False
    assert (context.artifacts_dir / "final.mp4").read_bytes() == b"fake mp4"
    assert quality.rendered_path == context.artifacts_dir / "final.mp4"
    assert quality.rendered_has_video is True
    assert quality.rendered_has_audio is True
    assert len(render_targets) == 1
    assert not render_targets[0].exists()


def test_rendering_stage_preserves_existing_delivery_when_quality_gate_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path)
    asset = _asset(tmp_path / "clip.mp4")
    scene = _scene(asset, score=90, start=0)
    _seed_project(context, [asset], [scene])
    plan = _timeline_plan(asset, scene)
    timeline_artifact = context.artifacts_dir / "quick_timeline.json"
    timeline_artifact.write_text(plan.model_dump_json(), encoding="utf-8")
    output_path = context.artifacts_dir / "final.mp4"
    quality_artifact = context.artifacts_dir / "montage_quality_report.json"
    cache_artifact = context.artifacts_dir / "rendering.cache.json"
    output_path.write_bytes(b"previous movie")
    quality_artifact.write_bytes(b"previous quality report")
    cache_artifact.write_bytes(b"previous cache manifest")
    render_targets: list[Path] = []

    class FakeRenderer:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def render(
            self,
            montage_plan: QuickMontagePlan,
            candidate_path: Path,
            work_dir: Path,
        ) -> str:
            del montage_plan, work_dir
            render_targets.append(candidate_path)
            candidate_path.write_bytes(b"rejected movie")
            return "fake-encoder"

    def fake_detect_resource_profile(*args: object, **kwargs: object) -> object:
        return type("Profile", (), {"render_workers": 1, "ffmpeg_threads": 1})()

    def fake_enrich(
        report: MontageQualityReport,
        candidate_path: Path,
        **kwargs: object,
    ) -> MontageQualityReport:
        return report.model_copy(
            update={
                "rendered_path": candidate_path,
                "rendered_duration_seconds": report.planned_duration_seconds,
                "rendered_has_video": True,
                "rendered_has_audio": True,
            }
        )

    def reject_quality(report: MontageQualityReport) -> None:
        assert report.rendered_path == render_targets[0]
        assert output_path.read_bytes() == b"previous movie"
        raise rendering.MontageError("simulated quality gate failure")

    monkeypatch.setattr(rendering, "QuickMontageRenderer", FakeRenderer)
    monkeypatch.setattr(rendering, "detect_resource_profile", fake_detect_resource_profile)
    monkeypatch.setattr(rendering, "enrich_montage_quality_report_with_render", fake_enrich)
    monkeypatch.setattr(rendering, "enforce_montage_quality", reject_quality)

    with pytest.raises(rendering.MontageError, match="quality gate failure"):
        RenderingStage().run(context)

    assert output_path.read_bytes() == b"previous movie"
    assert quality_artifact.read_bytes() == b"previous quality report"
    assert cache_artifact.read_bytes() == b"previous cache manifest"
    assert timeline_artifact.read_text(encoding="utf-8") == plan.model_dump_json()
    assert len(render_targets) == 1
    assert not render_targets[0].exists()


def test_rendering_stage_reuses_cached_movie_and_quality_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context = _context(tmp_path)
    asset = _asset(tmp_path / "clip.mp4")
    scene = _scene(asset, score=90, start=0)
    _seed_project(context, [asset], [scene])
    soundtrack = context.artifacts_dir / "render-theme.wav"
    soundtrack.write_bytes(b"first soundtrack")
    plan = _timeline_plan(asset, scene).model_copy(update={"music_path": soundtrack})
    (context.artifacts_dir / "quick_timeline.json").write_text(
        plan.model_dump_json(),
        encoding="utf-8",
    )
    calls = 0

    class FakeRenderer:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def render(
            self,
            montage_plan: QuickMontagePlan,
            output_path: Path,
            work_dir: Path,
        ) -> str:
            nonlocal calls
            calls += 1
            output_path.write_bytes(b"fake mp4")
            return "fake-encoder"

    def fake_detect_resource_profile(*args: object, **kwargs: object) -> object:
        return type("Profile", (), {"render_workers": 1, "ffmpeg_threads": 1})()

    class FakeFFprobeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def probe(self, path: Path) -> object:
            return type(
                "Probe",
                (),
                {
                    "duration_seconds": 3.0,
                    "width": 1280,
                    "height": 720,
                    "fps": 30.0,
                    "metadata": {
                        "streams": [
                            {"codec_type": "video", "pix_fmt": "yuv420p"},
                            {"codec_type": "audio"},
                        ],
                    },
                },
            )()

    def fake_enrich(
        report: MontageQualityReport,
        output_path: Path,
        **kwargs: object,
    ) -> MontageQualityReport:
        return report.model_copy(
            update={
                "rendered_path": output_path,
                "rendered_duration_seconds": report.planned_duration_seconds,
                "rendered_has_video": True,
                "rendered_has_audio": True,
            }
        )

    monkeypatch.setattr(rendering, "QuickMontageRenderer", FakeRenderer)
    monkeypatch.setattr(rendering, "detect_resource_profile", fake_detect_resource_profile)
    monkeypatch.setattr(rendering, "enrich_montage_quality_report_with_render", fake_enrich)
    monkeypatch.setattr(rendering, "FFprobeClient", FakeFFprobeClient)

    first = RenderingStage().run(context)
    second = RenderingStage().run(context)
    soundtrack.write_bytes(b"replacement soundtrack with another revision")
    third = RenderingStage().run(context)

    assert first.skipped is False
    assert second.status is StageStatus.DEGRADED
    assert "cached movie with quality warnings" in second.message
    assert third.skipped is False
    assert calls == 2
    assert (context.artifacts_dir / "rendering.cache.json").is_file()


def test_rendering_stage_rerenders_after_renderer_behavior_change(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context = _context(tmp_path)
    asset = _asset(tmp_path / "clip.mp4")
    scene = _scene(asset, score=90, start=0)
    _seed_project(context, [asset], [scene])
    plan = _timeline_plan(asset, scene)
    timeline_artifact = context.artifacts_dir / "quick_timeline.json"
    timeline_artifact.write_text(plan.model_dump_json(), encoding="utf-8")
    output_path = context.artifacts_dir / "final.mp4"
    quality_artifact = context.artifacts_dir / "montage_quality_report.json"
    calls = 0

    class FakeRenderer:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def render(
            self,
            montage_plan: QuickMontagePlan,
            output_path: Path,
            work_dir: Path,
        ) -> str:
            nonlocal calls
            calls += 1
            output_path.write_bytes(b"fake mp4")
            return "fake-encoder"

    class FakeFFprobeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def probe(self, path: Path) -> object:
            return type(
                "Probe",
                (),
                {
                    "duration_seconds": 3.0,
                    "width": 1280,
                    "height": 720,
                    "fps": 30.0,
                    "metadata": {
                        "streams": [
                            {"codec_type": "video", "pix_fmt": "yuv420p"},
                            {"codec_type": "audio"},
                        ],
                    },
                },
            )()

    def fake_detect_resource_profile(*args: object, **kwargs: object) -> object:
        return type("Profile", (), {"render_workers": 1, "ffmpeg_threads": 1})()

    def fake_enrich(
        report: MontageQualityReport,
        output_path: Path,
        **kwargs: object,
    ) -> MontageQualityReport:
        return report.model_copy(
            update={
                "rendered_path": output_path,
                "rendered_duration_seconds": report.planned_duration_seconds,
                "rendered_has_video": True,
                "rendered_has_audio": True,
            }
        )

    monkeypatch.setattr(rendering, "QuickMontageRenderer", FakeRenderer)
    monkeypatch.setattr(rendering, "detect_resource_profile", fake_detect_resource_profile)
    monkeypatch.setattr(rendering, "enrich_montage_quality_report_with_render", fake_enrich)
    monkeypatch.setattr(rendering, "FFprobeClient", FakeFFprobeClient)

    first = RenderingStage().run(context)
    old_config_fingerprint = rendering.artifact_fingerprint(
        {
            "ffmpeg_binary": context.settings.ffmpeg_binary,
            "ffprobe_binary": context.settings.ffprobe_binary,
            "output_path": output_path.resolve(),
            "workers": context.settings.workers,
            "batch_size": context.settings.batch_size,
            "render_timeout_seconds": context.settings.render_timeout_seconds,
            "renderer_behavior": "transitions-quality-gate-v4",
            "schema": rendering.ARTIFACT_SCHEMA_VERSION,
        }
    )
    rendering.write_stage_cache_manifest(
        context.artifacts_dir / "rendering.cache.json",
        stage=PipelineStage.RENDERING,
        artifact_schema_version=rendering.ARTIFACT_SCHEMA_VERSION,
        input_fingerprint=rendering.artifact_fingerprint(plan),
        config_fingerprint=old_config_fingerprint,
        artifacts=[output_path, quality_artifact],
    )
    second = RenderingStage().run(context)

    assert first.skipped is False
    assert second.skipped is False
    assert calls == 2


def test_rendering_stage_ignores_legacy_render_cache_schema(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context = _context(tmp_path)
    asset = _asset(tmp_path / "clip.mp4")
    scene = _scene(asset, score=90, start=0)
    _seed_project(context, [asset], [scene])
    plan = _timeline_plan(asset, scene)
    (context.artifacts_dir / "quick_timeline.json").write_text(
        plan.model_dump_json(),
        encoding="utf-8",
    )
    calls = 0

    class FakeRenderer:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def render(
            self,
            montage_plan: QuickMontagePlan,
            output_path: Path,
            work_dir: Path,
        ) -> str:
            nonlocal calls
            calls += 1
            output_path.write_bytes(b"fake mp4")
            return "fake-encoder"

    class FakeFFprobeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def probe(self, path: Path) -> object:
            return type(
                "Probe",
                (),
                {
                    "duration_seconds": 3.0,
                    "width": 1280,
                    "height": 720,
                    "fps": 30.0,
                    "metadata": {
                        "streams": [
                            {"codec_type": "video", "pix_fmt": "yuv420p"},
                            {"codec_type": "audio"},
                        ],
                    },
                },
            )()

    def fake_detect_resource_profile(*args: object, **kwargs: object) -> object:
        return type("Profile", (), {"render_workers": 1, "ffmpeg_threads": 1})()

    def fake_enrich(
        report: MontageQualityReport,
        output_path: Path,
        **kwargs: object,
    ) -> MontageQualityReport:
        return report.model_copy(
            update={
                "rendered_path": output_path,
                "rendered_duration_seconds": report.planned_duration_seconds,
                "rendered_has_video": True,
                "rendered_has_audio": True,
            }
        )

    monkeypatch.setattr(rendering, "QuickMontageRenderer", FakeRenderer)
    monkeypatch.setattr(rendering, "detect_resource_profile", fake_detect_resource_profile)
    monkeypatch.setattr(rendering, "enrich_montage_quality_report_with_render", fake_enrich)
    monkeypatch.setattr(rendering, "FFprobeClient", FakeFFprobeClient)

    first = RenderingStage().run(context)
    rendering.write_stage_cache_manifest(
        context.artifacts_dir / "rendering.cache.json",
        stage=PipelineStage.RENDERING,
        artifact_schema_version="rendering-v2",
        input_fingerprint="a" * 64,
        config_fingerprint="b" * 64,
        artifacts=[
            context.artifacts_dir / "final.mp4",
            context.artifacts_dir / "montage_quality_report.json",
        ],
    )
    second = RenderingStage().run(context)

    assert first.skipped is False
    assert second.skipped is False
    assert calls == 2


def test_rendering_stage_rerenders_when_cached_movie_fails_probe(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context = _context(tmp_path)
    asset = _asset(tmp_path / "clip.mp4")
    scene = _scene(asset, score=90, start=0)
    _seed_project(context, [asset], [scene])
    plan = _timeline_plan(asset, scene)
    (context.artifacts_dir / "quick_timeline.json").write_text(
        plan.model_dump_json(),
        encoding="utf-8",
    )
    calls = 0

    class FakeRenderer:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def render(
            self,
            montage_plan: QuickMontagePlan,
            output_path: Path,
            work_dir: Path,
        ) -> str:
            nonlocal calls
            calls += 1
            output_path.write_bytes(b"fake mp4")
            return "fake-encoder"

    class FailingFFprobeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def probe(self, path: Path) -> object:
            raise rendering.TravelMovieError("cached movie is corrupt")

    def fake_detect_resource_profile(*args: object, **kwargs: object) -> object:
        return type("Profile", (), {"render_workers": 1, "ffmpeg_threads": 1})()

    def fake_enrich(
        report: MontageQualityReport,
        output_path: Path,
        **kwargs: object,
    ) -> MontageQualityReport:
        return report.model_copy(
            update={
                "rendered_path": output_path,
                "rendered_duration_seconds": report.planned_duration_seconds,
                "rendered_has_video": True,
                "rendered_has_audio": True,
            }
        )

    monkeypatch.setattr(rendering, "QuickMontageRenderer", FakeRenderer)
    monkeypatch.setattr(rendering, "detect_resource_profile", fake_detect_resource_profile)
    monkeypatch.setattr(rendering, "enrich_montage_quality_report_with_render", fake_enrich)
    monkeypatch.setattr(rendering, "FFprobeClient", FailingFFprobeClient)

    first = RenderingStage().run(context)
    second = RenderingStage().run(context)

    assert first.skipped is False
    assert second.skipped is False
    assert calls == 2


def test_rendering_stage_skips_empty_timeline(tmp_path: Path) -> None:
    context = _context(tmp_path)
    plan = QuickMontagePlan(
        created_at=datetime.now(UTC),
        settings=QuickMontageSettings(),
    )
    (context.artifacts_dir / "quick_timeline.json").write_text(
        plan.model_dump_json(),
        encoding="utf-8",
    )

    result = RenderingStage().run(context)

    assert result.stage is PipelineStage.RENDERING
    assert result.skipped is True
    assert not (context.artifacts_dir / "final.mp4").exists()


def test_rendering_stage_checks_disk_space_before_starting_renderer(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context = _context(tmp_path)
    asset = _asset(tmp_path / "clip.mp4")
    scene = _scene(asset, score=90, start=0)
    _seed_project(context, [asset], [scene])
    plan = _timeline_plan(asset, scene)
    (context.artifacts_dir / "quick_timeline.json").write_text(
        plan.model_dump_json(),
        encoding="utf-8",
    )
    preflight: dict[str, object] = {}

    def reject_render(**kwargs: object) -> None:
        preflight.update(kwargs)
        raise rendering.MontageError("Not enough free disk space for rendering.")

    class UnexpectedRenderer:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("renderer must not start after a failed disk preflight")

    monkeypatch.setattr(rendering, "ensure_render_disk_space", reject_render)
    monkeypatch.setattr(rendering, "QuickMontageRenderer", UnexpectedRenderer)

    with pytest.raises(rendering.MontageError, match="Not enough free disk space"):
        RenderingStage().run(context)

    assert preflight["plan"] == plan
    assert not (context.artifacts_dir / "final.mp4").exists()


def test_story_builder_stage_applies_story_metadata_to_scenes(tmp_path: Path) -> None:
    context = _context(tmp_path)
    asset = _asset(tmp_path / "arrival.mp4")
    event_id = uuid4()
    scene = _scene(asset, score=88, start=0, event_id=str(event_id))
    event = Event(
        id=event_id,
        title="Arrival",
        scene_ids=[scene.id],
        summary="Arrival at the destination.",
        importance_score=88,
        start_at=datetime(2026, 1, 1, tzinfo=UTC),
        end_at=datetime(2026, 1, 1, 0, 0, 6, tzinfo=UTC),
        location_type=LocationType.AIRPORT,
        activity=ActivityType.ARRIVING,
        confidence=0.9,
    )
    _seed_project(context, [asset], [scene], [event])

    result = StoryBuilderStage().run(context)
    repository = MediaAssetRepository(context.database_path)
    stored = repository.list_scenes()[0]
    repository.synchronize_scenes(
        [
            stored.model_copy(
                update={
                    "metadata": {
                        **stored.metadata,
                        "story_section_index": 99,
                        "story_section_role": "finale",
                        "story_section_title": "Tampered",
                        "story_role_order": 99,
                        "unrelated_metadata": "preserve",
                    }
                }
            )
        ]
    )
    cached = StoryBuilderStage().run(context)
    restored = repository.list_scenes()[0]

    assert result.stage is PipelineStage.STORY_BUILDER
    assert result.skipped is False
    assert cached.skipped is True
    assert restored.metadata["story_section_index"] == 0
    assert restored.metadata["story_section_role"] == "opening"
    assert restored.metadata["story_section_title"] == "Arrival"
    assert restored.metadata["story_role_order"] == 0
    assert restored.metadata["unrelated_metadata"] == "preserve"


def test_narration_stage_writes_and_reuses_story_text(tmp_path: Path) -> None:
    context = _context(tmp_path)
    asset = _asset(tmp_path / "arrival.mp4")
    event_id = uuid4()
    scene = _scene(asset, score=88, start=0, event_id=str(event_id))
    event = Event(
        id=event_id,
        title="Arrival",
        scene_ids=[scene.id],
        summary="Arrival at the destination.",
        importance_score=88,
        start_at=datetime(2026, 1, 1, tzinfo=UTC),
        end_at=datetime(2026, 1, 1, 0, 0, 6, tzinfo=UTC),
        location_type=LocationType.AIRPORT,
        activity=ActivityType.ARRIVING,
        confidence=0.9,
    )
    _seed_project(context, [asset], [scene], [event])
    StoryBuilderStage().run(context)

    first = NarrationStage().run(context)
    storyboard_path = context.artifacts_dir / "storyboard.json"
    storyboard = Storyboard.model_validate_json(storyboard_path.read_text(encoding="utf-8"))
    timeline_builder.write_json_atomic(
        storyboard_path,
        storyboard.model_copy(update={"narration": []}),
    )
    second = NarrationStage().run(context)
    report = NarrationReport.model_validate_json(
        (context.artifacts_dir / "narration.json").read_text(encoding="utf-8")
    )

    assert first.skipped is False
    assert second.skipped is True
    assert report.lines[0].section_role == "opening"
    restored = Storyboard.model_validate_json(storyboard_path.read_text(encoding="utf-8"))
    assert restored.narration == [line.text for line in report.lines]


def test_timeline_builder_stage_skips_without_assets_or_scenes(tmp_path: Path) -> None:
    context = _context(tmp_path)
    context.prepare()
    for name in (
        "quick_timeline.json",
        "selection_decisions.json",
        "quick_timeline.cache.json",
    ):
        (context.artifacts_dir / name).write_text("stale", encoding="utf-8")

    result = TimelineBuilderStage().run(context)

    assert result.stage is PipelineStage.TIMELINE_BUILDER
    assert result.skipped is True
    assert not (context.artifacts_dir / "quick_timeline.json").exists()
    assert not (context.artifacts_dir / "selection_decisions.json").exists()
    assert not (context.artifacts_dir / "quick_timeline.cache.json").exists()


def _context(tmp_path: Path, *, settings: Settings | None = None) -> ProjectContext:
    context = ProjectContext(
        input_path=tmp_path / "media",
        workspace=tmp_path / "workspace",
        settings=settings or Settings(),
    )
    context.prepare()
    return context


def _seed_project(
    context: ProjectContext,
    assets: list[MediaAsset],
    scenes: list[Scene],
    events: list[Event] | None = None,
) -> None:
    repository = MediaAssetRepository(context.database_path)
    repository.initialize()
    repository.synchronize(assets, datetime.now(UTC))
    repository.synchronize_scenes(scenes)
    if events is not None:
        repository.synchronize_events(events)


def _asset(path: Path) -> MediaAsset:
    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    return MediaAsset(
        path=path,
        relative_path=Path(path.name),
        media_type=MediaType.VIDEO,
        extension=".mp4",
        size_bytes=1,
        modified_at=created_at,
        modified_ns=1,
        created_at=created_at,
        duration_seconds=8,
    )


def _timeline_plan(asset: MediaAsset, scene: Scene) -> QuickMontagePlan:
    return QuickMontagePlan(
        created_at=datetime.now(UTC),
        settings=QuickMontageSettings(
            semantic_analysis=True,
            target_duration_seconds=6,
            transition="none",
        ),
        clips=[
            MontageClip(
                asset_id=asset.id,
                source_path=asset.path,
                relative_path=asset.relative_path,
                media_type=asset.media_type,
                source_start_seconds=scene.start_seconds,
                duration_seconds=3,
                scene_id=scene.id,
                semantic_score=90,
                selection_reason="vision 90",
            )
        ],
        total_duration_seconds=3,
        selection_mode="semantic",
    )


def _scene(asset: MediaAsset, *, score: float, start: float, **metadata: object) -> Scene:
    event_id = uuid4()
    return Scene(
        asset_id=asset.id,
        start_seconds=start,
        end_seconds=start + 6,
        quality_score=80,
        importance_score=score,
        caption=asset.relative_path.stem,
        metadata={
            "event_id": str(event_id),
            "event_importance": score,
            "location_type": "city",
            "activity": "walking",
            "emotion": "joyful",
            **metadata,
        },
    )
