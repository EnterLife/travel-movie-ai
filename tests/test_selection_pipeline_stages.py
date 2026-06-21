from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from travelmovieai.application.context import ProjectContext
from travelmovieai.core.config import Settings
from travelmovieai.domain.enums import ActivityType, LocationType, MediaType, PipelineStage
from travelmovieai.domain.models import (
    AudioAnalysisReport,
    AudioSceneAnalysis,
    Event,
    MediaAsset,
    MontageClip,
    MontageQualityReport,
    MusicPlan,
    QualityAnalysisReport,
    QuickMontagePlan,
    QuickMontageSettings,
    Scene,
    SceneDetectionReport,
    SceneSelectionReport,
    SpeechAnalysisReport,
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
from travelmovieai.pipeline.stages.frame_sampling import FrameSamplingStage
from travelmovieai.pipeline.stages.music_selection import MusicSelectionStage
from travelmovieai.pipeline.stages.quality_analysis import QualityAnalysisStage
from travelmovieai.pipeline.stages.rendering import RenderingStage
from travelmovieai.pipeline.stages.scene_ranking import SceneRankingStage
from travelmovieai.pipeline.stages.speech_analysis import SpeechAnalysisStage
from travelmovieai.pipeline.stages.storyboard import StoryBuilderStage
from travelmovieai.pipeline.stages.timeline_builder import TimelineBuilderStage
from travelmovieai.pipeline.stages.vision_analysis import VisionAnalysisStage


def test_scene_ranking_stage_persists_ranked_scenes(tmp_path: Path) -> None:
    context = _context(tmp_path)
    asset = _asset(tmp_path / "clip.mp4")
    weaker = _scene(asset, score=45, start=0)
    stronger = _scene(asset, score=90, start=4)
    _seed_project(context, [asset], [weaker, stronger])

    result = SceneRankingStage().run(context)
    report = SceneDetectionReport.model_validate_json(
        (context.artifacts_dir / "ranked_scenes.json").read_text(encoding="utf-8")
    )
    stored = {
        scene.id: scene for scene in MediaAssetRepository(context.database_path).list_scenes()
    }

    assert result.stage is PipelineStage.SCENE_RANKING
    assert result.skipped is False
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
    assert [clip.scene_id for clip in plan.clips] == [scenes[0].id, scenes[1].id]
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
            pass

        def extract(self, source_scene: Scene, media_asset: MediaAsset, frames_dir: Path) -> Path:
            nonlocal calls
            calls += 1
            output = frames_dir / f"{source_scene.id}.png"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"png")
            return output

    monkeypatch.setattr(frame_sampling, "RepresentativeFrameExtractor", FakeExtractor)
    monkeypatch.setattr(
        frame_sampling, "check_cuda", lambda *args: type("Cuda", (), {"available": False})()
    )

    first = FrameSamplingStage().run(context)
    second = FrameSamplingStage().run(context)

    assert first.skipped is False
    assert second.skipped is True
    assert calls == 1
    assert (context.artifacts_dir / "frame_sampling.cache.json").is_file()


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

    def fake_analyze(scenes: list[Scene]) -> QualityAnalysisReport:
        nonlocal calls
        calls += 1
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

    first = QualityAnalysisStage().run(context)
    second = QualityAnalysisStage().run(context)

    assert first.skipped is False
    assert second.skipped is True
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

    def fake_profile(*args: object, **kwargs: object) -> object:
        return type(
            "Profile",
            (),
            {"gpu_memory_mb": None, "memory_mb": None, "model_batch_size": 1},
        )()

    def fake_provider(*args: object, **kwargs: object) -> object:
        return type("Provider", (), {"name": "fake-vision", "model": "fake-model"})()

    def fake_analyze(
        scenes: list[Scene],
        provider: object,
        style: object,
    ) -> VisionAnalysisReport:
        nonlocal calls
        calls += 1
        updated = [
            item.model_copy(
                update={
                    "caption": "Cached view",
                    "importance_score": 88,
                    "metadata": {**item.metadata, "vision_cache_key": "fake"},
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
    second = VisionAnalysisStage().run(context)

    assert first.skipped is False
    assert second.skipped is True
    assert calls == 1
    assert (context.artifacts_dir / "vision_analysis.cache.json").is_file()


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
    ) -> SpeechAnalysisReport:
        nonlocal calls
        calls += 1
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
    assert calls == 1
    assert (context.artifacts_dir / "speech_analysis.cache.json").is_file()


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
    ) -> AudioAnalysisReport:
        nonlocal calls
        calls += 1
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
    assert calls == 1
    assert (context.artifacts_dir / "audio_analysis.cache.json").is_file()


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
        return MusicPlan(
            mode="generated",
            source_path=context.artifacts_dir / "theme.wav",
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


def test_music_selection_stage_reuses_cached_artifacts(
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
        return MusicPlan(mode="none", duration_seconds=montage_plan.total_duration_seconds)

    monkeypatch.setattr(music_selection, "build_music_plan", fake_build_music_plan)

    first = MusicSelectionStage().run(context)
    second = MusicSelectionStage().run(context)

    assert first.skipped is False
    assert second.skipped is True
    assert calls == 1
    assert (context.artifacts_dir / "music_plan.cache.json").is_file()


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
    music_plan = MusicPlan(
        mode="generated",
        source_path=context.artifacts_dir / "theme.wav",
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
    assert quality.rendered_has_video is True
    assert quality.rendered_has_audio is True


def test_rendering_stage_reuses_cached_movie_and_quality_report(
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
                    "metadata": {
                        "streams": [{"codec_type": "video"}, {"codec_type": "audio"}],
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

    assert first.skipped is False
    assert second.skipped is True
    assert calls == 1
    assert (context.artifacts_dir / "rendering.cache.json").is_file()


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
    stored = MediaAssetRepository(context.database_path).list_scenes()[0]

    assert result.stage is PipelineStage.STORY_BUILDER
    assert result.skipped is False
    assert stored.metadata["story_section_role"] == "opening"
    assert stored.metadata["story_role_order"] == 0


def test_timeline_builder_stage_skips_without_assets_or_scenes(tmp_path: Path) -> None:
    context = _context(tmp_path)
    context.prepare()

    result = TimelineBuilderStage().run(context)

    assert result.stage is PipelineStage.TIMELINE_BUILDER
    assert result.skipped is True
    assert not (context.artifacts_dir / "quick_timeline.json").exists()


def _context(tmp_path: Path) -> ProjectContext:
    context = ProjectContext(
        input_path=tmp_path / "media",
        workspace=tmp_path / "workspace",
        settings=Settings(),
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
