from pathlib import Path

from travelmovieai.application.context import ProjectContext
from travelmovieai.application.service import TravelMovieService
from travelmovieai.core.config import Settings
from travelmovieai.core.exceptions import PipelineStageError
from travelmovieai.domain.enums import PipelineStage, StageStatus, StoryStyle
from travelmovieai.domain.models import QuickMontageResult, QuickMontageSettings, StageResult
from travelmovieai.pipeline.base import Stage
from travelmovieai.pipeline.registry import build_default_pipeline
from travelmovieai.pipeline.runner import PipelineRunner


def test_default_pipeline_matches_specification_order() -> None:
    pipeline = build_default_pipeline()

    assert [stage.name for stage in pipeline] == list(PipelineStage)
    concrete = {stage.name: type(stage).__name__ for stage in pipeline}
    assert concrete[PipelineStage.EMBEDDINGS] == "EmbeddingsStage"
    assert concrete[PipelineStage.NARRATION] == "NarrationStage"
    assert concrete[PipelineStage.VOICE_SYNTHESIS] == "VoiceSynthesisStage"


def test_pipeline_stops_at_requested_stage(tmp_path: Path) -> None:
    input_path = tmp_path / "media"
    input_path.mkdir()
    context = ProjectContext(
        input_path=input_path,
        workspace=tmp_path / "workspace",
        settings=Settings(),
    )

    result = PipelineRunner(build_default_pipeline()).run_until(
        context, PipelineStage.FRAME_SAMPLING
    )

    assert result.stage == PipelineStage.FRAME_SAMPLING
    assert result.skipped is True
    assert result.status is StageStatus.NO_INPUT
    assert [entry.stage for entry in result.trace] == [
        PipelineStage.MEDIA_SCAN,
        PipelineStage.SCENE_DETECTION,
        PipelineStage.FRAME_SAMPLING,
    ]


def test_pipeline_runner_exposes_typed_result_trace(tmp_path: Path) -> None:
    context = ProjectContext(
        input_path=tmp_path / "media",
        workspace=tmp_path / "workspace",
        settings=Settings(),
    )
    runner = PipelineRunner(
        [
            _StubStage(PipelineStage.MEDIA_SCAN, StageStatus.COMPLETED),
            _StubStage(PipelineStage.SCENE_DETECTION, StageStatus.CACHED),
        ]
    )

    result = runner.run_until(context, PipelineStage.SCENE_DETECTION)

    assert result.status is StageStatus.CACHED
    assert result.skipped is True
    assert [entry.status for entry in result.trace] == [
        StageStatus.COMPLETED,
        StageStatus.CACHED,
    ]
    assert runner.last_trace == tuple(result.trace)
    assert all(not entry.trace for entry in result.trace)


def test_pipeline_runner_rejects_mismatched_stage_result(tmp_path: Path) -> None:
    context = ProjectContext(
        input_path=tmp_path / "media",
        workspace=tmp_path / "workspace",
        settings=Settings(),
    )
    runner = PipelineRunner([_MismatchedStage()])

    try:
        runner.run_until(context, PipelineStage.MEDIA_SCAN)
    except PipelineStageError as error:
        assert "returned a result" in str(error)
    else:
        raise AssertionError("pipeline accepted a result for the wrong stage")


def test_pipeline_runner_retains_completed_trace_when_later_stage_fails(
    tmp_path: Path,
) -> None:
    context = ProjectContext(
        input_path=tmp_path / "media",
        workspace=tmp_path / "workspace",
        settings=Settings(),
    )
    runner = PipelineRunner(
        [
            _StubStage(PipelineStage.MEDIA_SCAN, StageStatus.COMPLETED),
            _FailingStage(),
        ]
    )

    try:
        runner.run_until(context, PipelineStage.SCENE_DETECTION)
    except RuntimeError as error:
        assert "synthetic failure" in str(error)
    else:
        raise AssertionError("pipeline did not propagate the stage failure")

    assert [entry.stage for entry in runner.last_trace] == [PipelineStage.MEDIA_SCAN]


def test_pipeline_runner_maps_stage_progress_and_propagates_cancellation(
    tmp_path: Path,
) -> None:
    context = ProjectContext(
        input_path=tmp_path / "media",
        workspace=tmp_path / "workspace",
        settings=Settings(),
    )
    events: list[tuple[int, int, str]] = []
    runner = PipelineRunner(
        [
            _ProgressStage(PipelineStage.MEDIA_SCAN),
            _ProgressStage(PipelineStage.SCENE_DETECTION),
        ]
    )

    result = runner.run_until(
        context,
        PipelineStage.SCENE_DETECTION,
        progress=lambda current, total, message: events.append((current, total, message)),
    )

    assert result.status is StageStatus.COMPLETED
    assert (250, 2000, "work media_scan") in events
    assert (1250, 2000, "work scene_detection") in events
    assert events[-1] == (2000, 2000, "scene_detection complete")

    def cancel_on_work(current: int, total: int, message: str) -> None:
        del current, total
        if message.startswith("work"):
            raise RuntimeError("cancel requested")

    cancelled = PipelineRunner(
        [
            _ProgressStage(PipelineStage.MEDIA_SCAN),
            _ProgressStage(PipelineStage.SCENE_DETECTION),
        ]
    )
    try:
        cancelled.run_until(
            context,
            PipelineStage.SCENE_DETECTION,
            progress=cancel_on_work,
        )
    except RuntimeError as error:
        assert str(error) == "cancel requested"
    else:
        raise AssertionError("pipeline swallowed cancellation from stage progress")
    assert cancelled.last_trace == ()


def test_stage_result_keeps_legacy_skipped_compatibility() -> None:
    legacy = StageResult(stage=PipelineStage.MEDIA_SCAN, skipped=True)

    assert legacy.status is StageStatus.NO_INPUT
    assert legacy.skipped is True

    try:
        StageResult(
            stage=PipelineStage.MEDIA_SCAN,
            status=StageStatus.CACHED,
            skipped=False,
        )
    except ValueError as error:
        assert "skipped must match" in str(error)
    else:
        raise AssertionError("contradictory status and skipped values were accepted")


def test_semantic_create_uses_shared_movie_use_case(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "media"
    input_path.mkdir()
    output_path = tmp_path / "movie.mp4"
    workspace = tmp_path / "workspace"
    captured: dict[str, object] = {}
    service = TravelMovieService(Settings())

    def fake_run_until(target: PipelineStage, **kwargs: object) -> StageResult:
        captured["target"] = target
        captured.update(kwargs)
        return StageResult(stage=PipelineStage.RENDERING)

    def fail_quick_montage(**kwargs: object) -> QuickMontageResult:
        raise AssertionError("semantic create must use the canonical pipeline")

    monkeypatch.setattr(service, "run_until", fake_run_until)
    monkeypatch.setattr(service, "create_quick_montage", fail_quick_montage)

    result = service.create(
        input_path=input_path,
        output_path=output_path,
        workspace=workspace,
        style=StoryStyle.CINEMATIC,
        semantic=True,
    )

    assert result.stage is PipelineStage.RENDERING
    assert captured["target"] is PipelineStage.RENDERING
    assert captured["output_path"] == output_path
    settings = captured["montage_settings"]
    assert isinstance(settings, QuickMontageSettings)
    assert settings.semantic_analysis is True
    assert settings.story_style is StoryStyle.CINEMATIC


def test_chronological_create_keeps_quick_montage_use_case(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "media"
    input_path.mkdir()
    output_path = tmp_path / "movie.mp4"
    workspace = tmp_path / "workspace"
    captured: dict[str, object] = {}
    service = TravelMovieService(Settings())

    def fake_create_quick_montage(**kwargs: object) -> QuickMontageResult:
        captured.update(kwargs)
        return QuickMontageResult(
            output_path=output_path,
            timeline_path=workspace / "artifacts" / "quick_timeline.json",
            clip_count=3,
            duration_seconds=12,
        )

    monkeypatch.setattr(service, "create_quick_montage", fake_create_quick_montage)

    result = service.create(
        input_path=input_path,
        output_path=output_path,
        workspace=workspace,
        style=StoryStyle.CINEMATIC,
        semantic=False,
    )

    assert result.stage is PipelineStage.RENDERING
    assert captured["output_path"] == output_path
    settings = captured["settings"]
    assert isinstance(settings, QuickMontageSettings)
    assert settings.semantic_analysis is False


def test_report_creates_local_html_artifact(tmp_path: Path) -> None:
    input_path = tmp_path / "media"
    input_path.mkdir()

    result = TravelMovieService(Settings()).report(
        input_path=input_path,
        workspace=tmp_path / "workspace",
    )

    assert result.status is StageStatus.COMPLETED
    assert len(result.artifacts) == 1
    assert result.artifacts[0].name == "report.html"
    assert result.artifacts[0].is_file()


class _StubStage(Stage):
    def __init__(self, name: PipelineStage, status: StageStatus) -> None:
        self.name = name
        self._status = status

    def run(self, context: ProjectContext) -> StageResult:
        return StageResult(stage=self.name, status=self._status)


class _MismatchedStage(Stage):
    name = PipelineStage.MEDIA_SCAN

    def run(self, context: ProjectContext) -> StageResult:
        return StageResult(stage=PipelineStage.SCENE_DETECTION)


class _FailingStage(Stage):
    name = PipelineStage.SCENE_DETECTION

    def run(self, context: ProjectContext) -> StageResult:
        raise RuntimeError("synthetic failure")


class _ProgressStage(Stage):
    def __init__(self, name: PipelineStage) -> None:
        self.name = name

    def run(self, context: ProjectContext) -> StageResult:
        assert context.progress is not None
        context.progress(1, 4, f"work {self.name.value}")
        return StageResult(stage=self.name, message=f"{self.name.value} complete")
