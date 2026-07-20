from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from travelmovieai.application.context import ProjectContext
from travelmovieai.application.service import TravelMovieService
from travelmovieai.application.workspace_identity import ensure_workspace_identity
from travelmovieai.application.workspace_lease import WorkspaceLease
from travelmovieai.core.config import Settings
from travelmovieai.core.exceptions import PipelineStageError, WorkspaceBusyError
from travelmovieai.domain.enums import PipelineStage, StageStatus, StoryStyle
from travelmovieai.domain.models import (
    QuickMontageResult,
    QuickMontageSettings,
    StageExecutionMetadata,
    StageResult,
)
from travelmovieai.pipeline.base import Stage
from travelmovieai.pipeline.progress import PipelineRunManifest, ProgressEvent
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


@pytest.mark.parametrize(("pool_size", "expected_clear_count"), [(0, 1), (1, 0), (4, 0)])
def test_pipeline_runner_preserves_configured_idle_vision_pool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pool_size: int,
    expected_clear_count: int,
) -> None:
    context = ProjectContext(
        input_path=tmp_path / "media",
        workspace=tmp_path / f"workspace-{pool_size}",
        settings=Settings(vision_model_pool_size=pool_size),
    )
    clear_calls = 0

    def record_clear() -> None:
        nonlocal clear_calls
        clear_calls += 1

    monkeypatch.setattr(
        "travelmovieai.pipeline.runner._clear_idle_vision_models",
        record_clear,
    )

    PipelineRunner([_StubStage(PipelineStage.VISION_ANALYSIS, StageStatus.COMPLETED)]).run_until(
        context, PipelineStage.VISION_ANALYSIS
    )

    assert clear_calls == expected_clear_count


def test_pipeline_runner_maps_stage_progress_and_propagates_cancellation(
    tmp_path: Path,
) -> None:
    context = ProjectContext(
        input_path=tmp_path / "media",
        workspace=tmp_path / "workspace",
        settings=Settings(),
    )
    events: list[tuple[int, int, str]] = []
    typed_events: list[ProgressEvent] = []
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
        progress_events=typed_events.append,
    )

    assert result.status is StageStatus.COMPLETED
    assert (42, 1000, "work media_scan") in events
    assert (375, 1000, "work scene_detection") in events
    assert events[-1] == (1000, 1000, "scene_detection complete")
    assert [event.overall_current for event in typed_events] == sorted(
        event.overall_current for event in typed_events
    )
    work_event = next(event for event in typed_events if event.message == "work scene_detection")
    assert work_event.stage is PipelineStage.SCENE_DETECTION
    assert work_event.current == 1
    assert work_event.total == 4
    assert work_event.unit == "assets"
    manifest = PipelineRunManifest.model_validate_json(
        (context.artifacts_dir / "pipeline_run.json").read_text(encoding="utf-8")
    )
    assert manifest.status == "completed"
    assert manifest.completed_stage_count == 2
    assert manifest.stages[0].weight < manifest.stages[1].weight
    assert all(stage.duration_seconds >= 0 for stage in manifest.stages)
    assert manifest.stages[1].execution.provider == "fake-provider"
    assert manifest.stages[1].execution.fallback_provider == "fake-fallback"
    assert manifest.stages[1].execution.fallback_count == 2

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


def test_pipeline_run_manifest_records_redacted_failure(tmp_path: Path) -> None:
    input_path = tmp_path / "private media"
    private_music = tmp_path / "private soundtrack" / "family-song.wav"
    context = ProjectContext(
        input_path=input_path,
        workspace=tmp_path / "workspace",
        settings=Settings(),
        montage_settings=QuickMontageSettings(music_mode="manual", music_path=private_music),
    )

    class PrivateFailingStage(Stage):
        name = PipelineStage.MEDIA_SCAN

        def run(self, context: ProjectContext) -> StageResult:
            raise RuntimeError(f"{context.input_path} {private_music} token=private-value")

    with pytest.raises(RuntimeError, match="private-value"):
        PipelineRunner([PrivateFailingStage()]).run_until(
            context,
            PipelineStage.MEDIA_SCAN,
        )

    manifest_text = (context.artifacts_dir / "pipeline_run.json").read_text(encoding="utf-8")
    manifest = PipelineRunManifest.model_validate_json(manifest_text)
    assert manifest.status == "failed"
    assert manifest.failure is not None
    assert manifest.failure.error_type == "RuntimeError"
    assert str(input_path) not in manifest_text
    assert str(private_music) not in manifest_text
    assert "private-value" not in manifest_text
    assert manifest.stages[-1].status == "failed"


def test_pipeline_run_manifest_redacts_resolved_default_model_cache_path(
    tmp_path: Path,
) -> None:
    context = ProjectContext(
        input_path=tmp_path / "media",
        workspace=tmp_path / "workspace",
        settings=Settings(),
    )
    resolved_cache = context.settings.model_cache.expanduser().resolve()
    private_model = resolved_cache / "voices" / "private-voice.onnx"

    class ModelPathFailingStage(Stage):
        name = PipelineStage.MEDIA_SCAN

        def run(self, context: ProjectContext) -> StageResult:
            del context
            raise RuntimeError(f"Local model could not be loaded from {private_model}")

    with pytest.raises(RuntimeError, match="Local model could not be loaded"):
        PipelineRunner([ModelPathFailingStage()]).run_until(
            context,
            PipelineStage.MEDIA_SCAN,
        )

    manifest_text = (context.artifacts_dir / "pipeline_run.json").read_text(encoding="utf-8")
    manifest = PipelineRunManifest.model_validate_json(manifest_text)
    assert manifest.failure is not None
    assert str(private_model) not in manifest_text
    assert str(resolved_cache) not in manifest_text
    assert str(Path.cwd()) not in manifest_text
    assert "<local-path>" in manifest.failure.message


def test_pipeline_run_manifest_closes_cleanly_on_keyboard_interrupt(tmp_path: Path) -> None:
    context = ProjectContext(
        input_path=tmp_path / "media",
        workspace=tmp_path / "workspace",
        settings=Settings(),
    )

    class InterruptedStage(Stage):
        name = PipelineStage.MEDIA_SCAN

        def run(self, context: ProjectContext) -> StageResult:
            del context
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        PipelineRunner([InterruptedStage()]).run_until(context, PipelineStage.MEDIA_SCAN)

    manifest = PipelineRunManifest.model_validate_json(
        (context.artifacts_dir / "pipeline_run.json").read_text(encoding="utf-8")
    )
    assert manifest.status == "failed"
    assert manifest.finished_at is not None
    assert manifest.failure is not None
    assert manifest.failure.error_type == "KeyboardInterrupt"


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

    degraded = StageResult(
        stage=PipelineStage.RENDERING,
        status=StageStatus.DEGRADED,
    )
    assert degraded.skipped is False


def test_degraded_stage_can_report_cached_execution() -> None:
    result = StageResult(
        stage=PipelineStage.RENDERING,
        status=StageStatus.DEGRADED,
        cache_hit=True,
    )

    assert result.status is StageStatus.DEGRADED
    assert result.cache_hit is True
    assert result.skipped is True


def test_pipeline_manifest_records_cached_degraded_execution(tmp_path: Path) -> None:
    context = ProjectContext(
        input_path=tmp_path / "media",
        workspace=tmp_path / "workspace",
        settings=Settings(),
    )

    class CachedDegradedStage(Stage):
        name = PipelineStage.RENDERING

        def run(self, context: ProjectContext) -> StageResult:
            del context
            return StageResult(
                stage=self.name,
                status=StageStatus.DEGRADED,
                cache_hit=True,
            )

    PipelineRunner([CachedDegradedStage()]).run_until(context, PipelineStage.RENDERING)

    manifest = PipelineRunManifest.model_validate_json(
        (context.artifacts_dir / "pipeline_run.json").read_text(encoding="utf-8")
    )
    assert manifest.stages[0].status is StageStatus.DEGRADED
    assert manifest.stages[0].cache_hit is True


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


def test_report_and_export_respect_workspace_lease(tmp_path: Path) -> None:
    input_path = tmp_path / "media"
    input_path.mkdir()
    workspace = tmp_path / "workspace"
    ensure_workspace_identity(input_path, workspace)
    service = TravelMovieService(Settings())

    def run_report() -> StageResult:
        return service.report(input_path=input_path, workspace=workspace)

    def run_export() -> StageResult:
        return service.export_project(
            input_path=input_path,
            workspace=workspace,
            output_path=tmp_path / "backup.zip",
        )

    with (
        WorkspaceLease(workspace, operation="pipeline"),
        ThreadPoolExecutor(max_workers=1) as executor,
    ):
        with pytest.raises(WorkspaceBusyError):
            executor.submit(run_report).result(timeout=5)
        with pytest.raises(WorkspaceBusyError):
            executor.submit(run_export).result(timeout=5)


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
        return StageResult(
            stage=self.name,
            message=f"{self.name.value} complete",
            execution=StageExecutionMetadata(
                provider="fake-provider" if self.name is PipelineStage.SCENE_DETECTION else None,
                fallback_provider=(
                    "fake-fallback" if self.name is PipelineStage.SCENE_DETECTION else None
                ),
                fallback_count=2 if self.name is PipelineStage.SCENE_DETECTION else 0,
            ),
        )
