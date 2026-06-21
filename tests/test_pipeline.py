from pathlib import Path

from travelmovieai.application import service as service_module
from travelmovieai.application.context import ProjectContext
from travelmovieai.application.service import TravelMovieService
from travelmovieai.core.config import Settings
from travelmovieai.domain.enums import PipelineStage, StoryStyle
from travelmovieai.domain.models import StageResult
from travelmovieai.pipeline.base import Stage
from travelmovieai.pipeline.registry import build_default_pipeline
from travelmovieai.pipeline.runner import PipelineRunner


def test_default_pipeline_matches_specification_order() -> None:
    assert [stage.name for stage in build_default_pipeline()] == list(PipelineStage)


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


def test_semantic_create_runs_canonical_pipeline(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "media"
    input_path.mkdir()
    output_path = tmp_path / "movie.mp4"
    workspace = tmp_path / "workspace"
    contexts: list[ProjectContext] = []

    class RecordingStage(Stage):
        def __init__(self, name: PipelineStage) -> None:
            self.name = name

        def run(self, context: ProjectContext) -> StageResult:
            contexts.append(context)
            return StageResult(
                stage=self.name,
                artifacts=[output_path] if self.name is PipelineStage.RENDERING else [],
                message=f"ran {self.name.value}",
            )

    monkeypatch.setattr(
        service_module,
        "build_default_pipeline",
        lambda: [
            RecordingStage(PipelineStage.MEDIA_SCAN),
            RecordingStage(PipelineStage.RENDERING),
        ],
    )

    result = TravelMovieService(Settings()).create(
        input_path=input_path,
        output_path=output_path,
        workspace=workspace,
        style=StoryStyle.CINEMATIC,
        semantic=True,
    )

    assert result.stage is PipelineStage.RENDERING
    assert [context.montage_settings is not None for context in contexts] == [True, True]
    assert contexts[-1].output_path == output_path.resolve()
    assert contexts[-1].montage_settings is not None
    assert contexts[-1].montage_settings.semantic_analysis is True
    assert contexts[-1].montage_settings.speech_analysis is False
