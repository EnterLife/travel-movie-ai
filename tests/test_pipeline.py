from pathlib import Path

from travelmovieai.application.context import ProjectContext
from travelmovieai.core.config import Settings
from travelmovieai.domain.enums import PipelineStage
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
