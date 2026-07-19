from pathlib import Path

from travelmovieai.application.context import ProjectContext
from travelmovieai.application.service import TravelMovieService
from travelmovieai.core.config import Settings
from travelmovieai.domain.enums import PipelineStage, StoryStyle
from travelmovieai.domain.models import QuickMontageResult, QuickMontageSettings
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

    def fake_create_quick_montage(**kwargs: object) -> QuickMontageResult:
        captured.update(kwargs)
        return QuickMontageResult(
            output_path=output_path,
            timeline_path=workspace / "artifacts" / "quick_timeline.json",
            clip_count=3,
            duration_seconds=12,
            selection_mode="semantic",
        )

    monkeypatch.setattr(service, "create_quick_montage", fake_create_quick_montage)

    result = service.create(
        input_path=input_path,
        output_path=output_path,
        workspace=workspace,
        style=StoryStyle.CINEMATIC,
        semantic=True,
    )

    assert result.stage is PipelineStage.RENDERING
    assert captured["output_path"] == output_path
    settings = captured["settings"]
    assert isinstance(settings, QuickMontageSettings)
    assert settings.semantic_analysis is True
    assert settings.story_style is StoryStyle.CINEMATIC


def test_report_does_not_claim_uncreated_artifact(tmp_path: Path) -> None:
    input_path = tmp_path / "media"
    input_path.mkdir()

    result = TravelMovieService(Settings()).report(
        input_path=input_path,
        workspace=tmp_path / "workspace",
    )

    assert result.skipped is True
    assert result.artifacts == []
