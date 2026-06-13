"""Use-case facade called by the CLI."""

from pathlib import Path

from travelmovieai.application.context import ProjectContext
from travelmovieai.core.config import Settings
from travelmovieai.domain.enums import PipelineStage, StoryStyle
from travelmovieai.domain.models import StageResult
from travelmovieai.pipeline.registry import build_default_pipeline
from travelmovieai.pipeline.runner import PipelineRunner


class TravelMovieService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def create(
        self,
        *,
        input_path: Path,
        output_path: Path,
        workspace: Path | None,
        style: StoryStyle,
        cloud: bool,
    ) -> StageResult:
        context = self._context(
            input_path=input_path,
            output_path=output_path,
            workspace=workspace,
            style=style,
            cloud=cloud,
        )
        return PipelineRunner(build_default_pipeline()).run_until(context, PipelineStage.RENDERING)

    def analyze(self, *, input_path: Path, workspace: Path | None) -> StageResult:
        return self.run_until(
            PipelineStage.MEDIA_SCAN,
            input_path=input_path,
            workspace=workspace,
        )

    def resolve_workspace(self, input_path: Path, workspace: Path | None) -> Path:
        return (workspace or self.settings.workspace / input_path.name).resolve()

    def run_until(
        self,
        target: PipelineStage,
        *,
        input_path: Path,
        workspace: Path | None,
        output_path: Path | None = None,
        style: StoryStyle = StoryStyle.CINEMATIC,
    ) -> StageResult:
        context = self._context(
            input_path=input_path,
            output_path=output_path,
            workspace=workspace,
            style=style,
        )
        return PipelineRunner(build_default_pipeline()).run_until(context, target)

    def report(self, *, input_path: Path, workspace: Path | None) -> StageResult:
        context = self._context(input_path=input_path, workspace=workspace)
        context.prepare()
        report_path = context.artifacts_dir / "report.html"
        return StageResult(
            stage=PipelineStage.EVENT_DETECTION,
            skipped=True,
            artifacts=[report_path],
            message=(
                "Project structure is ready. Report generation will be implemented "
                f"in a later milestone: {report_path}"
            ),
        )

    def _context(
        self,
        *,
        input_path: Path,
        workspace: Path | None,
        output_path: Path | None = None,
        style: StoryStyle = StoryStyle.CINEMATIC,
        cloud: bool = False,
    ) -> ProjectContext:
        project_workspace = self.resolve_workspace(input_path, workspace)
        return ProjectContext(
            input_path=input_path,
            workspace=project_workspace,
            output_path=output_path,
            settings=self.settings,
            style=style,
            cloud=cloud or self.settings.cloud_enabled,
        )
