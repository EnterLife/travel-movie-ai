"""Command-line interface for TravelMovieAI."""

from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import typer

from travelmovieai.application.service import TravelMovieService
from travelmovieai.core.config import Settings
from travelmovieai.core.exceptions import TravelMovieError
from travelmovieai.domain.enums import PipelineStage, StoryStyle
from travelmovieai.domain.models import StageResult

app = typer.Typer(
    name="travelmovieai",
    help="Turn raw travel media into a story-driven movie.",
    no_args_is_help=True,
)

InputOption = Annotated[
    Path,
    typer.Option(
        "--input",
        "-i",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Folder containing source media.",
    ),
]
WorkspaceOption = Annotated[
    Path | None,
    typer.Option(
        "--workspace",
        "-w",
        file_okay=False,
        resolve_path=True,
        help="Folder for project metadata and intermediate files.",
    ),
]


def _service() -> TravelMovieService:
    return TravelMovieService(Settings())


def _run(operation: Callable[[], StageResult]) -> None:
    try:
        result = operation()
    except TravelMovieError as error:
        typer.secho(str(error), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error
    typer.echo(result.message)


@app.command()
def create(
    input_path: InputOption,
    output: Annotated[
        Path,
        typer.Option("--output", "-o", dir_okay=False, resolve_path=True),
    ],
    workspace: WorkspaceOption = None,
    style: Annotated[StoryStyle, typer.Option(case_sensitive=False)] = StoryStyle.CINEMATIC,
    cloud: Annotated[bool, typer.Option("--cloud", help="Allow optional cloud providers.")] = False,
) -> None:
    """Run the complete analysis, story, timeline, and rendering pipeline."""
    _run(
        lambda: _service().create(
            input_path=input_path,
            output_path=output,
            workspace=workspace,
            style=style,
            cloud=cloud,
        )
    )


@app.command()
def analyze(
    input_path: InputOption,
    workspace: WorkspaceOption = None,
) -> None:
    """Scan media and persist the currently implemented analysis metadata."""
    _run(
        lambda: _service().analyze(
            input_path=input_path,
            workspace=workspace,
        )
    )


@app.command()
def storyboard(
    input_path: InputOption,
    workspace: WorkspaceOption = None,
    style: Annotated[StoryStyle, typer.Option(case_sensitive=False)] = StoryStyle.CINEMATIC,
) -> None:
    """Build a storyboard from previously analyzed media."""
    _run(
        lambda: _service().run_until(
            PipelineStage.STORY_BUILDER,
            input_path=input_path,
            workspace=workspace,
            style=style,
        )
    )


@app.command()
def render(
    input_path: InputOption,
    output: Annotated[
        Path,
        typer.Option("--output", "-o", dir_okay=False, resolve_path=True),
    ],
    workspace: WorkspaceOption = None,
) -> None:
    """Render a movie from the generated timeline."""
    _run(
        lambda: _service().run_until(
            PipelineStage.RENDERING,
            input_path=input_path,
            output_path=output,
            workspace=workspace,
        )
    )


@app.command()
def report(
    input_path: InputOption,
    workspace: WorkspaceOption = None,
) -> None:
    """Generate an HTML project report."""
    _run(lambda: _service().report(input_path=input_path, workspace=workspace))
