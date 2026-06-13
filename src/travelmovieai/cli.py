"""Command-line interface for TravelMovieAI."""

from pathlib import Path
from typing import Annotated

import typer

from travelmovieai.application.service import TravelMovieService
from travelmovieai.core.config import Settings
from travelmovieai.domain.enums import PipelineStage, StoryStyle

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
    result = _service().create(
        input_path=input_path,
        output_path=output,
        workspace=workspace,
        style=style,
        cloud=cloud,
    )
    typer.echo(result.message)


@app.command()
def analyze(
    input_path: InputOption,
    workspace: WorkspaceOption = None,
) -> None:
    """Scan media and persist the currently implemented analysis metadata."""
    result = _service().run_until(
        PipelineStage.MEDIA_SCAN,
        input_path=input_path,
        workspace=workspace,
    )
    typer.echo(result.message)


@app.command()
def storyboard(
    input_path: InputOption,
    workspace: WorkspaceOption = None,
    style: Annotated[StoryStyle, typer.Option(case_sensitive=False)] = StoryStyle.CINEMATIC,
) -> None:
    """Build a storyboard from previously analyzed media."""
    result = _service().run_until(
        PipelineStage.STORY_BUILDER,
        input_path=input_path,
        workspace=workspace,
        style=style,
    )
    typer.echo(result.message)


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
    result = _service().run_until(
        PipelineStage.RENDERING,
        input_path=input_path,
        output_path=output,
        workspace=workspace,
    )
    typer.echo(result.message)


@app.command()
def report(
    input_path: InputOption,
    workspace: WorkspaceOption = None,
) -> None:
    """Generate an HTML project report."""
    result = _service().report(input_path=input_path, workspace=workspace)
    typer.echo(result.message)
