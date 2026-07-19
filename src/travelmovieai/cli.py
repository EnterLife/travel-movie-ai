"""Command-line interface for TravelMovieAI."""

from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from travelmovieai.application.service import TravelMovieService
from travelmovieai.core.config import load_settings
from travelmovieai.core.exceptions import TravelMovieError
from travelmovieai.domain.enums import PipelineStage, StoryStyle
from travelmovieai.domain.models import QuickMontageSettings, StageResult

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
    return TravelMovieService(load_settings())


def _run(operation: Callable[[], StageResult]) -> None:
    try:
        result = operation()
    except (TravelMovieError, ValidationError) as error:
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
    style: Annotated[
        StoryStyle,
        typer.Option(
            case_sensitive=False,
            help="Story style used by semantic scene analysis and music selection.",
        ),
    ] = StoryStyle.CINEMATIC,
    semantic: Annotated[
        bool,
        typer.Option(
            "--semantic/--quick",
            help="Use local Vision AI analysis to select scenes.",
        ),
    ] = False,
    variant_name: Annotated[
        str,
        typer.Option("--variant", help="Safe name for this movie variant."),
    ] = "Default",
    target_duration: Annotated[
        float,
        typer.Option("--duration", min=5, max=3600, help="Target movie duration in seconds."),
    ] = 90,
    max_clip_duration: Annotated[
        float,
        typer.Option("--max-clip", min=1, max=60, help="Maximum video clip duration."),
    ] = 6,
    speech: Annotated[
        bool,
        typer.Option("--speech/--no-speech", help="Run local Faster Whisper analysis."),
    ] = False,
    narration: Annotated[
        bool,
        typer.Option("--narration/--no-narration", help="Synthesize local Piper narration."),
    ] = False,
    framing: Annotated[
        str,
        typer.Option("--framing", help="fit, fill, or AI-guided smart framing."),
    ] = "fit",
    vertical_layout: Annotated[
        str,
        typer.Option("--vertical-layout", help="fit, blur, or crop vertical footage."),
    ] = "fit",
    photo_motion: Annotated[
        str,
        typer.Option("--photo-motion", help="none or ken_burns."),
    ] = "none",
    color_normalization: Annotated[
        bool,
        typer.Option("--color-normalization/--no-color-normalization"),
    ] = False,
    hdr_to_sdr: Annotated[
        bool,
        typer.Option("--hdr-to-sdr/--keep-hdr"),
    ] = False,
    event_titles: Annotated[
        bool,
        typer.Option("--event-titles/--no-event-titles"),
    ] = False,
    subtitles: Annotated[
        bool,
        typer.Option("--subtitles/--no-subtitles"),
    ] = False,
    credits: Annotated[
        str | None,
        typer.Option("--credits", help="Optional final credit text."),
    ] = None,
    music_mode: Annotated[
        str,
        typer.Option("--music-mode", help="auto, generated, library, manual, or none."),
    ] = "auto",
    music_path: Annotated[
        Path | None,
        typer.Option("--music-path", exists=True, dir_okay=False, resolve_path=True),
    ] = None,
    bpm_analysis: Annotated[
        bool,
        typer.Option("--bpm-analysis/--no-bpm-analysis"),
    ] = False,
    music_envelope: Annotated[
        bool,
        typer.Option("--music-envelope/--no-music-envelope"),
    ] = False,
) -> None:
    """Create a quick or locally AI-directed montage."""

    def operation() -> StageResult:
        settings = QuickMontageSettings.model_validate(
            {
                "target_duration_seconds": target_duration,
                "max_video_clip_seconds": max_clip_duration,
                "semantic_analysis": semantic,
                "speech_analysis": speech,
                "narration_enabled": narration,
                "framing_mode": framing,
                "vertical_video_layout": vertical_layout,
                "photo_motion": photo_motion,
                "color_normalization": color_normalization,
                "hdr_to_sdr": hdr_to_sdr,
                "event_titles_enabled": event_titles,
                "scene_subtitles_enabled": subtitles,
                "credits_text": credits,
                "music_enabled": music_mode != "none",
                "music_mode": music_mode,
                "music_path": music_path,
                "music_bpm_analysis": bpm_analysis,
                "music_volume_envelope": music_envelope,
                "story_style": style,
            }
        )
        return _service().create(
            input_path=input_path,
            output_path=output,
            workspace=workspace,
            style=style,
            semantic=semantic,
            montage_settings=settings,
            variant_name=variant_name,
        )

    _run(operation)


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
def estimate(
    input_path: InputOption,
    workspace: WorkspaceOption = None,
    semantic: Annotated[
        bool,
        typer.Option("--semantic/--quick", help="Estimate the full local Vision AI workflow."),
    ] = False,
    speech: Annotated[
        bool,
        typer.Option("--speech/--no-speech", help="Include local speech recognition."),
    ] = False,
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Print the typed estimate as JSON."),
    ] = False,
) -> None:
    """Estimate runtime and peak workspace storage from probed project metadata."""
    try:
        report = _service().estimate(
            input_path=input_path,
            workspace=workspace,
            montage_settings=QuickMontageSettings(
                semantic_analysis=semantic,
                speech_analysis=speech,
            ),
        )
    except TravelMovieError as error:
        typer.secho(str(error), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error
    if as_json:
        typer.echo(report.model_dump_json(indent=2))
        return
    workload = report.workload
    typer.echo(
        f"{workload.asset_count} assets, {workload.estimated_scene_count} estimated scenes; "
        f"workspace {_format_bytes(report.estimated_peak_workspace_bytes)}; "
        f"runtime {report.runtime.lower_seconds / 60:.1f}-"
        f"{report.runtime.upper_seconds / 60:.1f} min "
        f"(likely {report.runtime.likely_seconds / 60:.1f} min)."
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


@app.command()
def search(
    input_path: InputOption,
    query: Annotated[
        str,
        typer.Argument(help="Natural-language description of scenes to find."),
    ],
    workspace: WorkspaceOption = None,
    limit: Annotated[
        int,
        typer.Option("--limit", "-n", min=1, max=100),
    ] = 10,
) -> None:
    """Search the local FAISS scene archive without exposing media externally."""
    try:
        result = _service().search(
            input_path=input_path,
            workspace=workspace,
            query=query,
            limit=limit,
        )
    except TravelMovieError as error:
        typer.secho(str(error), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error
    if not result.hits:
        typer.echo("No matching scenes found.")
        return
    for hit in result.hits:
        typer.echo(f"{hit.rank}. {hit.scene_id} score={hit.score:.4f}")


@app.command("export")
def export_project(
    input_path: InputOption,
    output: Annotated[
        Path,
        typer.Option("--output", "-o", dir_okay=False, resolve_path=True),
    ],
    workspace: WorkspaceOption = None,
    include_rendered_media: Annotated[
        bool,
        typer.Option(
            "--include-rendered-media/--metadata-only",
            help="Include generated audio and movies in the local backup.",
        ),
    ] = False,
    overwrite: Annotated[
        bool,
        typer.Option(help="Explicitly replace an existing archive."),
    ] = False,
) -> None:
    """Export a checksummed local project backup without source media."""
    _run(
        lambda: _service().export_project(
            input_path=input_path,
            workspace=workspace,
            output_path=output,
            include_rendered_media=include_rendered_media,
            overwrite=overwrite,
        )
    )


@app.command("restore")
def restore_project(
    archive: Annotated[
        Path,
        typer.Option(
            "--archive",
            "-a",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    workspace: Annotated[
        Path,
        typer.Option(
            "--workspace",
            "-w",
            file_okay=False,
            resolve_path=True,
            help="New or empty workspace folder for the restored project.",
        ),
    ],
) -> None:
    """Restore a validated project archive into a new or empty workspace."""
    _run(lambda: _service().restore_project(archive_path=archive, workspace=workspace))


@app.command()
def doctor() -> None:
    """Check FFmpeg, local AI packages, model cache, CUDA, and Piper readiness."""
    report = _service().diagnostics()
    labels = {"ok": "OK", "warning": "WARN", "error": "FAIL"}
    colors = {
        "ok": typer.colors.GREEN,
        "warning": typer.colors.YELLOW,
        "error": typer.colors.RED,
    }
    for check in report.checks:
        typer.secho(
            f"[{labels[check.level]}] {check.name}: {check.message}",
            fg=colors[check.level],
        )
    if not report.ready:
        raise typer.Exit(code=1)


def _format_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TiB"
