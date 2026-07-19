"""Offline HTML project report generation."""

import html
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

from pydantic import BaseModel, ValidationError

from travelmovieai.application.context import ProjectContext
from travelmovieai.core.exceptions import PipelineStageError
from travelmovieai.domain.models import (
    Event,
    MontageQualityReport,
    QuickMontagePlan,
    Scene,
    SceneSelectionDecision,
    SceneSelectionReport,
    Storyboard,
)
from travelmovieai.infrastructure.database import MediaAssetRepository


@dataclass(frozen=True, slots=True)
class ProjectReportResult:
    path: Path
    asset_count: int
    scene_count: int
    event_count: int
    selected_clip_count: int


def generate_project_report(context: ProjectContext) -> ProjectReportResult:
    """Build a self-contained report without external scripts, fonts, or telemetry."""

    context.prepare()
    repository = MediaAssetRepository(context.database_path)
    repository.initialize()
    assets = repository.list_assets()
    scenes = repository.list_scenes()
    events = repository.list_events()
    storyboard = _read_optional_model(
        context.artifacts_dir / "storyboard.json",
        Storyboard,
    )
    timeline = _read_optional_model(
        context.artifacts_dir / "quick_timeline.json",
        QuickMontagePlan,
    )
    quality = _read_optional_model(
        context.artifacts_dir / "montage_quality_report.json",
        MontageQualityReport,
    )
    selection_report = _read_optional_model(
        context.artifacts_dir / "selection_decisions.json",
        SceneSelectionReport,
    )
    decisions = (
        {decision.scene_id: decision for decision in selection_report.decisions}
        if selection_report is not None
        else {}
    )
    asset_names = {asset.id: asset.relative_path.as_posix() for asset in assets}
    selected_ids = (
        {clip.scene_id for clip in timeline.clips if clip.scene_id is not None}
        if timeline is not None
        else set()
    )
    report_path = context.artifacts_dir / "report.html"
    content = _build_html(
        asset_count=len(assets),
        scenes=scenes,
        events=events,
        storyboard=storyboard,
        timeline=timeline,
        quality=quality,
        asset_names=asset_names,
        selected_ids=selected_ids,
        decisions=decisions,
    )
    _write_text_atomic(report_path, content)
    return ProjectReportResult(
        path=report_path,
        asset_count=len(assets),
        scene_count=len(scenes),
        event_count=len(events),
        selected_clip_count=len(selected_ids),
    )


def _build_html(
    *,
    asset_count: int,
    scenes: list[Scene],
    events: list[Event],
    storyboard: Storyboard | None,
    timeline: QuickMontagePlan | None,
    quality: MontageQualityReport | None,
    asset_names: dict[UUID, str],
    selected_ids: set[UUID],
    decisions: dict[UUID, SceneSelectionDecision],
) -> str:
    title = storyboard.title if storyboard is not None else "TravelMovieAI project"
    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    duration = timeline.total_duration_seconds if timeline is not None else 0
    quality_score = f"{quality.score:.1f}/100" if quality is not None else "Not rendered"
    issue_count = len(quality.issues) if quality is not None else 0
    event_cards = "".join(_event_html(event) for event in events) or (
        '<p class="empty">No events have been detected yet.</p>'
    )
    scene_rows = (
        "".join(
            _scene_html(
                scene,
                asset_names.get(scene.asset_id, "unknown"),
                scene.id in selected_ids,
                decisions.get(scene.id),
            )
            for scene in scenes
        )
        or '<tr><td colspan="8" class="empty">No scenes have been analyzed yet.</td></tr>'
    )
    issues = (
        "".join(f"<li>{html.escape(issue.message)}</li>" for issue in quality.issues)
        if quality is not None and quality.issues
        else "<li>No montage diagnostics.</li>"
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="Content-Security-Policy"
        content="default-src 'none'; style-src 'unsafe-inline'">
  <title>{html.escape(title)}</title>
  <style>
    :root {{ color-scheme: dark; font-family: Inter, Segoe UI, sans-serif; }}
    body {{ margin: 0; background: #0d1117; color: #e6edf3; }}
    main {{ max-width: 1180px; margin: auto; padding: 32px; }}
    h1 {{ margin-bottom: 4px; }} .muted, .empty {{ color: #8b949e; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(150px,1fr));
             gap: 12px; margin: 24px 0; }}
    .metric, .event {{ background: #161b22; border: 1px solid #30363d;
                      border-radius: 10px; padding: 16px; }}
    .metric strong {{ display: block; font-size: 1.55rem; margin-top: 6px; }}
    .events {{ display: grid; gap: 10px;
               grid-template-columns: repeat(auto-fit,minmax(260px,1fr)); }}
    table {{ width: 100%; border-collapse: collapse; font-size: .9rem; }}
    th, td {{ padding: 9px; border-bottom: 1px solid #30363d;
              text-align: left; vertical-align: top; }}
    th {{ position: sticky; top: 0; background: #161b22; }}
    .selected {{ color: #3fb950; font-weight: 600; }}
    .table-wrap {{ overflow: auto; max-height: 720px; }}
    section {{ margin-top: 30px; }} code {{ color: #79c0ff; }}
  </style>
</head>
<body><main>
  <h1>{html.escape(title)}</h1><p class="muted">Generated locally at {generated_at}</p>
  <div class="grid">
    {_metric("Assets", str(asset_count))}
    {_metric("Scenes", str(len(scenes)))}
    {_metric("Events", str(len(events)))}
    {_metric("Selected clips", str(len(selected_ids)))}
    {_metric("Planned duration", f"{duration:.1f} s")}
    {_metric("Montage quality", quality_score)}
  </div>
  <section><h2>Story events</h2><div class="events">{event_cards}</div></section>
  <section><h2>Scene decisions</h2><div class="table-wrap"><table>
    <thead><tr><th>Selected</th><th>Reason</th><th>Source</th><th>Start</th><th>End</th><th>Quality</th><th>Importance</th><th>Caption</th></tr></thead>
    <tbody>{scene_rows}</tbody>
  </table></div></section>
  <section><h2>Diagnostics</h2><p>Issue count: {issue_count}</p><ul>{issues}</ul></section>
</main></body></html>
"""


def _metric(label: str, value: str) -> str:
    return (
        '<div class="metric"><span class="muted">'
        f"{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>"
    )


def _event_html(event: Event) -> str:
    summary = event.summary or "No summary"
    return (
        '<article class="event">'
        f"<h3>{html.escape(event.title)}</h3>"
        f"<p>{html.escape(summary)}</p>"
        f'<span class="muted">{len(event.scene_ids)} scene(s), '
        f"importance {event.importance_score:.1f}</span>"
        "</article>"
    )


def _scene_html(
    scene: Scene,
    asset_name: str,
    selected: bool,
    decision: SceneSelectionDecision | None,
) -> str:
    caption = scene.caption or ""
    quality = "—" if scene.quality_score is None else f"{scene.quality_score:.1f}"
    importance = "—" if scene.importance_score is None else f"{scene.importance_score:.1f}"
    reason = decision.reason if decision is not None else "No recorded selection explanation"
    return (
        "<tr>"
        f'<td class="{"selected" if selected else ""}">{"yes" if selected else "no"}</td>'
        f"<td>{html.escape(reason)}</td>"
        f"<td><code>{html.escape(asset_name)}</code></td>"
        f"<td>{scene.start_seconds:.2f}</td><td>{scene.end_seconds:.2f}</td>"
        f"<td>{quality}</td><td>{importance}</td><td>{html.escape(caption)}</td>"
        "</tr>"
    )


def _read_optional_model[ModelT: BaseModel](path: Path, model: type[ModelT]) -> ModelT | None:
    if not path.is_file():
        return None
    try:
        return model.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as error:
        raise PipelineStageError(f"Could not read {path.name} for the HTML report.") from error


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
