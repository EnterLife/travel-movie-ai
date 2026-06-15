"""Quality gates for planned quick montage timelines."""

from collections.abc import Iterable
from datetime import UTC, datetime
from uuid import UUID

from travelmovieai.domain.models import (
    MontageClip,
    MontageQualityIssue,
    MontageQualityReport,
    QuickMontagePlan,
    Scene,
)


def build_montage_quality_report(
    plan: QuickMontagePlan,
    scenes: list[Scene],
) -> MontageQualityReport:
    scenes_by_id = {scene.id: scene for scene in scenes}
    selected_scenes = [
        scenes_by_id[clip.scene_id]
        for clip in plan.clips
        if clip.scene_id is not None and clip.scene_id in scenes_by_id
    ]
    selected_events = {
        clip.event_id
        for clip in plan.clips
        if clip.event_id is not None
    }
    total_events = {
        _event_id(scene)
        for scene in scenes
        if _event_id(scene) is not None
    }
    source_counts = _source_counts(plan.clips)
    source_count = len(source_counts)
    dominant_source_ratio = (
        max(source_counts.values()) / len(plan.clips) if plan.clips and source_counts else 0.0
    )
    average_semantic_score = _average(
        clip.semantic_score for clip in plan.clips if clip.semantic_score is not None
    )
    average_quality_score = _average(
        scene.quality_score for scene in selected_scenes if scene.quality_score is not None
    )
    issues = _quality_issues(
        plan,
        selected_scenes,
        source_count=source_count,
        dominant_source_ratio=dominant_source_ratio,
        average_semantic_score=average_semantic_score,
        average_quality_score=average_quality_score,
    )
    event_coverage = (
        len(selected_events) / len(total_events)
        if total_events
        else 1.0 if selected_events or not scenes else 0.0
    )
    return MontageQualityReport(
        created_at=datetime.now(UTC),
        score=_report_score(issues),
        target_duration_seconds=plan.settings.target_duration_seconds,
        planned_duration_seconds=plan.total_duration_seconds,
        duration_ratio=_duration_ratio(plan),
        clip_count=len(plan.clips),
        selected_scene_count=len(selected_scenes),
        selected_event_count=len(selected_events),
        total_event_count=len(total_events),
        event_coverage_ratio=event_coverage,
        source_count=source_count,
        dominant_source_ratio=dominant_source_ratio,
        average_semantic_score=average_semantic_score,
        average_quality_score=average_quality_score,
        window_selection=_window_selection(plan.clips),
        music_mode=plan.music_plan.mode if plan.music_plan else None,
        music_duration_seconds=(
            plan.music_plan.duration_seconds if plan.music_plan is not None else None
        ),
        music_accent_count=len(plan.music_plan.accents) if plan.music_plan else 0,
        issues=issues,
    )


def _quality_issues(
    plan: QuickMontagePlan,
    selected_scenes: list[Scene],
    *,
    source_count: int,
    dominant_source_ratio: float,
    average_semantic_score: float | None,
    average_quality_score: float | None,
) -> list[MontageQualityIssue]:
    issues: list[MontageQualityIssue] = []
    duration_ratio = _duration_ratio(plan)
    if duration_ratio < 0.82:
        issues.append(
            MontageQualityIssue(
                severity="warning",
                code="short_timeline",
                message=(
                    "Planned movie is much shorter than the target duration; "
                    "the archive may not contain enough strong scenes."
                ),
            )
        )
    if not plan.clips:
        issues.append(
            MontageQualityIssue(
                severity="critical",
                code="empty_timeline",
                message="Timeline has no clips.",
            )
        )
    if source_count > 1 and dominant_source_ratio > 0.55:
        issues.append(
            MontageQualityIssue(
                severity="warning",
                code="source_dominance",
                message=(
                    "One source file dominates the montage; consider increasing "
                    "event diversity or reviewing scene selection."
                ),
            )
        )
    if average_semantic_score is not None and average_semantic_score < 55:
        issues.append(
            MontageQualityIssue(
                severity="warning",
                code="low_semantic_score",
                message="Selected scenes have a low average semantic score.",
            )
        )
    if average_quality_score is not None and average_quality_score < 42:
        issues.append(
            MontageQualityIssue(
                severity="warning",
                code="low_visual_quality",
                message="Selected scenes have a low average visual quality score.",
            )
        )
    if plan.music_plan is None or plan.music_plan.mode == "none":
        issues.append(
            MontageQualityIssue(
                severity="info",
                code="music_disabled",
                message="Music is disabled; the final movie will rely on source audio only.",
            )
        )
    elif (
        plan.music_plan.duration_seconds is not None
        and plan.music_plan.duration_seconds < plan.total_duration_seconds * 0.92
    ):
        issues.append(
            MontageQualityIssue(
                severity="warning",
                code="short_music_plan",
                message="Music plan is shorter than the montage timeline.",
            )
        )
    if (
        plan.music_plan is not None
        and plan.settings.music_sync
        and len(plan.music_plan.accents) < 2
    ):
        issues.append(
            MontageQualityIssue(
                severity="info",
                code="few_music_accents",
                message="Music sync has very few accents for this timeline.",
            )
        )

    for index, scene in enumerate(selected_scenes):
        metrics = scene.metadata.get("quality_metrics", {})
        if not isinstance(metrics, dict):
            continue
        reasons = [str(reason) for reason in metrics.get("rejection_reasons", [])]
        brightness = _float_value(metrics.get("brightness"))
        sharpness = _float_value(metrics.get("sharpness"))
        if "too_dark" in reasons or (brightness is not None and brightness < 18):
            issues.append(
                MontageQualityIssue(
                    severity="warning",
                    code="selected_dark_scene",
                    message="A selected scene appears too dark.",
                    scene_id=scene.id,
                    clip_index=index,
                )
            )
        if "blurred" in reasons or (sharpness is not None and sharpness < 24):
            issues.append(
                MontageQualityIssue(
                    severity="warning",
                    code="selected_blurred_scene",
                    message="A selected scene appears blurred.",
                    scene_id=scene.id,
                    clip_index=index,
                )
            )
    return issues


def _report_score(issues: list[MontageQualityIssue]) -> float:
    penalty = 0
    for issue in issues:
        if issue.severity == "critical":
            penalty += 35
        elif issue.severity == "warning":
            penalty += 12
        else:
            penalty += 3
    return max(0.0, 100.0 - penalty)


def _duration_ratio(plan: QuickMontagePlan) -> float:
    target = max(plan.settings.target_duration_seconds, 0.001)
    return min(1.0, plan.total_duration_seconds / target)


def _source_counts(clips: list[MontageClip]) -> dict[UUID, int]:
    counts: dict[UUID, int] = {}
    for clip in clips:
        counts[clip.asset_id] = counts.get(clip.asset_id, 0) + 1
    return counts


def _window_selection(clips: list[MontageClip]) -> dict[str, int]:
    counts = {
        "highlight": 0,
        "visual": 0,
        "center": 0,
        "other": 0,
    }
    for clip in clips:
        reason = clip.selection_reason.casefold()
        if "highlight window" in reason:
            counts["highlight"] += 1
        elif "visual" in reason:
            counts["visual"] += 1
        elif "center of scene" in reason:
            counts["center"] += 1
        else:
            counts["other"] += 1
    return counts


def _event_id(scene: Scene) -> str | None:
    value = scene.metadata.get("event_id")
    return str(value) if value else None


def _average(values: Iterable[object]) -> float | None:
    items = [float(value) for value in values if isinstance(value, int | float)]
    return sum(items) / len(items) if items else None


def _float_value(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None
