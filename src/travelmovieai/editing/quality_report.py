"""Quality gates for planned quick montage timelines."""

import json
import math
import re
import shutil
import struct
import subprocess
import wave
from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TypedDict
from uuid import UUID

from travelmovieai.core.exceptions import DependencyUnavailableError, MontageError
from travelmovieai.core.security import sanitize_process_error
from travelmovieai.domain.enums import MediaType
from travelmovieai.domain.models import (
    MontageClip,
    MontageQualityIssue,
    MontageQualityReport,
    QuickMontagePlan,
    RenderedMediaMetrics,
    Scene,
)
from travelmovieai.editing.timeline import semantic_score_threshold
from travelmovieai.story.editorial import is_generic_caption, is_generic_title


class _RenderedProbe(TypedDict):
    duration: float
    has_video: bool
    has_audio: bool
    video_duration: float | None
    audio_duration: float | None


_ScanFailureReason = Literal[
    "not_requested",
    "process_unavailable",
    "timeout",
    "ffmpeg_error",
]


MUSIC_FADE_OUT_SECONDS = 1.5
_WindowSource = Literal[
    "vision_highlight",
    "visual_quality",
    "speech",
    "people",
    "center",
    "scene_bounds",
    "other",
]


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
    selected_events = {clip.event_id for clip in plan.clips if clip.event_id is not None}
    total_events = {_event_id(scene) for scene in scenes if _event_id(scene) is not None}
    source_counts = _source_counts(plan.clips)
    photo_clip_count = sum(clip.media_type is MediaType.PHOTO for clip in plan.clips)
    photo_duration = sum(
        clip.duration_seconds for clip in plan.clips if clip.media_type is MediaType.PHOTO
    )
    photo_duration_ratio = (
        min(1.0, photo_duration / plan.total_duration_seconds)
        if plan.total_duration_seconds > 0
        else 0.0
    )
    event_counts = Counter(str(clip.event_id) for clip in plan.clips if clip.event_id is not None)
    role_counts = Counter(
        _story_role(scenes_by_id.get(clip.scene_id))
        for clip in plan.clips
        if clip.scene_id is not None and clip.scene_id in scenes_by_id
    )
    source_count = len(source_counts)
    dominant_source_ratio = (
        max(source_counts.values()) / len(plan.clips) if plan.clips and source_counts else 0.0
    )
    semantic_scores = _numeric_values(clip.semantic_score for clip in plan.clips)
    effective_semantic_threshold = semantic_score_threshold(scenes, plan.settings)
    quality_scores = _numeric_values(scene.quality_score for scene in selected_scenes)
    average_semantic_score = _average(semantic_scores)
    average_quality_score = _average(quality_scores)
    dominant_event_ratio = _dominant_ratio(event_counts, len(plan.clips))
    dominant_role_ratio = _dominant_ratio(role_counts, len(selected_scenes))
    adjacent_source_repeat_count = sum(
        first.asset_id == second.asset_id
        for first, second in zip(plan.clips, plan.clips[1:], strict=False)
    )
    adjacent_source_repeat_ratio = (
        adjacent_source_repeat_count / (len(plan.clips) - 1) if len(plan.clips) > 1 else 0.0
    )
    window_selection = _window_selection(plan.clips, scenes_by_id)
    center_cut_ratio = window_selection.get("center", 0) / len(plan.clips) if plan.clips else 0.0
    generic_caption_count = sum(
        is_generic_caption(clip.caption) for clip in plan.clips if clip.caption
    )
    generic_caption_ratio = generic_caption_count / len(plan.clips) if plan.clips else 0.0
    generic_title_count = sum(
        is_generic_title(clip.event_title) for clip in plan.clips if clip.event_title
    )
    music_stats = _music_source_stats(plan)
    issues = _quality_issues(
        plan,
        selected_scenes,
        source_count=source_count,
        dominant_source_ratio=dominant_source_ratio,
        dominant_event_ratio=dominant_event_ratio,
        dominant_role_ratio=dominant_role_ratio,
        adjacent_source_repeat_ratio=adjacent_source_repeat_ratio,
        average_semantic_score=average_semantic_score,
        semantic_score_p10=_percentile(semantic_scores, 0.10),
        effective_semantic_threshold=effective_semantic_threshold,
        average_quality_score=average_quality_score,
        quality_score_p10=_percentile(quality_scores, 0.10),
        window_selection=window_selection,
        generic_caption_ratio=generic_caption_ratio,
        generic_title_count=generic_title_count,
        selected_event_count=len(selected_events),
        total_event_count=len(total_events),
        music_stats=music_stats,
    )
    event_coverage = (
        len(selected_events) / len(total_events)
        if total_events
        else 1.0
        if selected_events or not scenes
        else 0.0
    )
    return MontageQualityReport(
        created_at=datetime.now(UTC),
        gate_status=_gate_status(issues),
        score=_report_score(issues),
        target_duration_seconds=plan.settings.target_duration_seconds,
        planned_duration_seconds=plan.total_duration_seconds,
        duration_ratio=_duration_ratio(plan),
        clip_count=len(plan.clips),
        photo_clip_count=photo_clip_count,
        photo_duration_ratio=photo_duration_ratio,
        selected_scene_count=len(selected_scenes),
        selected_event_count=len(selected_events),
        total_event_count=len(total_events),
        event_coverage_ratio=event_coverage,
        source_count=source_count,
        dominant_source_ratio=dominant_source_ratio,
        dominant_event_ratio=dominant_event_ratio,
        dominant_role_ratio=dominant_role_ratio,
        adjacent_source_repeat_count=adjacent_source_repeat_count,
        adjacent_source_repeat_ratio=adjacent_source_repeat_ratio,
        average_semantic_score=average_semantic_score,
        minimum_semantic_score=min(semantic_scores) if semantic_scores else None,
        semantic_score_p10=_percentile(semantic_scores, 0.10),
        median_semantic_score=_percentile(semantic_scores, 0.50),
        effective_semantic_threshold=effective_semantic_threshold,
        average_quality_score=average_quality_score,
        minimum_quality_score=min(quality_scores) if quality_scores else None,
        quality_score_p10=_percentile(quality_scores, 0.10),
        median_quality_score=_percentile(quality_scores, 0.50),
        window_selection=window_selection,
        center_cut_ratio=center_cut_ratio,
        generic_caption_count=generic_caption_count,
        generic_caption_ratio=generic_caption_ratio,
        generic_title_count=generic_title_count,
        music_mode=plan.music_plan.mode if plan.music_plan else None,
        music_duration_seconds=(
            plan.music_plan.duration_seconds if plan.music_plan is not None else None
        ),
        music_accent_count=len(plan.music_plan.accents) if plan.music_plan else 0,
        music_cue_section_count=len(plan.music_plan.cue_sections) if plan.music_plan else 0,
        music_beat_count=len(plan.music_plan.beat_grid) if plan.music_plan else 0,
        music_loudness_rms=music_stats.get("rms"),
        music_peak_ratio=music_stats.get("peak_ratio"),
        music_clipping_ratio=music_stats.get("clipping_ratio"),
        issues=issues,
    )


def enrich_montage_quality_report_with_render(
    report: MontageQualityReport,
    output_path: Path,
    *,
    ffprobe_binary: str = "ffprobe",
    ffmpeg_binary: str = "ffmpeg",
    timeout_seconds: float = 7200,
    full_scan: bool = True,
    require_full_scan: bool = False,
) -> MontageQualityReport:
    probe = _probe_rendered_movie(output_path, ffprobe_binary, timeout_seconds)
    audio_rms = _rendered_audio_rms(
        output_path,
        duration_seconds=probe["duration"],
        has_audio=probe["has_audio"],
        ffmpeg_binary=ffmpeg_binary,
        timeout_seconds=timeout_seconds,
    )
    video_luma = _rendered_video_luma(
        output_path,
        duration_seconds=probe["duration"],
        has_video=probe["has_video"],
        ffmpeg_binary=ffmpeg_binary,
        timeout_seconds=timeout_seconds,
    )
    media_metrics = (
        _full_duration_media_metrics(
            output_path,
            probe=probe,
            ffmpeg_binary=ffmpeg_binary,
            timeout_seconds=timeout_seconds,
        )
        if full_scan
        else _probe_media_metrics(probe, failure_reason="not_requested")
    )
    issues = [
        *report.issues,
        *_render_issues(
            report,
            rendered_duration=probe["duration"],
            has_video=probe["has_video"],
            has_audio=probe["has_audio"],
            audio_rms=audio_rms,
            video_luma=video_luma,
            media_metrics=media_metrics,
            require_full_scan=require_full_scan,
        ),
    ]
    return report.model_copy(
        update={
            "score": _report_score(issues),
            "gate_status": _gate_status(issues),
            "rendered_path": output_path.resolve(),
            "rendered_duration_seconds": probe["duration"],
            "rendered_duration_delta_seconds": probe["duration"] - report.planned_duration_seconds,
            "rendered_has_video": probe["has_video"],
            "rendered_has_audio": probe["has_audio"],
            "rendered_audio_rms": audio_rms,
            "rendered_video_luma": video_luma,
            "rendered_media_metrics": media_metrics,
            "issues": issues,
        }
    )


def enforce_montage_quality(report: MontageQualityReport) -> None:
    critical = [issue for issue in report.issues if issue.severity == "critical"]
    if not critical:
        return
    details = "; ".join(issue.message for issue in critical[:3])
    raise MontageError(f"Rendered movie failed the quality gate: {details}")


def _quality_issues(
    plan: QuickMontagePlan,
    selected_scenes: list[Scene],
    *,
    source_count: int,
    dominant_source_ratio: float,
    dominant_event_ratio: float,
    dominant_role_ratio: float,
    adjacent_source_repeat_ratio: float,
    average_semantic_score: float | None,
    semantic_score_p10: float | None,
    effective_semantic_threshold: float,
    average_quality_score: float | None,
    quality_score_p10: float | None,
    window_selection: dict[str, int],
    generic_caption_ratio: float,
    generic_title_count: int,
    selected_event_count: int,
    total_event_count: int,
    music_stats: dict[str, float],
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
    if total_event_count > 1 and selected_event_count > 1 and dominant_event_ratio > 0.55:
        issues.append(
            MontageQualityIssue(
                severity="warning",
                code="event_dominance",
                message="One detected event occupies too much of the selected timeline.",
            )
        )
    selected_roles = {_story_role(scene) for scene in selected_scenes}
    if len(selected_scenes) >= 4 and len(selected_roles) > 1 and dominant_role_ratio > 0.60:
        issues.append(
            MontageQualityIssue(
                severity="warning",
                code="story_role_dominance",
                message="One story role dominates the selected scenes.",
            )
        )
    if len(plan.clips) >= 4 and adjacent_source_repeat_ratio > 0.25:
        issues.append(
            MontageQualityIssue(
                severity="warning",
                code="adjacent_source_repetition",
                message="Adjacent clips repeat the same source too often.",
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
    if semantic_score_p10 is not None and semantic_score_p10 < effective_semantic_threshold:
        issues.append(
            MontageQualityIssue(
                severity="warning",
                code="low_semantic_score_tail",
                message=(
                    "The lowest-scoring selected scenes fall below the configured semantic "
                    "threshold."
                ),
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
    if quality_score_p10 is not None and quality_score_p10 < plan.settings.min_quality_score:
        issues.append(
            MontageQualityIssue(
                severity="warning",
                code="low_visual_quality_tail",
                message=(
                    "The lowest-quality selected scenes fall below the configured visual threshold."
                ),
            )
        )
    if plan.selection_mode == "semantic" and len(plan.clips) >= 4:
        center_ratio = window_selection.get("center", 0) / len(plan.clips)
        if center_ratio > 0.55:
            issues.append(
                MontageQualityIssue(
                    severity="warning",
                    code="excessive_center_cuts",
                    message=(
                        "Many selected clips fall back to center cuts instead of "
                        "explicit highlights."
                    ),
                )
            )
    if len(plan.clips) >= 4 and generic_caption_ratio > 0.35:
        issues.append(
            MontageQualityIssue(
                severity="warning",
                code="generic_scene_captions",
                message="Many selected scene captions contain model boilerplate.",
            )
        )
    if generic_title_count:
        issues.append(
            MontageQualityIssue(
                severity=(
                    "warning"
                    if plan.settings.text_overlays_enabled and plan.settings.event_titles_enabled
                    else "info"
                ),
                code="generic_event_titles",
                message="Generated event titles contain generic or weak visual guesses.",
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
    if (
        plan.music_plan is not None
        and plan.music_plan.mode != "none"
        and not plan.music_plan.cue_sections
    ):
        issues.append(
            MontageQualityIssue(
                severity="warning",
                code="missing_music_cue_sections",
                message="Music plan does not contain arrangement cue sections.",
            )
        )
    if (
        plan.music_plan is not None
        and plan.music_plan.mode != "none"
        and not plan.music_plan.beat_grid
    ):
        issues.append(
            MontageQualityIssue(
                severity="info",
                code="missing_music_beat_grid",
                message="Music plan does not contain beat grid metadata.",
            )
        )
    beat_alignment_ratio = _beat_alignment_ratio(plan)
    if beat_alignment_ratio is not None and beat_alignment_ratio < 0.5:
        issues.append(
            MontageQualityIssue(
                severity="warning",
                code="unsynced_music_cuts",
                message="Most scene cuts are not aligned to strong music beats.",
            )
        )
    if (clipping := music_stats.get("clipping_ratio")) is not None and clipping > 0.002:
        issues.append(
            MontageQualityIssue(
                severity="warning",
                code="music_source_clipping",
                message="Music source appears clipped before rendering.",
            )
        )
    if (rms := music_stats.get("rms")) is not None and rms < 80:
        issues.append(
            MontageQualityIssue(
                severity="info",
                code="quiet_music_source",
                message="Music source is very quiet before rendering.",
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
    issues.extend(_speech_cut_issues(plan, selected_scenes))
    return issues


def _render_issues(
    report: MontageQualityReport,
    *,
    rendered_duration: float,
    has_video: bool,
    has_audio: bool,
    audio_rms: dict[str, float],
    video_luma: dict[str, float],
    media_metrics: RenderedMediaMetrics,
    require_full_scan: bool,
) -> list[MontageQualityIssue]:
    issues: list[MontageQualityIssue] = []
    if not has_video:
        issues.append(
            MontageQualityIssue(
                severity="critical",
                code="render_missing_video",
                message="Rendered movie does not contain a video stream.",
            )
        )
    if not has_audio:
        issues.append(
            MontageQualityIssue(
                severity="critical",
                code="render_missing_audio",
                message="Rendered movie does not contain an audio stream.",
            )
        )
    delta = abs(rendered_duration - report.planned_duration_seconds)
    if delta > max(0.35, report.planned_duration_seconds * 0.03):
        issues.append(
            MontageQualityIssue(
                severity="warning",
                code="render_duration_mismatch",
                message=(
                    f"Rendered duration differs from the planned timeline by {delta:.2f} seconds."
                ),
            )
        )
    if has_audio and audio_rms:
        audible_segments = sum(1 for value in audio_rms.values() if value >= 50)
        if audible_segments == 0:
            issues.append(
                MontageQualityIssue(
                    severity="critical",
                    code="render_silent_audio",
                    message="Rendered movie audio is silent in sampled sections.",
                )
            )
        elif audio_rms.get("end", 0.0) < 50 and report.music_mode not in {None, "none"}:
            issues.append(
                MontageQualityIssue(
                    severity="warning",
                    code="render_audio_fades_out_early",
                    message="Rendered movie audio is very quiet near the end.",
                )
            )
    elif has_audio:
        issues.append(
            MontageQualityIssue(
                severity="info",
                code="render_audio_rms_unavailable",
                message="Rendered audio RMS could not be measured.",
            )
        )
    if has_video and video_luma:
        if any(value < 3 for value in video_luma.values()):
            issues.append(
                MontageQualityIssue(
                    severity="warning",
                    code="render_black_video_sample",
                    message="Rendered movie contains a sampled frame that is nearly black.",
                )
            )
        if max(video_luma.values()) - min(video_luma.values()) > 65:
            issues.append(
                MontageQualityIssue(
                    severity="info",
                    code="render_exposure_jump",
                    message="Rendered movie has a large exposure jump across sampled sections.",
                )
            )
    if (
        media_metrics.av_duration_delta_seconds is not None
        and media_metrics.av_duration_delta_seconds > 0.10
    ):
        issues.append(
            MontageQualityIssue(
                severity="warning",
                code="render_av_duration_mismatch",
                message="Rendered audio and video stream durations differ noticeably.",
            )
        )
    if not media_metrics.scan_completed:
        issues.append(
            MontageQualityIssue(
                severity="critical" if require_full_scan else "warning",
                code="render_full_scan_unavailable",
                message=(
                    "Full-duration black, freeze, silence, and loudness checks are unavailable"
                    + (
                        f" ({media_metrics.scan_failure_reason.replace('_', ' ')})."
                        if media_metrics.scan_failure_reason
                        else "."
                    )
                ),
            )
        )
        return issues

    if (black_ratio := media_metrics.black_ratio) is not None and black_ratio > 0.01:
        issues.append(
            MontageQualityIssue(
                severity="critical" if black_ratio > 0.20 else "warning",
                code="render_black_duration",
                message="Rendered movie contains a material amount of black video.",
            )
        )
    freeze_ratio = media_metrics.freeze_ratio
    unexpected_freeze_ratio = (
        max(0.0, freeze_ratio - report.photo_duration_ratio) if freeze_ratio is not None else None
    )
    if unexpected_freeze_ratio is not None and unexpected_freeze_ratio > 0.05:
        issues.append(
            MontageQualityIssue(
                severity="critical" if unexpected_freeze_ratio > 0.45 else "warning",
                code="render_freeze_duration",
                message="Rendered movie contains frozen video beyond planned photo holds.",
            )
        )
    if (
        (silence_ratio := media_metrics.silence_ratio) is not None
        and silence_ratio > 0.02
        and (media_metrics.silence_duration_seconds or 0) > 1.0
    ):
        issues.append(
            MontageQualityIssue(
                severity="critical" if silence_ratio > 0.95 else "warning",
                code="render_silence_duration",
                message="Rendered movie contains unexpectedly long silent audio sections.",
            )
        )
    loudness = media_metrics.integrated_loudness_lufs
    if loudness is not None and (loudness < -26 or loudness > -12):
        issues.append(
            MontageQualityIssue(
                severity="warning",
                code="render_loudness_out_of_range",
                message="Integrated movie loudness is outside the expected delivery range.",
            )
        )
    if media_metrics.true_peak_dbfs is not None and media_metrics.true_peak_dbfs > -1.0:
        issues.append(
            MontageQualityIssue(
                severity="warning",
                code="render_true_peak_high",
                message="Rendered movie true peak is too close to clipping.",
            )
        )
    return issues


def _probe_rendered_movie(
    output_path: Path,
    ffprobe_binary: str,
    timeout_seconds: float,
) -> _RenderedProbe:
    resolved = shutil.which(ffprobe_binary) or ffprobe_binary
    try:
        completed = subprocess.run(
            [
                resolved,
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(output_path),
            ],
            capture_output=True,
            check=False,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
    except FileNotFoundError as error:
        raise DependencyUnavailableError(
            f"FFprobe executable was not found: {ffprobe_binary}"
        ) from error
    except subprocess.TimeoutExpired as error:
        raise MontageError(
            f"FFprobe timed out after {timeout_seconds:g}s while validating the final movie."
        ) from error
    if completed.returncode != 0:
        detail = sanitize_process_error(
            completed.stderr,
            private_paths=[output_path],
            fallback="unknown FFprobe error",
        )
        raise MontageError(f"Could not validate the final movie: {detail}")
    try:
        payload = json.loads(completed.stdout)
        streams = payload.get("streams", [])
        stream_types = {stream.get("codec_type") for stream in streams}
        duration = float(payload.get("format", {}).get("duration", 0))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise MontageError("FFprobe returned invalid final movie data.") from error
    return {
        "duration": max(0.0, duration),
        "has_video": "video" in stream_types,
        "has_audio": "audio" in stream_types,
        "video_duration": _stream_duration(streams, "video"),
        "audio_duration": _stream_duration(streams, "audio"),
    }


def _stream_duration(streams: object, codec_type: str) -> float | None:
    if not isinstance(streams, list):
        return None
    for stream in streams:
        if not isinstance(stream, dict) or stream.get("codec_type") != codec_type:
            continue
        value = _float_value(stream.get("duration"))
        if value is not None and value >= 0:
            return value
    return None


def _probe_media_metrics(
    probe: _RenderedProbe,
    *,
    failure_reason: _ScanFailureReason | None = None,
) -> RenderedMediaMetrics:
    video_duration = probe["video_duration"]
    audio_duration = probe["audio_duration"]
    return RenderedMediaMetrics(
        scan_completed=False,
        scan_failure_reason=failure_reason,
        av_duration_delta_seconds=(
            abs(video_duration - audio_duration)
            if video_duration is not None and audio_duration is not None
            else None
        ),
    )


def _full_duration_media_metrics(
    output_path: Path,
    *,
    probe: _RenderedProbe,
    ffmpeg_binary: str,
    timeout_seconds: float,
) -> RenderedMediaMetrics:
    base = _probe_media_metrics(probe)
    resolved = shutil.which(ffmpeg_binary) or ffmpeg_binary
    command = [
        resolved,
        "-hide_banner",
        "-nostats",
        "-i",
        str(output_path),
    ]
    if probe["has_video"]:
        command.extend(
            [
                "-vf",
                "blackdetect=d=0.4:pix_th=0.02,freezedetect=n=-50dB:d=1",
            ]
        )
    if probe["has_audio"]:
        command.extend(
            [
                "-af",
                "silencedetect=noise=-50dB:d=0.4,ebur128=peak=true",
            ]
        )
    command.extend(["-f", "null", "-"])
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            check=False,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return base.model_copy(update={"scan_failure_reason": "timeout"})
    except OSError:
        return base.model_copy(update={"scan_failure_reason": "process_unavailable"})
    if completed.returncode != 0:
        return base.model_copy(update={"scan_failure_reason": "ffmpeg_error"})

    duration = max(0.001, probe["duration"])
    stderr = completed.stderr
    black_duration = _sum_metric(stderr, r"black_duration:\s*([0-9]+(?:\.[0-9]+)?)")
    freeze_duration = _sum_metric(stderr, r"freeze_duration:\s*([0-9]+(?:\.[0-9]+)?)")
    silence_duration = _sum_metric(stderr, r"silence_duration:\s*([0-9]+(?:\.[0-9]+)?)")
    black_duration = min(duration, black_duration)
    freeze_duration = min(duration, freeze_duration)
    silence_duration = min(duration, silence_duration)
    return base.model_copy(
        update={
            "scan_completed": True,
            "scan_failure_reason": None,
            "black_duration_seconds": black_duration if probe["has_video"] else None,
            "black_ratio": black_duration / duration if probe["has_video"] else None,
            "freeze_duration_seconds": freeze_duration if probe["has_video"] else None,
            "freeze_ratio": freeze_duration / duration if probe["has_video"] else None,
            "silence_duration_seconds": silence_duration if probe["has_audio"] else None,
            "silence_ratio": silence_duration / duration if probe["has_audio"] else None,
            "integrated_loudness_lufs": _last_metric(
                stderr,
                r"\bI:\s*(-?[0-9]+(?:\.[0-9]+)?)\s*LUFS",
            ),
            "loudness_range_lu": _last_metric(
                stderr,
                r"\bLRA:\s*([0-9]+(?:\.[0-9]+)?)\s*LU",
            ),
            "true_peak_dbfs": _last_metric(
                stderr,
                r"\bPeak:\s*(-?[0-9]+(?:\.[0-9]+)?)\s*dBFS",
            ),
        }
    )


def _sum_metric(text: str, pattern: str) -> float:
    return sum(float(value) for value in re.findall(pattern, text, flags=re.IGNORECASE))


def _last_metric(text: str, pattern: str) -> float | None:
    values = re.findall(pattern, text, flags=re.IGNORECASE)
    return float(values[-1]) if values else None


def _rendered_audio_rms(
    output_path: Path,
    *,
    duration_seconds: object,
    has_audio: object,
    ffmpeg_binary: str,
    timeout_seconds: float,
) -> dict[str, float]:
    if not has_audio or not isinstance(duration_seconds, int | float) or duration_seconds <= 0:
        return {}
    sample_duration = min(0.5, max(0.2, duration_seconds / 8))
    starts = {
        "start": 0.0,
        "middle": max(0.0, duration_seconds * 0.5 - sample_duration / 2),
    }
    values: dict[str, float] = {}
    for label, start in starts.items():
        rms = _audio_rms(
            output_path,
            start_seconds=start,
            duration_seconds=sample_duration,
            ffmpeg_binary=ffmpeg_binary,
            timeout_seconds=timeout_seconds,
        )
        if rms is not None:
            values[label] = rms
    end_latest = max(
        0.0,
        duration_seconds - MUSIC_FADE_OUT_SECONDS - sample_duration - 0.05,
    )
    end_samples = []
    for start in sorted({max(0.0, end_latest - offset) for offset in (0.0, 1.5, 3.0)}):
        rms = _audio_rms(
            output_path,
            start_seconds=start,
            duration_seconds=sample_duration,
            ffmpeg_binary=ffmpeg_binary,
            timeout_seconds=timeout_seconds,
        )
        if rms is not None:
            end_samples.append(rms)
    if end_samples:
        values["end"] = max(end_samples)
    return values


def _audio_rms(
    output_path: Path,
    *,
    start_seconds: float,
    duration_seconds: float,
    ffmpeg_binary: str,
    timeout_seconds: float,
) -> float | None:
    resolved = shutil.which(ffmpeg_binary) or ffmpeg_binary
    try:
        completed = subprocess.run(
            [
                resolved,
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{start_seconds:.3f}",
                "-t",
                f"{duration_seconds:.3f}",
                "-i",
                str(output_path),
                "-vn",
                "-f",
                "s16le",
                "-ac",
                "1",
                "-ar",
                "8000",
                "pipe:1",
            ],
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    sample_count = len(completed.stdout) // 2
    if sample_count <= 0:
        return 0.0
    samples = struct.unpack(f"<{sample_count}h", completed.stdout[: sample_count * 2])
    return math.sqrt(sum(sample * sample for sample in samples) / sample_count)


def _rendered_video_luma(
    output_path: Path,
    *,
    duration_seconds: object,
    has_video: object,
    ffmpeg_binary: str,
    timeout_seconds: float,
) -> dict[str, float]:
    if not has_video or not isinstance(duration_seconds, int | float) or duration_seconds <= 0:
        return {}
    starts = {
        "start": min(0.05, max(0.0, duration_seconds / 10)),
        "middle": max(0.0, duration_seconds * 0.5),
        "end": max(0.0, duration_seconds - 0.08),
    }
    values: dict[str, float] = {}
    for label, start in starts.items():
        luma = _video_luma(
            output_path,
            start_seconds=start,
            ffmpeg_binary=ffmpeg_binary,
            timeout_seconds=timeout_seconds,
        )
        if luma is not None:
            values[label] = luma
    return values


def _video_luma(
    output_path: Path,
    *,
    start_seconds: float,
    ffmpeg_binary: str,
    timeout_seconds: float,
) -> float | None:
    resolved = shutil.which(ffmpeg_binary) or ffmpeg_binary
    width = 16
    height = 16
    try:
        completed = subprocess.run(
            [
                resolved,
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{start_seconds:.3f}",
                "-i",
                str(output_path),
                "-frames:v",
                "1",
                "-vf",
                f"scale={width}:{height},format=gray",
                "-f",
                "rawvideo",
                "pipe:1",
            ],
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    expected = width * height
    if completed.returncode != 0 or len(completed.stdout) < expected:
        return None
    return sum(completed.stdout[:expected]) / expected / 255 * 100


def _music_source_stats(plan: QuickMontagePlan) -> dict[str, float]:
    music_plan = plan.music_plan
    if music_plan is None or music_plan.source_path is None or music_plan.mode == "none":
        return {}
    path = music_plan.source_path
    if path.suffix.casefold() != ".wav" or not path.is_file():
        return {}
    try:
        with wave.open(str(path), "rb") as audio:
            sample_width = audio.getsampwidth()
            channels = audio.getnchannels()
            frame_count = audio.getnframes()
            if sample_width != 2 or channels <= 0 or frame_count <= 0:
                return {}
            total_samples = 0
            squares = 0
            peak = 0
            clipped = 0
            chunk_frames = max(1, audio.getframerate() * 4)
            while True:
                raw = audio.readframes(chunk_frames)
                sample_count = len(raw) // 2
                if sample_count <= 0:
                    break
                samples = struct.unpack(f"<{sample_count}h", raw[: sample_count * 2])
                total_samples += sample_count
                squares += sum(sample * sample for sample in samples)
                peak = max(peak, max(abs(sample) for sample in samples))
                clipped += sum(1 for sample in samples if abs(sample) >= 32760)
    except (OSError, wave.Error):
        return {}
    if total_samples <= 0:
        return {}
    return {
        "rms": math.sqrt(squares / total_samples),
        "peak_ratio": peak / 32767,
        "clipping_ratio": clipped / total_samples,
    }


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


def _window_selection(
    clips: list[MontageClip],
    scenes_by_id: dict[UUID, Scene],
) -> dict[str, int]:
    counts = {
        "highlight": 0,
        "visual": 0,
        "speech": 0,
        "people": 0,
        "center": 0,
        "scene_bounds": 0,
        "other": 0,
    }
    for clip in clips:
        source = clip.window_source
        if source == "other":
            scene = scenes_by_id.get(clip.scene_id) if clip.scene_id is not None else None
            source = _legacy_geometric_window_source(clip, scene)
        if source == "vision_highlight":
            counts["highlight"] += 1
        elif source == "visual_quality":
            counts["visual"] += 1
        elif source == "speech":
            counts["speech"] += 1
        elif source == "people":
            counts["people"] += 1
        elif source == "center":
            counts["center"] += 1
        elif source == "scene_bounds":
            counts["scene_bounds"] += 1
        else:
            counts["other"] += 1
    return counts


def _legacy_geometric_window_source(
    clip: MontageClip,
    scene: Scene | None,
) -> _WindowSource:
    if scene is None or clip.media_type.value != "video":
        return "other"
    available = max(0.0, scene.end_seconds - scene.start_seconds)
    if available <= clip.duration_seconds + 0.05:
        return "scene_bounds"
    expected_center = scene.start_seconds + (available - clip.duration_seconds) / 2
    tolerance = max(0.08, clip.duration_seconds * 0.02)
    if abs(clip.source_start_seconds - expected_center) <= tolerance:
        return "center"
    return "other"


def _speech_cut_issues(
    plan: QuickMontagePlan,
    selected_scenes: list[Scene],
) -> list[MontageQualityIssue]:
    scenes_by_id = {scene.id: scene for scene in selected_scenes}
    issues: list[MontageQualityIssue] = []
    for index, clip in enumerate(plan.clips):
        if clip.scene_id is None:
            continue
        scene = scenes_by_id.get(clip.scene_id)
        if scene is None:
            continue
        relative_start = max(0.0, clip.source_start_seconds - scene.start_seconds)
        relative_end = relative_start + clip.duration_seconds
        if _cuts_speech_segment(scene, relative_start, relative_end):
            issues.append(
                MontageQualityIssue(
                    severity="warning",
                    code="speech_boundary_cut",
                    message="A selected clip starts or ends inside a speech segment.",
                    scene_id=scene.id,
                    clip_index=index,
                )
            )
    return issues


def _cuts_speech_segment(
    scene: Scene,
    relative_start_seconds: float,
    relative_end_seconds: float,
) -> bool:
    segments = scene.metadata.get("speech_segments", [])
    if not isinstance(segments, list):
        return False
    for item in segments:
        if not isinstance(item, dict):
            continue
        start = _float_value(item.get("start_seconds"))
        end = _float_value(item.get("end_seconds"))
        if start is None or end is None or end <= start:
            continue
        protected_start = start + 0.16
        protected_end = end - 0.16
        if protected_start < relative_start_seconds < protected_end:
            return True
        if protected_start < relative_end_seconds < protected_end:
            return True
    return False


def _beat_alignment_ratio(plan: QuickMontagePlan) -> float | None:
    music_plan = plan.music_plan
    if (
        music_plan is None
        or not plan.settings.music_sync
        or len(plan.clips) < 3
        or not music_plan.beat_grid
    ):
        return None
    strong_beats = [
        beat.time_seconds
        for beat in music_plan.beat_grid
        if beat.strength >= 0.68
        or beat.nearest_accent_kind in {"scene_change", "event_change", "highlight"}
    ]
    if not strong_beats:
        return None
    starts = _clip_starts(plan)[1:]
    if not starts:
        return None
    aligned = sum(
        1 for start in starts if min(abs(start - beat_time) for beat_time in strong_beats) <= 0.1
    )
    return aligned / len(starts)


def _clip_starts(plan: QuickMontagePlan) -> list[float]:
    transition = _effective_transition_duration(plan)
    starts: list[float] = []
    elapsed = 0.0
    for index, clip in enumerate(plan.clips):
        starts.append(elapsed)
        if index < len(plan.clips) - 1:
            overlap = transition if _clip_uses_transition(plan, index + 1) else 0.0
            elapsed += clip.duration_seconds - overlap
    return starts


def _effective_transition_duration(plan: QuickMontagePlan) -> float:
    if len(plan.clips) < 2 or not any(
        _clip_uses_transition(plan, index) for index in range(1, len(plan.clips))
    ):
        return 0.0
    return min(
        plan.settings.transition_duration_seconds,
        min(clip.duration_seconds for clip in plan.clips) * 0.45,
    )


def _clip_uses_transition(plan: QuickMontagePlan, clip_index: int) -> bool:
    settings = plan.settings
    if settings.transition == "none" or settings.transition_duration_seconds <= 0:
        return False
    if settings.transition == "cinematic":
        return plan.clips[clip_index].transition == "fade"
    return settings.transition in {"fade", "wipeleft", "slideright"}


def _event_id(scene: Scene) -> str | None:
    value = scene.metadata.get("event_id")
    return str(value) if value else None


def _story_role(scene: Scene | None) -> str:
    if scene is None:
        return "unassigned"
    value = str(scene.metadata.get("story_section_role", "")).casefold()
    return value if value in {"opening", "journey", "highlight", "finale"} else "unassigned"


def _dominant_ratio(counts: Mapping[str, int], total: int) -> float:
    return max(counts.values(), default=0) / total if total > 0 else 0.0


def _numeric_values(values: Iterable[object]) -> list[float]:
    return [float(value) for value in values if isinstance(value, int | float)]


def _average(values: Iterable[object]) -> float | None:
    items = _numeric_values(values)
    return sum(items) / len(items) if items else None


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = max(0.0, min(1.0, fraction)) * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _gate_status(
    issues: list[MontageQualityIssue],
) -> Literal["passed", "degraded", "failed"]:
    if any(issue.severity == "critical" for issue in issues):
        return "failed"
    if any(issue.severity == "warning" for issue in issues):
        return "degraded"
    return "passed"


def _float_value(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None
