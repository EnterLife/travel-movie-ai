"""Quality gates for planned quick montage timelines."""

import json
import math
import shutil
import struct
import subprocess
import wave
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict
from uuid import UUID

from travelmovieai.core.exceptions import DependencyUnavailableError, MontageError
from travelmovieai.domain.models import (
    MontageClip,
    MontageQualityIssue,
    MontageQualityReport,
    QuickMontagePlan,
    Scene,
)


class _RenderedProbe(TypedDict):
    duration: float
    has_video: bool
    has_audio: bool


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
    music_stats = _music_source_stats(plan)
    issues = _quality_issues(
        plan,
        selected_scenes,
        source_count=source_count,
        dominant_source_ratio=dominant_source_ratio,
        average_semantic_score=average_semantic_score,
        average_quality_score=average_quality_score,
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
) -> MontageQualityReport:
    probe = _probe_rendered_movie(output_path, ffprobe_binary)
    audio_rms = _rendered_audio_rms(
        output_path,
        duration_seconds=probe["duration"],
        has_audio=probe["has_audio"],
        ffmpeg_binary=ffmpeg_binary,
    )
    video_luma = _rendered_video_luma(
        output_path,
        duration_seconds=probe["duration"],
        has_video=probe["has_video"],
        ffmpeg_binary=ffmpeg_binary,
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
        ),
    ]
    return report.model_copy(
        update={
            "score": _report_score(issues),
            "rendered_path": output_path.resolve(),
            "rendered_duration_seconds": probe["duration"],
            "rendered_duration_delta_seconds": probe["duration"] - report.planned_duration_seconds,
            "rendered_has_video": probe["has_video"],
            "rendered_has_audio": probe["has_audio"],
            "rendered_audio_rms": audio_rms,
            "rendered_video_luma": video_luma,
            "issues": issues,
        }
    )


def _quality_issues(
    plan: QuickMontagePlan,
    selected_scenes: list[Scene],
    *,
    source_count: int,
    dominant_source_ratio: float,
    average_semantic_score: float | None,
    average_quality_score: float | None,
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
    window_selection = _window_selection(plan.clips)
    if plan.selection_mode == "semantic" and len(plan.clips) >= 4:
        center_ratio = window_selection.get("center", 0) / len(plan.clips)
        if center_ratio > 0.55:
            issues.append(
                MontageQualityIssue(
                    severity="info",
                    code="excessive_center_cuts",
                    message=(
                        "Many selected clips fall back to center cuts instead of "
                        "explicit highlights."
                    ),
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
    return issues


def _probe_rendered_movie(output_path: Path, ffprobe_binary: str) -> _RenderedProbe:
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
        )
    except FileNotFoundError as error:
        raise DependencyUnavailableError(
            f"FFprobe executable was not found: {ffprobe_binary}"
        ) from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "unknown FFprobe error"
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
    }


def _rendered_audio_rms(
    output_path: Path,
    *,
    duration_seconds: object,
    has_audio: object,
    ffmpeg_binary: str,
) -> dict[str, float]:
    if not has_audio or not isinstance(duration_seconds, int | float) or duration_seconds <= 0:
        return {}
    sample_duration = min(0.5, max(0.2, duration_seconds / 8))
    starts = {
        "start": 0.0,
        "middle": max(0.0, duration_seconds * 0.5 - sample_duration / 2),
        "end": max(0.0, duration_seconds - sample_duration - 0.05),
    }
    values: dict[str, float] = {}
    for label, start in starts.items():
        rms = _audio_rms(
            output_path,
            start_seconds=start,
            duration_seconds=sample_duration,
            ffmpeg_binary=ffmpeg_binary,
        )
        if rms is not None:
            values[label] = rms
    return values


def _audio_rms(
    output_path: Path,
    *,
    start_seconds: float,
    duration_seconds: float,
    ffmpeg_binary: str,
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
        )
    except FileNotFoundError:
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
        )
        if luma is not None:
            values[label] = luma
    return values


def _video_luma(
    output_path: Path,
    *,
    start_seconds: float,
    ffmpeg_binary: str,
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
        )
    except FileNotFoundError:
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
    transition = (
        0.0 if plan.settings.transition == "none" else plan.settings.transition_duration_seconds
    )
    if plan.clips:
        transition = min(transition, min(clip.duration_seconds for clip in plan.clips) * 0.45)
    starts: list[float] = []
    elapsed = 0.0
    for index, clip in enumerate(plan.clips):
        starts.append(elapsed)
        if index < len(plan.clips) - 1:
            elapsed += clip.duration_seconds - transition
    return starts


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
