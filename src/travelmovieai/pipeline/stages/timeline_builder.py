"""Pipeline stage that builds a declarative semantic montage timeline."""

from pathlib import Path

from pydantic import ValidationError

from travelmovieai.application.context import ProjectContext
from travelmovieai.core.exceptions import MontageError
from travelmovieai.domain.enums import PipelineStage, StageStatus
from travelmovieai.domain.models import (
    MusicPlan,
    NarrationAudioCue,
    QuickMontagePlan,
    QuickMontageSettings,
    SceneSelectionReport,
    StageResult,
    SynthesizedNarrationLine,
    VoiceSynthesisReport,
)
from travelmovieai.editing.narration_audio import (
    compose_narration_track,
    narration_track_matches,
)
from travelmovieai.editing.timeline import (
    apply_music_directing,
    build_selection_report,
    build_semantic_montage_plan,
)
from travelmovieai.infrastructure.artifacts import (
    artifact_fingerprint,
    stage_cache_manifest_matches,
    write_json_atomic,
    write_stage_cache_manifest,
)
from travelmovieai.infrastructure.database import MediaAssetRepository
from travelmovieai.pipeline.base import Stage

ARTIFACT_SCHEMA_VERSION = "timeline-builder-v9-clean-overlay-captions"


class TimelineBuilderStage(Stage):
    name = PipelineStage.TIMELINE_BUILDER

    def run(self, context: ProjectContext) -> StageResult:
        repository = MediaAssetRepository(context.database_path)
        repository.initialize()
        assets = repository.list_assets()
        scenes = repository.list_scenes()
        timeline_artifact = context.artifacts_dir / "quick_timeline.json"
        decisions_artifact = context.artifacts_dir / "selection_decisions.json"
        cache_artifact = context.artifacts_dir / "quick_timeline.cache.json"
        narration_artifact = context.artifacts_dir / "narration.wav"
        if not assets or not scenes:
            timeline_artifact.unlink(missing_ok=True)
            decisions_artifact.unlink(missing_ok=True)
            cache_artifact.unlink(missing_ok=True)
            narration_artifact.unlink(missing_ok=True)
            return StageResult(
                stage=self.name,
                status=StageStatus.NO_INPUT,
                message="Timeline builder needs media assets and ranked scenes.",
            )

        settings = _semantic_montage_settings(context)
        music_artifact = context.artifacts_dir / "music_plan.json"
        music_plan = _read_music_plan(music_artifact)
        narration_report = _read_narration_audio(context, settings)
        if narration_report is None:
            narration_artifact.unlink(missing_ok=True)
        input_fingerprint = artifact_fingerprint(
            assets,
            scenes,
            music_plan,
            _file_revision(music_plan.source_path if music_plan is not None else None),
            narration_report,
            [
                _file_revision(line.audio_path)
                for line in (narration_report.lines if narration_report is not None else [])
            ],
        )
        config_fingerprint = artifact_fingerprint(settings, ARTIFACT_SCHEMA_VERSION)
        cached_plan = _read_cached_timeline_artifacts(timeline_artifact, decisions_artifact)
        if cached_plan is not None and not _cached_narration_matches(
            cached_plan,
            narration_report,
            narration_artifact,
        ):
            cached_plan = None
        cached_artifacts = [timeline_artifact, decisions_artifact]
        if cached_plan is not None and cached_plan.narration_path is not None:
            cached_artifacts.append(cached_plan.narration_path)
        if (
            stage_cache_manifest_matches(
                cache_artifact,
                stage=self.name,
                artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
                input_fingerprint=input_fingerprint,
                config_fingerprint=config_fingerprint,
                artifacts=cached_artifacts,
            )
            and cached_plan is not None
        ):
            dropped_count = _dropped_narration_line_count(narration_report, cached_plan)
            return StageResult(
                stage=self.name,
                status=StageStatus.DEGRADED if dropped_count else StageStatus.CACHED,
                cache_hit=True,
                artifacts=[*cached_artifacts, cache_artifact],
                message=(
                    "Timeline builder reused a cached timeline with "
                    f"{dropped_count} narration line(s) omitted for duration."
                    if dropped_count
                    else "Timeline builder reused cached timeline artifacts."
                ),
            )

        plan = build_semantic_montage_plan(assets, scenes, settings, music_plan)
        if music_plan is not None:
            plan = apply_music_directing(plan, scenes)
        narration_cues: list[NarrationAudioCue] = []
        dropped_count = 0
        if narration_report is not None:
            narration_cues = _schedule_narration_cues(
                narration_report.lines,
                plan.total_duration_seconds,
            )
            dropped_count = narration_report.line_count - len(narration_cues)
            if narration_cues:
                _validate_narration_timeline(narration_cues, plan.total_duration_seconds)
                compose_narration_track(narration_cues, narration_artifact)
                plan = plan.model_copy(
                    update={
                        "narration_path": narration_artifact.resolve(),
                        "narration_cues": narration_cues,
                    }
                )
            else:
                narration_artifact.unlink(missing_ok=True)
                plan = plan.model_copy(
                    update={
                        "settings": plan.settings.model_copy(update={"narration_enabled": False})
                    }
                )
        write_json_atomic(timeline_artifact, plan)
        write_json_atomic(
            decisions_artifact,
            build_selection_report(scenes, plan, settings),
        )
        output_artifacts = [timeline_artifact, decisions_artifact]
        if plan.narration_path is not None:
            output_artifacts.append(plan.narration_path)
        write_stage_cache_manifest(
            cache_artifact,
            stage=self.name,
            artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
            input_fingerprint=input_fingerprint,
            config_fingerprint=config_fingerprint,
            artifacts=output_artifacts,
        )
        repository.record_timeline_version(
            plan,
            phase="built",
            variant_name=context.variant_name,
            variant_slug=context.variant_slug,
        )
        return StageResult(
            stage=self.name,
            status=StageStatus.DEGRADED if dropped_count else StageStatus.COMPLETED,
            artifacts=[*output_artifacts, cache_artifact],
            message=(
                f"Timeline builder selected {len(plan.clips)} clip(s); "
                f"omitted {dropped_count} narration line(s) that did not fit."
                if dropped_count
                else f"Timeline builder selected {len(plan.clips)} clip(s)."
            ),
        )


def _read_music_plan(path: Path) -> MusicPlan | None:
    if not path.is_file():
        return None
    try:
        music_plan = MusicPlan.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as error:
        raise MontageError("Could not read music_plan.json for the timeline.") from error
    if (
        music_plan.mode != "none"
        and music_plan.source_path is not None
        and not music_plan.source_path.is_file()
    ):
        raise MontageError(
            f"Music plan references a missing soundtrack file: {music_plan.source_path}"
        )
    if music_plan.mode != "none" and music_plan.source_path is None:
        raise MontageError("Music plan is missing a soundtrack file path.")
    return music_plan


def _read_cached_timeline_artifacts(
    timeline_path: Path,
    decisions_path: Path,
) -> QuickMontagePlan | None:
    try:
        plan = QuickMontagePlan.model_validate_json(timeline_path.read_text(encoding="utf-8"))
        SceneSelectionReport.model_validate_json(decisions_path.read_text(encoding="utf-8"))
    except (OSError, ValidationError):
        return None
    try:
        if plan.narration_cues and (
            plan.narration_path is None
            or not plan.narration_path.is_file()
            or plan.narration_path.stat().st_size <= 0
        ):
            return None
    except OSError:
        return None
    return plan


def _read_narration_audio(
    context: ProjectContext,
    settings: QuickMontageSettings,
) -> VoiceSynthesisReport | None:
    if not settings.narration_enabled:
        return None
    report_path = context.artifacts_dir / "voice_synthesis.json"
    try:
        report = VoiceSynthesisReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as error:
        raise MontageError(
            "Narration audio was requested, but Voice Synthesis has no valid result."
        ) from error
    line_audio_dir = (context.artifacts_dir / "narration_lines").resolve()
    if report.line_count != len(report.lines) or not report.lines:
        raise MontageError("Voice Synthesis report has no complete narration line set.")
    try:
        for line in report.lines:
            resolved = line.audio_path.resolve()
            if (
                not resolved.is_relative_to(line_audio_dir)
                or not resolved.is_file()
                or resolved.stat().st_size <= 0
            ):
                raise MontageError(
                    "Voice Synthesis report references missing or unexpected narration line audio."
                )
    except OSError as error:
        raise MontageError("Could not inspect synthesized narration line audio.") from error
    return report


def _schedule_narration_cues(
    lines: list[SynthesizedNarrationLine],
    film_duration_seconds: float,
) -> list[NarrationAudioCue]:
    if film_duration_seconds <= 0:
        return []
    role_priority = {"opening": 4, "finale": 4, "highlight": 3, "journey": 2}
    for cue_count in range(len(lines), 0, -1):
        slot_duration = film_duration_seconds / cue_count
        padding = min(1.0, slot_duration * 0.1)
        available = slot_duration - 2 * padding
        eligible = [line for line in lines if line.duration_seconds <= available + 1e-6]
        if len(eligible) < cue_count:
            continue
        selected = sorted(
            sorted(
                eligible,
                key=lambda line: (-role_priority[line.section_role], line.line_index),
            )[:cue_count],
            key=lambda line: line.line_index,
        )
        return [
            NarrationAudioCue(
                line_index=index,
                section_role=line.section_role,
                audio_path=line.audio_path,
                cue_start_seconds=index * slot_duration + padding,
                cue_end_seconds=(index * slot_duration + padding + line.duration_seconds),
                duration_seconds=line.duration_seconds,
            )
            for index, line in enumerate(selected)
        ]
    return []


def _dropped_narration_line_count(
    report: VoiceSynthesisReport | None,
    plan: QuickMontagePlan,
) -> int:
    if report is None:
        return 0
    return max(0, report.line_count - len(plan.narration_cues))


def _cached_narration_matches(
    plan: QuickMontagePlan,
    report: VoiceSynthesisReport | None,
    narration_artifact: Path,
) -> bool:
    if report is None:
        return plan.narration_path is None and not plan.narration_cues
    expected_cues = _schedule_narration_cues(report.lines, plan.total_duration_seconds)
    if plan.narration_cues != expected_cues:
        return False
    if not expected_cues:
        return plan.narration_path is None and not plan.settings.narration_enabled
    return (
        plan.settings.narration_enabled
        and plan.narration_path is not None
        and plan.narration_path.resolve() == narration_artifact.resolve()
        and narration_track_matches(plan.narration_path, expected_cues)
    )


def _validate_narration_timeline(
    cues: list[NarrationAudioCue],
    film_duration_seconds: float,
) -> None:
    previous_end = 0.0
    for index, cue in enumerate(cues):
        if cue.line_index != index or cue.cue_start_seconds < previous_end - 0.01:
            raise MontageError("Timed narration cues are out of order or overlap.")
        if cue.cue_end_seconds > film_duration_seconds + 0.05:
            raise MontageError(
                "Timed narration does not fit the rendered movie duration. "
                "Shorten narration or increase the target duration."
            )
        previous_end = cue.cue_end_seconds


def _file_revision(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    try:
        stat = path.stat()
    except OSError as error:
        raise MontageError("Could not inspect an audio source for the timeline.") from error
    return {
        "path": path,
        "size": stat.st_size,
        "modified_ns": stat.st_mtime_ns,
    }


def _semantic_montage_settings(context: ProjectContext) -> QuickMontageSettings:
    if context.montage_settings is None:
        return QuickMontageSettings(semantic_analysis=True, story_style=context.style)
    return context.montage_settings.model_copy(
        update={"semantic_analysis": True, "story_style": context.style}
    )
