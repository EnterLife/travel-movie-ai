"""Pipeline stage that builds a declarative semantic montage timeline."""

from pathlib import Path

from pydantic import ValidationError

from travelmovieai.application.context import ProjectContext
from travelmovieai.core.exceptions import MontageError
from travelmovieai.domain.enums import PipelineStage, StageStatus
from travelmovieai.domain.models import (
    MusicPlan,
    QuickMontagePlan,
    QuickMontageSettings,
    SceneSelectionReport,
    StageResult,
    VoiceSynthesisReport,
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

ARTIFACT_SCHEMA_VERSION = "timeline-builder-v5-media-revisions"


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
        if not assets or not scenes:
            timeline_artifact.unlink(missing_ok=True)
            decisions_artifact.unlink(missing_ok=True)
            cache_artifact.unlink(missing_ok=True)
            return StageResult(
                stage=self.name,
                status=StageStatus.NO_INPUT,
                message="Timeline builder needs media assets and ranked scenes.",
            )

        settings = _semantic_montage_settings(context)
        music_artifact = context.artifacts_dir / "music_plan.json"
        music_plan = _read_music_plan(music_artifact)
        narration_path = _read_narration_audio(context, settings)
        input_fingerprint = artifact_fingerprint(
            assets,
            scenes,
            music_plan,
            _file_revision(music_plan.source_path if music_plan is not None else None),
            _file_revision(narration_path),
        )
        config_fingerprint = artifact_fingerprint(settings, ARTIFACT_SCHEMA_VERSION)
        if stage_cache_manifest_matches(
            cache_artifact,
            stage=self.name,
            artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
            input_fingerprint=input_fingerprint,
            config_fingerprint=config_fingerprint,
            artifacts=[timeline_artifact, decisions_artifact],
        ) and _cached_timeline_artifacts_valid(timeline_artifact, decisions_artifact):
            return StageResult(
                stage=self.name,
                status=StageStatus.CACHED,
                artifacts=[timeline_artifact, decisions_artifact, cache_artifact],
                message="Timeline builder reused cached timeline artifacts.",
            )

        plan = build_semantic_montage_plan(assets, scenes, settings, music_plan)
        if music_plan is not None:
            plan = apply_music_directing(plan, scenes)
        if narration_path is not None:
            plan = plan.model_copy(update={"narration_path": narration_path})
        write_json_atomic(timeline_artifact, plan)
        write_json_atomic(
            decisions_artifact,
            build_selection_report(scenes, plan, settings),
        )
        write_stage_cache_manifest(
            cache_artifact,
            stage=self.name,
            artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
            input_fingerprint=input_fingerprint,
            config_fingerprint=config_fingerprint,
            artifacts=[timeline_artifact, decisions_artifact],
        )
        repository.record_timeline_version(
            plan,
            phase="built",
            variant_name=context.variant_name,
            variant_slug=context.variant_slug,
        )
        return StageResult(
            stage=self.name,
            artifacts=[timeline_artifact, decisions_artifact, cache_artifact],
            message=f"Timeline builder selected {len(plan.clips)} clip(s).",
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


def _cached_timeline_artifacts_valid(timeline_path: Path, decisions_path: Path) -> bool:
    try:
        QuickMontagePlan.model_validate_json(timeline_path.read_text(encoding="utf-8"))
        SceneSelectionReport.model_validate_json(decisions_path.read_text(encoding="utf-8"))
    except (OSError, ValidationError):
        return False
    return True


def _read_narration_audio(
    context: ProjectContext,
    settings: QuickMontageSettings,
) -> Path | None:
    if not settings.narration_enabled:
        return None
    report_path = context.artifacts_dir / "voice_synthesis.json"
    try:
        report = VoiceSynthesisReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as error:
        raise MontageError(
            "Narration audio was requested, but Voice Synthesis has no valid result."
        ) from error
    expected_path = (context.artifacts_dir / "narration.wav").resolve()
    if report.audio_path.resolve() != expected_path or not expected_path.is_file():
        raise MontageError(
            "Voice Synthesis report references missing or unexpected narration audio."
        )
    return expected_path


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
