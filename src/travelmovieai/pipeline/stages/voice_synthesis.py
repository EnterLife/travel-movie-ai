"""Explicit optional local Piper voice-synthesis stage."""

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from travelmovieai.application.context import ProjectContext
from travelmovieai.core.exceptions import PipelineStageError
from travelmovieai.domain.enums import PipelineStage, StageStatus
from travelmovieai.domain.models import NarrationReport, StageResult, VoiceSynthesisReport
from travelmovieai.infrastructure.artifacts import (
    artifact_fingerprint,
    stage_cache_manifest_matches,
    write_json_atomic,
    write_stage_cache_manifest,
)
from travelmovieai.infrastructure.voice import PiperVoiceProvider, SynthesizedVoice
from travelmovieai.pipeline.base import Stage

ARTIFACT_SCHEMA_VERSION = "voice-synthesis-v1"
VoiceProviderFactory = Callable[[ProjectContext], PiperVoiceProvider]


class VoiceSynthesisStage(Stage):
    name = PipelineStage.VOICE_SYNTHESIS

    def __init__(self, provider_factory: VoiceProviderFactory | None = None) -> None:
        self._provider_factory = provider_factory

    def run(self, context: ProjectContext) -> StageResult:
        narration_path = context.artifacts_dir / "narration.json"
        audio_path = context.artifacts_dir / "narration.wav"
        report_path = context.artifacts_dir / "voice_synthesis.json"
        cache_path = context.artifacts_dir / "voice_synthesis.cache.json"
        owned_paths = (audio_path, report_path, cache_path)
        narration_requested = (
            context.montage_settings is not None and context.montage_settings.narration_enabled
        )
        if not narration_requested:
            _remove_owned_artifacts(*owned_paths)
            return StageResult(
                stage=self.name,
                status=StageStatus.DISABLED,
                message="Voice synthesis was not requested by montage settings.",
            )
        if context.settings.voice_provider == "disabled":
            _remove_owned_artifacts(*owned_paths)
            raise PipelineStageError(
                "Narration audio was requested, but voice_provider is disabled. "
                "Configure a local Piper voice or disable narration."
            )
        if not narration_path.is_file():
            _remove_owned_artifacts(*owned_paths)
            return StageResult(
                stage=self.name,
                status=StageStatus.NO_INPUT,
                message="Voice synthesis needs narration.json.",
            )
        narration = _read_narration(narration_path)
        if not narration.lines:
            _remove_owned_artifacts(*owned_paths)
            return StageResult(
                stage=self.name,
                status=StageStatus.NO_INPUT,
                message="Voice synthesis needs at least one narration line.",
            )
        if context.settings.piper_model is None:
            raise PipelineStageError(
                "Piper voice synthesis is enabled, but piper_model is not configured."
            )

        model_path = context.settings.piper_model.expanduser().resolve()
        input_fingerprint = artifact_fingerprint(narration)
        config_fingerprint = artifact_fingerprint(
            {
                "schema": ARTIFACT_SCHEMA_VERSION,
                "provider": context.settings.voice_provider,
                "binary": context.settings.piper_binary,
                "model": model_path,
                "model_revision": _file_revision(model_path),
            }
        )
        if stage_cache_manifest_matches(
            cache_path,
            stage=self.name,
            artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
            input_fingerprint=input_fingerprint,
            config_fingerprint=config_fingerprint,
            artifacts=[audio_path, report_path],
        ) and _cached_report_valid(report_path, audio_path, len(narration.lines)):
            return StageResult(
                stage=self.name,
                status=StageStatus.CACHED,
                artifacts=[audio_path, report_path, cache_path],
                message="Voice synthesis reused cached local narration audio.",
            )

        provider = self._provider(context)
        if context.progress is not None:
            context.progress(0, 1, "Piper: synthesizing narration")
        synthesized = provider.synthesize(
            "\n\n".join(line.text for line in narration.lines),
            audio_path,
        )
        if context.progress is not None:
            context.progress(1, 1, "Piper: narration complete")
        report = _voice_report(synthesized, len(narration.lines))
        write_json_atomic(report_path, report)
        write_stage_cache_manifest(
            cache_path,
            stage=self.name,
            artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
            input_fingerprint=input_fingerprint,
            config_fingerprint=config_fingerprint,
            artifacts=[audio_path, report_path],
        )
        return StageResult(
            stage=self.name,
            artifacts=[audio_path, report_path, cache_path],
            message=(
                f"Voice synthesis produced {report.duration_seconds:.1f}s of local narration."
            ),
        )

    def _provider(self, context: ProjectContext) -> PiperVoiceProvider:
        if self._provider_factory is not None:
            return self._provider_factory(context)
        assert context.settings.piper_model is not None
        return PiperVoiceProvider(
            executable=context.settings.piper_binary,
            model_path=context.settings.piper_model,
            timeout_seconds=context.settings.voice_synthesis_timeout_seconds,
        )


def _read_narration(path: Path) -> NarrationReport:
    try:
        return NarrationReport.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as error:
        raise PipelineStageError("Could not read narration.json for voice synthesis.") from error


def _voice_report(result: SynthesizedVoice, line_count: int) -> VoiceSynthesisReport:
    return VoiceSynthesisReport(
        created_at=datetime.now(UTC),
        provider=result.provider,
        model=result.model,
        audio_path=result.output_path,
        duration_seconds=result.duration_seconds,
        sample_rate=result.sample_rate,
        channels=result.channels,
        line_count=line_count,
    )


def _cached_report_valid(path: Path, audio_path: Path, line_count: int) -> bool:
    try:
        report = VoiceSynthesisReport.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError):
        return False
    return (
        report.audio_path.resolve() == audio_path.resolve()
        and report.line_count == line_count
        and audio_path.is_file()
        and audio_path.stat().st_size > 0
    )


def _file_revision(path: Path) -> dict[str, int | str]:
    try:
        stat = path.stat()
    except OSError:
        return {"path": str(path), "size": -1, "modified_ns": -1}
    return {
        "path": str(path),
        "size": stat.st_size,
        "modified_ns": stat.st_mtime_ns,
    }


def _remove_owned_artifacts(*paths: Path) -> None:
    for path in paths:
        path.unlink(missing_ok=True)
