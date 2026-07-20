"""Explicit optional local Piper voice-synthesis stage."""

import wave
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from travelmovieai.application.context import ProjectContext
from travelmovieai.core.exceptions import PipelineStageError
from travelmovieai.domain.enums import PipelineStage, StageStatus
from travelmovieai.domain.models import (
    NarrationReport,
    StageExecutionMetadata,
    StageResult,
    SynthesizedNarrationLine,
    VoiceSynthesisReport,
)
from travelmovieai.infrastructure.artifacts import (
    artifact_fingerprint,
    stage_cache_manifest_matches,
    write_json_atomic,
    write_stage_cache_manifest,
)
from travelmovieai.infrastructure.voice import PiperVoiceProvider
from travelmovieai.pipeline.base import Stage

ARTIFACT_SCHEMA_VERSION = "voice-synthesis-v3-untimed-lines"
VoiceProviderFactory = Callable[[ProjectContext], PiperVoiceProvider]


class VoiceSynthesisStage(Stage):
    name = PipelineStage.VOICE_SYNTHESIS

    def __init__(self, provider_factory: VoiceProviderFactory | None = None) -> None:
        self._provider_factory = provider_factory

    def run(self, context: ProjectContext) -> StageResult:
        narration_path = context.artifacts_dir / "narration.json"
        audio_path = context.artifacts_dir / "narration.wav"
        line_audio_dir = context.artifacts_dir / "narration_lines"
        report_path = context.artifacts_dir / "voice_synthesis.json"
        cache_path = context.artifacts_dir / "voice_synthesis.cache.json"
        narration_requested = (
            context.montage_settings is not None and context.montage_settings.narration_enabled
        )
        if not narration_requested:
            _remove_owned_artifacts(audio_path, report_path, cache_path)
            _remove_line_audio(line_audio_dir)
            return StageResult(
                stage=self.name,
                status=StageStatus.DISABLED,
                message="Voice synthesis was not requested by montage settings.",
            )
        if context.settings.voice_provider == "disabled":
            _remove_owned_artifacts(audio_path, report_path, cache_path)
            _remove_line_audio(line_audio_dir)
            raise PipelineStageError(
                "Narration audio was requested, but voice_provider is disabled. "
                "Configure a local Piper voice or disable narration."
            )
        if not narration_path.is_file():
            _remove_owned_artifacts(audio_path, report_path, cache_path)
            _remove_line_audio(line_audio_dir)
            return StageResult(
                stage=self.name,
                status=StageStatus.NO_INPUT,
                message="Voice synthesis needs narration.json.",
            )
        narration = _read_narration(narration_path)
        if not narration.lines:
            _remove_owned_artifacts(audio_path, report_path, cache_path)
            _remove_line_audio(line_audio_dir)
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
        cached_report = _read_voice_report(report_path)
        cached_artifacts = [report_path]
        if cached_report is not None:
            cached_artifacts.extend(line.audio_path for line in cached_report.lines)
        if (
            stage_cache_manifest_matches(
                cache_path,
                stage=self.name,
                artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
                input_fingerprint=input_fingerprint,
                config_fingerprint=config_fingerprint,
                artifacts=cached_artifacts,
            )
            and cached_report is not None
            and _cached_report_valid(
                cached_report,
                line_audio_dir,
                narration,
            )
        ):
            return StageResult(
                stage=self.name,
                status=StageStatus.CACHED,
                artifacts=[*cached_artifacts, cache_path],
                message="Voice synthesis reused cached local narration audio.",
                execution=StageExecutionMetadata(
                    provider=cached_report.provider,
                    model=cached_report.model,
                ),
            )

        provider = self._provider(context)
        line_audio_dir.mkdir(parents=True, exist_ok=True)
        audio_path.unlink(missing_ok=True)
        generated_paths: list[Path] = []
        synthesized_lines: list[SynthesizedNarrationLine] = []
        if context.progress is not None:
            context.progress(0, len(narration.lines), "Piper: synthesizing timed narration")
        try:
            for index, line in enumerate(narration.lines):
                line_key = artifact_fingerprint(line, config_fingerprint)[:16]
                line_path = line_audio_dir / f"line-{index + 1:03d}-{line_key}.wav"
                synthesized = provider.synthesize(
                    line.text,
                    line_path,
                    heartbeat=(
                        _piper_heartbeat(
                            context.progress,
                            index=index,
                            total=len(narration.lines),
                        )
                        if context.progress is not None
                        else None
                    ),
                )
                generated_paths.append(synthesized.output_path)
                cue_end = line.cue_start_seconds + synthesized.duration_seconds
                if cue_end > line.cue_end_seconds + 0.05:
                    raise PipelineStageError(
                        f"Narration line {index + 1} is {synthesized.duration_seconds:.1f}s long "
                        "and does not fit its storyboard cue window. Shorten the narration text "
                        "or increase the target movie duration."
                    )
                synthesized_lines.append(
                    SynthesizedNarrationLine(
                        line_index=index,
                        section_role=line.section_role,
                        audio_path=synthesized.output_path.resolve(),
                        duration_seconds=synthesized.duration_seconds,
                        sample_rate=synthesized.sample_rate,
                        channels=synthesized.channels,
                    )
                )
                if context.progress is not None:
                    context.progress(
                        index + 1,
                        len(narration.lines),
                        f"Piper: narration line {index + 1}/{len(narration.lines)} complete",
                    )
        except BaseException:
            _remove_owned_artifacts(*generated_paths, audio_path)
            raise
        report = VoiceSynthesisReport(
            created_at=datetime.now(UTC),
            provider=provider.provider,
            model=provider.model,
            line_count=len(synthesized_lines),
            lines=synthesized_lines,
        )
        write_json_atomic(report_path, report)
        _remove_stale_line_audio(
            line_audio_dir,
            {line.audio_path.resolve() for line in synthesized_lines},
        )
        write_stage_cache_manifest(
            cache_path,
            stage=self.name,
            artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
            input_fingerprint=input_fingerprint,
            config_fingerprint=config_fingerprint,
            artifacts=[report_path, *(line.audio_path for line in synthesized_lines)],
        )
        return StageResult(
            stage=self.name,
            artifacts=[
                report_path,
                *(line.audio_path for line in synthesized_lines),
                cache_path,
            ],
            message=(
                "Voice synthesis produced "
                f"{sum(line.duration_seconds for line in report.lines):.1f}s "
                "of local narration lines."
            ),
            execution=StageExecutionMetadata(
                provider=report.provider,
                model=report.model,
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


def _piper_heartbeat(
    progress: Callable[[int, int, str], None],
    *,
    index: int,
    total: int,
) -> Callable[[], bool]:
    def heartbeat() -> bool:
        progress(
            index,
            total,
            f"Piper: narration line {index + 1}/{total} is still running",
        )
        return False

    return heartbeat


def _read_voice_report(path: Path) -> VoiceSynthesisReport | None:
    try:
        return VoiceSynthesisReport.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError):
        return None


def _cached_report_valid(
    report: VoiceSynthesisReport,
    line_audio_dir: Path,
    narration: NarrationReport,
) -> bool:
    try:
        return (
            report.line_count == len(narration.lines)
            and len(report.lines) == len(narration.lines)
            and all(
                synthesized.line_index == index
                and synthesized.section_role == line.section_role
                and synthesized.duration_seconds
                <= line.cue_end_seconds - line.cue_start_seconds + 0.05
                and synthesized.audio_path.resolve().is_relative_to(line_audio_dir.resolve())
                and _valid_cached_wave(synthesized)
                for index, (synthesized, line) in enumerate(
                    zip(report.lines, narration.lines, strict=True)
                )
            )
        )
    except OSError:
        return False


def _valid_cached_wave(line: SynthesizedNarrationLine) -> bool:
    try:
        if not line.audio_path.is_file() or line.audio_path.stat().st_size <= 0:
            return False
        with wave.open(str(line.audio_path), "rb") as audio:
            frame_rate = audio.getframerate()
            frame_count = audio.getnframes()
            return (
                frame_rate == line.sample_rate
                and audio.getnchannels() == line.channels
                and frame_count > 0
                and abs(frame_count / frame_rate - line.duration_seconds) <= 0.05
            )
    except (EOFError, OSError, wave.Error):
        return False


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


def _remove_line_audio(path: Path) -> None:
    if not path.is_dir():
        return
    for audio_path in path.glob("line-*.wav"):
        audio_path.unlink(missing_ok=True)


def _remove_stale_line_audio(path: Path, active_paths: set[Path]) -> None:
    for audio_path in path.glob("line-*.wav"):
        if audio_path.resolve() not in active_paths:
            audio_path.unlink(missing_ok=True)
