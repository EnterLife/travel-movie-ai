import wave
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest

from travelmovieai.application.context import ProjectContext
from travelmovieai.core.config import Settings
from travelmovieai.core.exceptions import MontageError, PipelineStageError
from travelmovieai.domain.enums import MediaType, StageStatus
from travelmovieai.domain.models import (
    MediaAsset,
    NarrationLine,
    NarrationReport,
    QuickMontagePlan,
    QuickMontageSettings,
    Scene,
    SynthesizedNarrationLine,
    VoiceSynthesisReport,
)
from travelmovieai.infrastructure.artifacts import write_json_atomic
from travelmovieai.infrastructure.database import MediaAssetRepository
from travelmovieai.infrastructure.voice import PiperVoiceProvider, SynthesizedVoice
from travelmovieai.pipeline.stages.timeline_builder import TimelineBuilderStage
from travelmovieai.pipeline.stages.voice_synthesis import VoiceSynthesisStage


def test_narration_line_rejects_invalid_cue_window() -> None:
    with pytest.raises(ValueError, match="cue_end_seconds must be greater"):
        NarrationLine(
            section_role="opening",
            text="Journey",
            cue_start_seconds=2,
            cue_end_seconds=1,
        )


def test_voice_stage_disabled_removes_owned_stale_artifacts(tmp_path: Path) -> None:
    context = _context(tmp_path, Settings(voice_provider="disabled"))
    owned = [
        context.artifacts_dir / "narration.wav",
        context.artifacts_dir / "voice_synthesis.json",
        context.artifacts_dir / "voice_synthesis.cache.json",
    ]
    for path in owned:
        path.write_bytes(b"stale")

    result = VoiceSynthesisStage().run(context)

    assert result.status is StageStatus.DISABLED
    assert not any(path.exists() for path in owned)


def test_voice_stage_generates_and_reuses_typed_piper_artifact(tmp_path: Path) -> None:
    model_path = tmp_path / "voice.onnx"
    model_path.write_bytes(b"model")
    context = _context(
        tmp_path,
        Settings(voice_provider="piper", piper_model=model_path),
        montage_settings=QuickMontageSettings(narration_enabled=True),
    )
    write_json_atomic(
        context.artifacts_dir / "narration.json",
        NarrationReport(
            created_at=datetime.now(UTC),
            lines=[
                NarrationLine(
                    section_role="opening",
                    text="Our journey begins.",
                    cue_start_seconds=1,
                    cue_end_seconds=30,
                ),
                NarrationLine(
                    section_role="finale",
                    text="The journey ends.",
                    cue_start_seconds=45,
                    cue_end_seconds=75,
                ),
            ],
        ),
    )

    class FakePiper(PiperVoiceProvider):
        calls = 0
        texts: list[str] = []

        def synthesize(
            self,
            text: str,
            output_path: Path,
            *,
            heartbeat: Callable[[], bool] | None = None,
        ) -> SynthesizedVoice:
            del heartbeat
            self.calls += 1
            self.texts.append(text)
            with wave.open(str(output_path), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(16_000)
                audio.writeframes(b"\0\0" * 8_000)
            return SynthesizedVoice(
                output_path=output_path.resolve(),
                duration_seconds=0.5,
                sample_rate=16_000,
                channels=1,
                provider="piper",
                model="voice.onnx",
            )

    provider = FakePiper(executable="piper", model_path=model_path)
    stage = VoiceSynthesisStage(provider_factory=lambda _: provider)

    first = stage.run(context)
    second = stage.run(context)
    report = VoiceSynthesisReport.model_validate_json(
        (context.artifacts_dir / "voice_synthesis.json").read_text(encoding="utf-8")
    )
    report.lines[0].audio_path.write_bytes(b"corrupt cached wav")
    rebuilt = stage.run(context)

    assert first.status is StageStatus.COMPLETED
    assert second.status is StageStatus.CACHED
    assert first.cache_hit is False
    assert first.skipped is False
    assert second.cache_hit is True
    assert second.skipped is True
    assert first.execution.provider == "piper"
    assert first.execution.model == "voice.onnx"
    assert first.execution.fallback_count == 0
    assert second.execution.provider == "piper"
    assert second.execution.model == "voice.onnx"
    assert second.execution.fallback_count == 0
    assert rebuilt.status is StageStatus.COMPLETED
    assert rebuilt.cache_hit is False
    assert provider.calls == 4
    assert provider.texts == [
        "Our journey begins.",
        "The journey ends.",
        "Our journey begins.",
        "The journey ends.",
    ]
    assert report.line_count == 2
    assert [line.duration_seconds for line in report.lines] == [0.5, 0.5]
    assert not (context.artifacts_dir / "narration.wav").exists()


def test_voice_stage_requires_configured_local_model(tmp_path: Path) -> None:
    context = _context(
        tmp_path,
        Settings(voice_provider="piper", piper_model=None),
        montage_settings=QuickMontageSettings(narration_enabled=True),
    )
    write_json_atomic(
        context.artifacts_dir / "narration.json",
        NarrationReport(
            created_at=datetime.now(UTC),
            lines=[
                NarrationLine(
                    section_role="opening",
                    text="Journey",
                    cue_start_seconds=1,
                    cue_end_seconds=20,
                )
            ],
        ),
    )

    with pytest.raises(PipelineStageError, match="piper_model"):
        VoiceSynthesisStage().run(context)


def test_voice_stage_rejects_line_that_does_not_fit_storyboard_cue(tmp_path: Path) -> None:
    model_path = tmp_path / "voice.onnx"
    model_path.write_bytes(b"model")
    context = _context(
        tmp_path,
        Settings(voice_provider="piper", piper_model=model_path),
        montage_settings=QuickMontageSettings(narration_enabled=True),
    )
    write_json_atomic(
        context.artifacts_dir / "narration.json",
        NarrationReport(
            created_at=datetime.now(UTC),
            lines=[
                NarrationLine(
                    section_role="opening",
                    text="A line that is too long.",
                    cue_start_seconds=0,
                    cue_end_seconds=0.2,
                )
            ],
        ),
    )

    class SlowFakePiper(PiperVoiceProvider):
        def synthesize(
            self,
            text: str,
            output_path: Path,
            *,
            heartbeat: Callable[[], bool] | None = None,
        ) -> SynthesizedVoice:
            del text, heartbeat
            with wave.open(str(output_path), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(16_000)
                audio.writeframes(b"\0\0" * 8_000)
            return SynthesizedVoice(
                output_path=output_path.resolve(),
                duration_seconds=0.5,
                sample_rate=16_000,
                channels=1,
                provider="piper",
                model="voice.onnx",
            )

    stage = VoiceSynthesisStage(
        provider_factory=lambda _: SlowFakePiper(executable="piper", model_path=model_path)
    )

    with pytest.raises(PipelineStageError, match="does not fit its storyboard cue window"):
        stage.run(context)

    assert not (context.artifacts_dir / "narration.wav").exists()
    assert not (context.artifacts_dir / "voice_synthesis.json").exists()


def test_voice_stage_forwards_active_line_heartbeat_for_immediate_cancellation(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "voice.onnx"
    model_path.write_bytes(b"model")
    progress_events: list[tuple[int, int, str]] = []

    def cancel_progress(current: int, total: int, message: str) -> None:
        progress_events.append((current, total, message))
        if "is still running" in message:
            raise MontageError("web full stop")

    context = ProjectContext(
        input_path=tmp_path,
        workspace=tmp_path / "workspace",
        settings=Settings(voice_provider="piper", piper_model=model_path),
        montage_settings=QuickMontageSettings(narration_enabled=True),
        progress=cancel_progress,
    )
    context.prepare()
    write_json_atomic(
        context.artifacts_dir / "narration.json",
        NarrationReport(
            created_at=datetime.now(UTC),
            lines=[
                NarrationLine(
                    section_role="opening",
                    text="Journey",
                    cue_start_seconds=0,
                    cue_end_seconds=20,
                )
            ],
        ),
    )

    class HeartbeatPiper(PiperVoiceProvider):
        def synthesize(
            self,
            text: str,
            output_path: Path,
            *,
            heartbeat: Callable[[], bool] | None = None,
        ) -> SynthesizedVoice:
            del text, output_path
            assert heartbeat is not None
            heartbeat()
            raise AssertionError("cancellation callback must interrupt synthesis")

    stage = VoiceSynthesisStage(
        provider_factory=lambda _: HeartbeatPiper(executable="piper", model_path=model_path)
    )

    with pytest.raises(MontageError, match="web full stop"):
        stage.run(context)

    assert progress_events[-1] == (0, 1, "Piper: narration line 1/1 is still running")
    assert not (context.artifacts_dir / "voice_synthesis.json").exists()
    assert not list((context.artifacts_dir / "narration_lines").glob("*.wav"))


def test_voice_stage_not_requested_is_disabled_even_when_piper_is_configured(
    tmp_path: Path,
) -> None:
    context = _context(
        tmp_path,
        Settings(voice_provider="piper", piper_model=tmp_path / "voice.onnx"),
    )

    result = VoiceSynthesisStage().run(context)

    assert result.status is StageStatus.DISABLED


def test_voice_stage_rejects_requested_narration_when_provider_is_disabled(
    tmp_path: Path,
) -> None:
    context = _context(
        tmp_path,
        Settings(voice_provider="disabled"),
        montage_settings=QuickMontageSettings(narration_enabled=True),
    )

    with pytest.raises(PipelineStageError, match="voice_provider is disabled"):
        VoiceSynthesisStage().run(context)


def test_timeline_includes_valid_requested_narration_audio(tmp_path: Path) -> None:
    input_path = tmp_path / "input"
    input_path.mkdir()
    context = ProjectContext(
        input_path=input_path,
        workspace=tmp_path / "workspace",
        settings=Settings(),
        montage_settings=QuickMontageSettings(
            narration_enabled=True,
            music_enabled=False,
            music_mode="none",
        ),
    )
    context.prepare()
    repository = MediaAssetRepository(context.database_path)
    repository.initialize()
    asset = MediaAsset(
        id=UUID(int=1),
        path=input_path / "clip.mp4",
        relative_path=Path("clip.mp4"),
        media_type=MediaType.VIDEO,
        extension=".mp4",
        size_bytes=1,
        modified_at=datetime.now(UTC),
        modified_ns=1,
        duration_seconds=8,
    )
    scene = Scene(
        id=UUID(int=2),
        asset_id=asset.id,
        start_seconds=0,
        end_seconds=8,
        importance_score=90,
    )
    repository.synchronize([asset], datetime.now(UTC))
    repository.synchronize_scenes([scene])
    cue_audio = context.artifacts_dir / "narration_lines" / "line-001.wav"
    cue_audio.parent.mkdir()
    _write_wave(cue_audio, duration_seconds=2)
    write_json_atomic(
        context.artifacts_dir / "voice_synthesis.json",
        VoiceSynthesisReport(
            created_at=datetime.now(UTC),
            provider="piper",
            model="voice.onnx",
            line_count=1,
            lines=[
                SynthesizedNarrationLine(
                    line_index=0,
                    section_role="opening",
                    audio_path=cue_audio,
                    duration_seconds=2,
                    sample_rate=16_000,
                    channels=1,
                )
            ],
        ),
    )

    stage = TimelineBuilderStage()
    first = stage.run(context)
    cached = stage.run(context)
    cue_audio.write_bytes(cue_audio.read_bytes() + b"\0")
    rebuilt = stage.run(context)
    plan = QuickMontagePlan.model_validate_json(
        (context.artifacts_dir / "quick_timeline.json").read_text(encoding="utf-8")
    )

    assert first.status is StageStatus.COMPLETED
    assert cached.status is StageStatus.CACHED
    assert rebuilt.status is StageStatus.COMPLETED
    assert plan.narration_path == (context.artifacts_dir / "narration.wav").resolve()
    assert plan.narration_path.is_file()
    assert len(plan.narration_cues) == 1
    assert plan.narration_cues[0].cue_end_seconds <= plan.total_duration_seconds

    disabled_context = ProjectContext(
        input_path=input_path,
        workspace=context.workspace,
        settings=context.settings,
        montage_settings=context.montage_settings.model_copy(update={"narration_enabled": False}),
    )
    disabled = stage.run(disabled_context)
    disabled_plan = QuickMontagePlan.model_validate_json(
        (context.artifacts_dir / "quick_timeline.json").read_text(encoding="utf-8")
    )

    assert disabled.status is StageStatus.COMPLETED
    assert disabled_plan.narration_path is None
    assert disabled_plan.narration_cues == []
    assert not (context.artifacts_dir / "narration.wav").exists()


def test_timeline_rejects_requested_narration_outside_owned_artifact(tmp_path: Path) -> None:
    context = _context(
        tmp_path,
        Settings(),
        montage_settings=QuickMontageSettings(narration_enabled=True),
    )
    repository = MediaAssetRepository(context.database_path)
    repository.initialize()
    asset = MediaAsset(
        id=UUID(int=3),
        path=tmp_path / "clip.mp4",
        relative_path=Path("clip.mp4"),
        media_type=MediaType.VIDEO,
        extension=".mp4",
        size_bytes=1,
        modified_at=datetime.now(UTC),
        modified_ns=1,
        duration_seconds=5,
    )
    repository.synchronize([asset], datetime.now(UTC))
    repository.synchronize_scenes(
        [
            Scene(
                id=UUID(int=4),
                asset_id=asset.id,
                start_seconds=0,
                end_seconds=5,
            )
        ]
    )
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"audio")
    write_json_atomic(
        context.artifacts_dir / "voice_synthesis.json",
        VoiceSynthesisReport(
            created_at=datetime.now(UTC),
            provider="piper",
            model="voice.onnx",
            line_count=1,
            lines=[
                SynthesizedNarrationLine(
                    line_index=0,
                    section_role="opening",
                    audio_path=outside,
                    duration_seconds=1,
                    sample_rate=16_000,
                    channels=1,
                )
            ],
        ),
    )

    with pytest.raises(MontageError, match="missing or unexpected"):
        TimelineBuilderStage().run(context)


def test_timeline_drops_narration_that_does_not_fit_actual_duration_and_caches_degraded(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input"
    input_path.mkdir()
    context = ProjectContext(
        input_path=input_path,
        workspace=tmp_path / "workspace",
        settings=Settings(),
        montage_settings=QuickMontageSettings(
            narration_enabled=True,
            music_enabled=False,
            music_mode="none",
            target_duration_seconds=90,
            transition="none",
        ),
    )
    context.prepare()
    repository = MediaAssetRepository(context.database_path)
    repository.initialize()
    asset = MediaAsset(
        id=UUID(int=10),
        path=input_path / "short.mp4",
        relative_path=Path("short.mp4"),
        media_type=MediaType.VIDEO,
        extension=".mp4",
        size_bytes=1,
        modified_at=datetime.now(UTC),
        modified_ns=1,
        duration_seconds=2,
    )
    repository.synchronize([asset], datetime.now(UTC))
    repository.synchronize_scenes(
        [
            Scene(
                id=UUID(int=11),
                asset_id=asset.id,
                start_seconds=0,
                end_seconds=2,
                importance_score=90,
            )
        ]
    )
    line_dir = context.artifacts_dir / "narration_lines"
    line_dir.mkdir()
    line_paths = [line_dir / "line-001.wav", line_dir / "line-002.wav"]
    for line_path in line_paths:
        _write_wave(line_path, duration_seconds=2)
    (context.artifacts_dir / "narration.wav").write_bytes(b"stale")
    write_json_atomic(
        context.artifacts_dir / "voice_synthesis.json",
        VoiceSynthesisReport(
            created_at=datetime.now(UTC),
            provider="piper",
            model="voice.onnx",
            line_count=2,
            lines=[
                SynthesizedNarrationLine(
                    line_index=index,
                    section_role=role,
                    audio_path=line_path,
                    duration_seconds=2,
                    sample_rate=16_000,
                    channels=1,
                )
                for index, (role, line_path) in enumerate(
                    zip(("opening", "finale"), line_paths, strict=True)
                )
            ],
        ),
    )

    first = TimelineBuilderStage().run(context)
    second = TimelineBuilderStage().run(context)
    plan = QuickMontagePlan.model_validate_json(
        (context.artifacts_dir / "quick_timeline.json").read_text(encoding="utf-8")
    )

    assert first.status is StageStatus.DEGRADED
    assert first.cache_hit is False
    assert second.status is StageStatus.DEGRADED
    assert second.cache_hit is True
    assert plan.narration_cues == []
    assert plan.narration_path is None
    assert plan.total_duration_seconds < context.montage_settings.target_duration_seconds
    assert plan.settings.narration_enabled is False
    assert not (context.artifacts_dir / "narration.wav").exists()


def _write_wave(path: Path, *, duration_seconds: float) -> None:
    sample_rate = 16_000
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(b"\0\0" * round(sample_rate * duration_seconds))


def _context(
    tmp_path: Path,
    settings: Settings,
    *,
    montage_settings: QuickMontageSettings | None = None,
) -> ProjectContext:
    context = ProjectContext(
        input_path=tmp_path,
        workspace=tmp_path / "workspace",
        settings=settings,
        montage_settings=montage_settings,
    )
    context.prepare()
    return context
