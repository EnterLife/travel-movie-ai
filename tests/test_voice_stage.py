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
    VoiceSynthesisReport,
)
from travelmovieai.infrastructure.artifacts import write_json_atomic
from travelmovieai.infrastructure.database import MediaAssetRepository
from travelmovieai.infrastructure.voice import PiperVoiceProvider, SynthesizedVoice
from travelmovieai.pipeline.stages.timeline_builder import TimelineBuilderStage
from travelmovieai.pipeline.stages.voice_synthesis import VoiceSynthesisStage


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
                NarrationLine(section_role="opening", text="Our journey begins."),
                NarrationLine(section_role="finale", text="The journey ends."),
            ],
        ),
    )

    class FakePiper(PiperVoiceProvider):
        calls = 0

        def synthesize(self, text: str, output_path: Path) -> SynthesizedVoice:
            self.calls += 1
            assert text == "Our journey begins.\n\nThe journey ends."
            output_path.write_bytes(b"RIFF-fake-wave")
            return SynthesizedVoice(
                output_path=output_path.resolve(),
                duration_seconds=4.5,
                sample_rate=22_050,
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

    assert first.status is StageStatus.COMPLETED
    assert second.status is StageStatus.CACHED
    assert provider.calls == 1
    assert report.line_count == 2
    assert report.duration_seconds == 4.5


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
            lines=[NarrationLine(section_role="opening", text="Journey")],
        ),
    )

    with pytest.raises(PipelineStageError, match="piper_model"):
        VoiceSynthesisStage().run(context)


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
    narration_audio = context.artifacts_dir / "narration.wav"
    narration_audio.write_bytes(b"RIFF-audio")
    write_json_atomic(
        context.artifacts_dir / "voice_synthesis.json",
        VoiceSynthesisReport(
            created_at=datetime.now(UTC),
            provider="piper",
            model="voice.onnx",
            audio_path=narration_audio,
            duration_seconds=2,
            sample_rate=22_050,
            channels=1,
            line_count=1,
        ),
    )

    TimelineBuilderStage().run(context)
    plan = QuickMontagePlan.model_validate_json(
        (context.artifacts_dir / "quick_timeline.json").read_text(encoding="utf-8")
    )

    assert plan.narration_path == narration_audio.resolve()


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
            audio_path=outside,
            duration_seconds=1,
            sample_rate=16_000,
            channels=1,
            line_count=1,
        ),
    )

    with pytest.raises(MontageError, match="missing or unexpected"):
        TimelineBuilderStage().run(context)


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
