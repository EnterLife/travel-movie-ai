import wave
from datetime import UTC, datetime
from pathlib import Path

import pytest

from travelmovieai.core.exceptions import MusicGenerationError
from travelmovieai.domain.models import QuickMontagePlan, QuickMontageSettings
from travelmovieai.infrastructure.music_generation import (
    AceStepMusicGenerator,
    resolve_local_music_model,
)
from travelmovieai.story.music import build_music_plan


class FakeMusicGenerator:
    name = "ace-step"
    model = "ACE-Step/acestep-v15-turbo"

    def __init__(self) -> None:
        self.calls = 0

    def generate(
        self,
        output_path: Path,
        *,
        prompt: str,
        cue_sheet: object,
        duration_seconds: float,
        bpm: int,
        seed: int,
        progress: object = None,
    ) -> None:
        self.calls += 1
        assert "no vocals" in prompt
        assert "no lyrics" in prompt
        assert "low dynamic range" in prompt
        assert "mastered with headroom" in prompt
        assert "no clipping" in prompt
        assert cue_sheet
        assert bpm == 76
        assert seed >= 0
        with wave.open(str(output_path), "wb") as audio:
            audio.setnchannels(2)
            audio.setsampwidth(2)
            audio.setframerate(8000)
            audio.writeframes(b"\0\0\0\0" * round(duration_seconds * 8000))


class FailingMusicGenerator(FakeMusicGenerator):
    def generate(
        self,
        output_path: Path,
        *,
        prompt: str,
        cue_sheet: object,
        duration_seconds: float,
        bpm: int,
        seed: int,
        progress: object = None,
    ) -> None:
        raise MusicGenerationError("model unavailable")


def test_music_model_auto_resolution() -> None:
    assert resolve_local_music_model("auto", gpu_memory_mb=6144) == "ACE-Step/acestep-v15-turbo"
    assert resolve_local_music_model("custom/model", gpu_memory_mb=6144) == "custom/model"


def test_ace_step_configuration_enables_low_vram_offload(tmp_path: Path) -> None:
    generator = AceStepMusicGenerator(
        "ACE-Step/acestep-v15-turbo",
        runtime_dir=tmp_path / "runtime",
        model_cache=tmp_path / "models",
        ffmpeg_binary="ffmpeg",
        allow_download=True,
        device="auto",
        gpu_memory_mb=6144,
    )

    config = generator._configuration(
        prompt="Instrumental lounge, no vocals",
        duration_seconds=90,
        bpm=84,
        seed=42,
        output_dir=tmp_path / "output",
    )

    assert 'config_path = "acestep-v15-turbo"' in config
    assert "instrumental = true" in config
    assert "thinking = false" in config
    assert "inference_steps = 16" in config
    assert "guidance_scale = 1.35" in config
    assert "offload_to_cpu = true" in config
    assert "offload_dit_to_cpu = true" in config


def test_ace_step_normalization_loops_short_model_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = AceStepMusicGenerator(
        "ACE-Step/acestep-v15-turbo",
        runtime_dir=tmp_path / "runtime",
        model_cache=tmp_path / "models",
        ffmpeg_binary="ffmpeg",
        allow_download=True,
        device="auto",
        gpu_memory_mb=6144,
    )
    source = tmp_path / "short.wav"
    source.touch()
    output = tmp_path / "full.wav"
    commands: list[list[str]] = []

    def run_ffmpeg(
        command: list[str],
        **kwargs: object,
    ) -> object:
        commands.append(command)
        Path(command[-1]).write_bytes(b"normalized")
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(
        "travelmovieai.infrastructure.music_generation.subprocess.run",
        run_ffmpeg,
    )

    generator._normalize(source, output, 90)

    assert commands[0][5:7] == ["-stream_loop", "-1"]
    assert output.read_bytes() == b"normalized"


def test_ace_step_runtime_uses_unified_windows_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = tmp_path / "runtime"
    generator = AceStepMusicGenerator(
        "ACE-Step/acestep-v15-turbo",
        runtime_dir=runtime,
        model_cache=tmp_path / "models",
        ffmpeg_binary="ffmpeg",
        allow_download=True,
        device="auto",
        gpu_memory_mb=6144,
    )
    commands: list[list[str]] = []

    def run_setup(
        command: list[str],
        *,
        cwd: Path,
        environment: dict[str, str],
        progress: object,
    ) -> list[str]:
        commands.append(command)
        executable = runtime / ".venv" / "Scripts" / "python.exe"
        executable.parent.mkdir(parents=True)
        executable.touch()
        return []

    monkeypatch.setattr(generator, "_run_streaming", run_setup)

    generator._ensure_runtime(None)

    assert Path(commands[0][3]).name == "setup_windows.bat"
    assert commands[0][4:] == ["--music-ai-only", str(runtime)]


def test_build_music_plan_uses_local_model(tmp_path: Path) -> None:
    settings = QuickMontageSettings(
        target_duration_seconds=5,
        music_engine="ace-step",
        music_profile="lounge",
    )
    montage = QuickMontagePlan(
        created_at=datetime.now(UTC),
        settings=settings,
        total_duration_seconds=5,
    )

    plan = build_music_plan(
        [],
        [],
        settings,
        tmp_path / "library",
        tmp_path / "soundtrack.wav",
        montage,
        neural_generator=FakeMusicGenerator(),
    )

    assert plan.generator == "ace-step"
    assert plan.model == "ACE-Step/acestep-v15-turbo"
    assert plan.fallback_used is False
    assert plan.source_path is not None and plan.source_path.is_file()


def test_auto_music_falls_back_but_explicit_model_fails(tmp_path: Path) -> None:
    automatic = QuickMontageSettings(
        target_duration_seconds=5,
        music_engine="auto",
        music_profile="lounge",
    )
    montage = QuickMontagePlan(
        created_at=datetime.now(UTC),
        settings=automatic,
        total_duration_seconds=5,
    )
    fallback = build_music_plan(
        [],
        [],
        automatic,
        tmp_path / "library",
        tmp_path / "fallback.wav",
        montage,
        neural_generator=FailingMusicGenerator(),
    )

    assert fallback.generator == "procedural"
    assert fallback.fallback_used is True

    explicit = automatic.model_copy(update={"music_engine": "ace-step"})
    with pytest.raises(MusicGenerationError):
        build_music_plan(
            [],
            [],
            explicit,
            tmp_path / "library",
            tmp_path / "explicit.wav",
            montage.model_copy(update={"settings": explicit}),
            neural_generator=FailingMusicGenerator(),
        )


def test_generated_music_is_reused_when_timeline_and_model_match(
    tmp_path: Path,
) -> None:
    settings = QuickMontageSettings(
        target_duration_seconds=5,
        music_engine="ace-step",
        music_profile="lounge",
    )
    montage = QuickMontagePlan(
        created_at=datetime.now(UTC),
        settings=settings,
        total_duration_seconds=5,
    )
    generator = FakeMusicGenerator()
    output = tmp_path / "soundtrack.wav"
    first = build_music_plan(
        [],
        [],
        settings,
        tmp_path / "library",
        output,
        montage,
        neural_generator=generator,
    )
    (tmp_path / "music_plan.json").write_text(
        first.model_dump_json(),
        encoding="utf-8",
    )

    second = build_music_plan(
        [],
        [],
        settings,
        tmp_path / "library",
        output,
        montage,
        neural_generator=generator,
    )

    assert generator.calls == 1
    assert second.cache_key == first.cache_key
    assert "кэша" in second.reasoning
