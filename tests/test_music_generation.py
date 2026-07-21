import hashlib
import json
import math
import sys
import wave
from array import array
from datetime import UTC, datetime
from pathlib import Path

import pytest

from travelmovieai.core.exceptions import MusicGenerationError
from travelmovieai.domain.models import QuickMontagePlan, QuickMontageSettings
from travelmovieai.infrastructure.music_generation import (
    ACE_STEP_REQUIRED_CONFIGS,
    AceStepMusicGenerator,
    resolve_local_music_model,
)
from travelmovieai.story.music import (
    MusicPlanExecution,
    _score_music_candidate,
    build_music_plan,
)


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
        normalized_prompt = prompt.casefold()
        assert "no vocals" in normalized_prompt
        assert "lyrics" in normalized_prompt
        assert "2020s" in normalized_prompt
        assert "macro arrangement" in normalized_prompt
        assert "avoid mechanical looping" in normalized_prompt
        assert "clipping" in normalized_prompt
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


class CandidateMusicGenerator:
    name = "ace-step"
    model = "ACE-Step/acestep-v15-turbo"

    def __init__(self) -> None:
        self.seeds: list[int] = []

    def generate_candidates(
        self,
        output_dir: Path,
        *,
        prompt: str,
        cue_sheet: object,
        duration_seconds: float,
        bpm: int,
        keyscale: str,
        seeds: list[int],
        progress: object = None,
    ) -> list[tuple[Path, int]]:
        del prompt, cue_sheet, bpm, keyscale, progress
        self.seeds = seeds
        output_dir.mkdir(parents=True, exist_ok=True)
        generated: list[tuple[Path, int]] = []
        sample_rate = 8000
        frame_count = round(duration_seconds * sample_rate)
        for index, seed in enumerate(seeds):
            path = output_dir / f"candidate-{index + 1:02d}.wav"
            samples = array("h")
            for frame in range(frame_count):
                if index == 0:
                    left = right = 0
                else:
                    seconds = frame / sample_rate
                    envelope = 0.55 + 0.25 * math.sin(math.tau * seconds / max(duration_seconds, 1))
                    frequency = 180 + index * 37 + 12 * math.sin(math.tau * seconds / 1.7)
                    left = round(12000 * envelope * math.sin(math.tau * frequency * seconds))
                    right = round(
                        11500 * envelope * math.sin(math.tau * frequency * seconds + 0.35)
                    )
                samples.extend((left, right))
            with wave.open(str(path), "wb") as audio:
                audio.setnchannels(2)
                audio.setsampwidth(2)
                audio.setframerate(sample_rate)
                audio.writeframes(samples.tobytes())
            generated.append((path, seed))
        return generated


def _complete_ace_cache(path: Path) -> None:
    for relative_path in (*ACE_STEP_REQUIRED_CONFIGS, "acestep-v15-turbo/config.json"):
        config = path / relative_path
        config.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            '{"model_type": "qwen3"}' if "Qwen3-Embedding" in relative_path else '{"valid": true}'
        )
        config.write_text(payload, encoding="utf-8")


def test_music_model_auto_resolution() -> None:
    assert resolve_local_music_model("auto", gpu_memory_mb=6144) == "ACE-Step/acestep-v15-turbo"
    assert (
        resolve_local_music_model("auto", gpu_memory_mb=24 * 1024, quality="balanced")
        == "ACE-Step/acestep-v15-xl-turbo"
    )
    assert (
        resolve_local_music_model("auto", gpu_memory_mb=12 * 1024, quality="studio")
        == "ACE-Step/acestep-v15-sft"
    )
    assert (
        resolve_local_music_model("auto", gpu_memory_mb=24 * 1024, quality="studio")
        == "ACE-Step/acestep-v15-xl-sft"
    )
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


def test_ace_step_studio_request_uses_slow_sampler_lm_reference_and_lora(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.wav"
    lora = tmp_path / "adapter.safetensors"
    generator = AceStepMusicGenerator(
        "ACE-Step/acestep-v15-sft",
        runtime_dir=tmp_path / "runtime",
        model_cache=tmp_path / "models",
        ffmpeg_binary="ffmpeg",
        allow_download=True,
        device="auto",
        gpu_memory_mb=12 * 1024,
        quality="studio",
        reference_audio=reference,
        reference_strength=0.35,
        lora_path=lora,
        lora_strength=0.8,
    )

    request = generator._worker_request(
        prompt="Modern instrumental",
        duration_seconds=120,
        bpm=92,
        keyscale="D Dorian",
        seeds=[11, 22],
        output_dir=tmp_path / "output",
    )

    assert request["inference_steps"] == 64
    assert request["guidance_scale"] == 7.5
    assert request["use_adg"] is True
    assert request["thinking"] is True
    assert request["lm_model"] == "acestep-5Hz-lm-1.7B"
    assert request["reference_audio"] == str(reference)
    assert request["reference_strength"] == 0.35
    assert request["lora_path"] == str(lora)
    assert request["lora_strength"] == 0.8


def test_ace_step_rejects_duration_beyond_native_model_limit(tmp_path: Path) -> None:
    generator = AceStepMusicGenerator(
        "ACE-Step/acestep-v15-turbo",
        runtime_dir=tmp_path / "runtime",
        model_cache=tmp_path / "models",
        ffmpeg_binary="ffmpeg",
        allow_download=False,
        device="cpu",
        gpu_memory_mb=None,
    )

    with pytest.raises(MusicGenerationError, match="up to 600 seconds"):
        generator.generate_candidates(
            tmp_path / "output",
            prompt="Instrumental",
            cue_sheet=[],
            duration_seconds=601,
            bpm=80,
            keyscale="A Minor",
            seeds=[42],
        )


def test_ace_step_generation_requests_native_full_duration_and_normalizes_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = tmp_path / "runtime"
    executable = runtime / ".venv" / "Scripts" / "python.exe"
    executable.parent.mkdir(parents=True)
    executable.touch()
    (runtime / "cli.py").write_text("# fake cli\n", encoding="utf-8")
    model_cache = tmp_path / "models"
    _complete_ace_cache(model_cache)
    generator = AceStepMusicGenerator(
        "ACE-Step/acestep-v15-turbo",
        runtime_dir=runtime,
        model_cache=model_cache,
        ffmpeg_binary="ffmpeg",
        allow_download=True,
        device="auto",
        gpu_memory_mb=6144,
    )
    captured_request: dict[str, object] = {}
    normalized_duration = 0.0

    def run_model(
        command: list[str],
        *,
        cwd: Path,
        environment: dict[str, str],
        progress: object,
    ) -> list[str]:
        nonlocal captured_request
        request_path = Path(command[command.index("--request") + 1])
        captured_request = json.loads(request_path.read_text(encoding="utf-8"))
        model_output = Path(str(captured_request["output_dir"]))
        source = model_output / "base.wav"
        source.write_bytes(b"fake wav")
        return [
            "done",
            "TRAVELMOVIEAI_RESULT="
            + json.dumps({"candidates": [{"path": str(source), "seed": 42}]}),
        ]

    def normalize(source_path: Path, output_path: Path, duration_seconds: float) -> None:
        nonlocal normalized_duration
        assert source_path.name == "base.wav"
        normalized_duration = duration_seconds
        output_path.write_bytes(b"normalized")

    monkeypatch.setattr(generator, "_run_streaming", run_model)
    monkeypatch.setattr(generator, "_normalize", normalize)

    output = tmp_path / "soundtrack.wav"
    generator.generate(
        output,
        prompt="Instrumental lounge, no vocals",
        cue_sheet=[],
        duration_seconds=120,
        bpm=76,
        seed=42,
    )

    assert captured_request["duration_seconds"] == 120
    assert captured_request["seeds"] == [42]
    assert normalized_duration == 120
    assert output.read_bytes() == b"normalized"


def test_ace_step_repairs_invalid_model_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = tmp_path / "runtime"
    executable = runtime / ".venv" / "Scripts" / "python.exe"
    executable.parent.mkdir(parents=True)
    executable.touch()
    model_cache = tmp_path / "models"
    _complete_ace_cache(model_cache)
    missing = model_cache / "Qwen3-Embedding-0.6B" / "config.json"
    missing.write_text("{}", encoding="utf-8")
    generator = AceStepMusicGenerator(
        "ACE-Step/acestep-v15-turbo",
        runtime_dir=runtime,
        model_cache=model_cache,
        ffmpeg_binary="ffmpeg",
        allow_download=True,
        device="auto",
        gpu_memory_mb=6144,
    )
    commands: list[list[str]] = []

    def repair(
        command: list[str],
        *,
        cwd: Path,
        environment: dict[str, str],
        progress: object,
    ) -> list[str]:
        commands.append(command)
        relative_path = command[-2]
        repaired = model_cache / relative_path
        repaired.parent.mkdir(parents=True, exist_ok=True)
        repaired.write_text('{"model_type": "qwen3"}', encoding="utf-8")
        return []

    monkeypatch.setattr(generator, "_run_streaming", repair)

    generator._ensure_model_configs(None)

    assert missing.is_file()
    assert commands[0][-3:-1] == ["ACE-Step/Ace-Step1.5", "Qwen3-Embedding-0.6B/config.json"]


def test_ace_step_rejects_incomplete_offline_model_cache(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    executable = runtime / ".venv" / "Scripts" / "python.exe"
    executable.parent.mkdir(parents=True)
    executable.touch()
    generator = AceStepMusicGenerator(
        "ACE-Step/acestep-v15-turbo",
        runtime_dir=runtime,
        model_cache=tmp_path / "models",
        ffmpeg_binary="ffmpeg",
        allow_download=False,
        device="auto",
        gpu_memory_mb=6144,
    )

    with pytest.raises(MusicGenerationError, match="cache is incomplete.*downloads are disabled"):
        generator._ensure_model_configs(None)


def test_ace_step_normalization_pads_without_looping_and_writes_48khz_pcm(
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

    class FinishedProcess:
        returncode: int | None = None

        def __init__(self, command: list[str]) -> None:
            self.command = command

        def poll(self) -> int:
            Path(self.command[-1]).write_bytes(b"normalized")
            self.returncode = 0
            return 0

        def communicate(self, timeout: float) -> tuple[str, str]:
            assert timeout == 5
            return "", ""

    def start_ffmpeg(command: list[str], **kwargs: object) -> FinishedProcess:
        del kwargs
        commands.append(command)
        return FinishedProcess(command)

    monkeypatch.setattr(
        "travelmovieai.infrastructure.music_generation.subprocess.Popen",
        start_ffmpeg,
    )

    generator._normalize(source, output, 90)

    assert "-stream_loop" not in commands[0]
    assert commands[0][commands[0].index("-af") + 1] == "apad"
    assert commands[0][commands[0].index("-ar") + 1] == "48000"
    assert commands[0][commands[0].index("-c:a") + 1] == "pcm_s24le"
    assert output.read_bytes() == b"normalized"


def test_ace_step_normalization_reports_ffmpeg_timeout(
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
        ffmpeg_timeout_seconds=0.25,
    )
    source = tmp_path / "short.wav"
    source.touch()
    output = tmp_path / "full.wav"
    terminated: list[object] = []

    class HangingProcess:
        returncode: int | None = None

        def poll(self) -> None:
            return None

    hanging = HangingProcess()

    monkeypatch.setattr(
        "travelmovieai.infrastructure.music_generation.subprocess.Popen",
        lambda *args, **kwargs: hanging,
    )
    monkeypatch.setattr(
        "travelmovieai.infrastructure.music_generation.terminate_process_tree",
        lambda process: terminated.append(process),
    )

    with pytest.raises(MusicGenerationError, match="timed out.*0.25s"):
        generator._normalize(source, output, 120)

    assert terminated == [hanging]
    assert not output.exists()


def test_ace_step_normalization_honors_cancellation_and_stops_tree(
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
        cancel_requested=lambda: True,
    )
    source = tmp_path / "short.wav"
    source.touch()
    output = tmp_path / "full.wav"
    terminated: list[object] = []

    class HangingProcess:
        returncode: int | None = None

        def poll(self) -> None:
            return None

    hanging = HangingProcess()
    monkeypatch.setattr(
        "travelmovieai.infrastructure.music_generation.subprocess.Popen",
        lambda *args, **kwargs: hanging,
    )
    monkeypatch.setattr(
        "travelmovieai.infrastructure.music_generation.terminate_process_tree",
        lambda process: terminated.append(process),
    )

    with pytest.raises(MusicGenerationError, match="normalization was cancelled"):
        generator._normalize(source, output, 120)

    assert terminated == [hanging]
    assert not output.exists()


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
    assert commands[0][4:] == ["--music-ai-only", str(runtime), "--non-interactive"]


def test_ace_step_streaming_process_times_out_and_stops_tree(tmp_path: Path) -> None:
    generator = AceStepMusicGenerator(
        "ACE-Step/acestep-v15-turbo",
        runtime_dir=tmp_path,
        model_cache=tmp_path / "models",
        ffmpeg_binary="ffmpeg",
        allow_download=False,
        device="cpu",
        gpu_memory_mb=None,
        process_timeout_seconds=0.2,
    )

    with pytest.raises(MusicGenerationError, match="timed out after 0.2s.*process tree"):
        generator._run_streaming(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            cwd=tmp_path,
            environment={},
            progress=None,
        )


def test_ace_step_streaming_process_honors_cancellation(tmp_path: Path) -> None:
    generator = AceStepMusicGenerator(
        "ACE-Step/acestep-v15-turbo",
        runtime_dir=tmp_path,
        model_cache=tmp_path / "models",
        ffmpeg_binary="ffmpeg",
        allow_download=False,
        device="cpu",
        gpu_memory_mb=None,
        cancel_requested=lambda: True,
    )

    with pytest.raises(MusicGenerationError, match="was cancelled"):
        generator._run_streaming(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            cwd=tmp_path,
            environment={},
            progress=None,
        )


def test_frozen_ace_step_runtime_reports_missing_packaged_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = AceStepMusicGenerator(
        "ACE-Step/acestep-v15-turbo",
        runtime_dir=tmp_path / "missing-runtime",
        model_cache=tmp_path / "models",
        ffmpeg_binary="ffmpeg",
        allow_download=True,
        device="cpu",
        gpu_memory_mb=None,
    )
    monkeypatch.setattr(
        "travelmovieai.infrastructure.music_generation._find_setup_script",
        lambda: None,
    )
    monkeypatch.setattr(sys, "frozen", True, raising=False)

    with pytest.raises(MusicGenerationError, match="packaged application.*does not include"):
        generator._ensure_runtime(None)


def test_build_music_plan_uses_local_model(tmp_path: Path) -> None:
    settings = QuickMontageSettings(
        target_duration_seconds=5,
        music_engine="ace-step",
        music_profile="lounge",
        music_quality="draft",
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


def test_balanced_music_generates_four_candidates_and_selects_audited_winner(
    tmp_path: Path,
) -> None:
    settings = QuickMontageSettings(
        target_duration_seconds=5,
        music_engine="ace-step",
        music_profile="lounge",
        music_quality="balanced",
    )
    montage = QuickMontagePlan(
        created_at=datetime.now(UTC),
        settings=settings,
        total_duration_seconds=5,
    )
    generator = CandidateMusicGenerator()

    plan = build_music_plan(
        [],
        [],
        settings,
        tmp_path / "library",
        tmp_path / "soundtrack.wav",
        montage,
        neural_generator=generator,
    )

    assert len(generator.seeds) == 4
    assert len(set(generator.seeds)) == 4
    assert len(plan.candidates) == 4
    assert sum(candidate.selected for candidate in plan.candidates) == 1
    assert plan.selected_candidate_index != 0
    assert plan.candidates[0].notes
    assert plan.source_path is not None
    selected = next(candidate for candidate in plan.candidates if candidate.selected)
    assert plan.source_path.read_bytes() == selected.source_path.read_bytes()


def test_music_candidate_scoring_rejects_long_silent_tail(tmp_path: Path) -> None:
    def write_tone(path: Path, *, silent_tail_seconds: float) -> None:
        sample_rate = 8000
        duration_seconds = 5
        active_frames = round((duration_seconds - silent_tail_seconds) * sample_rate)
        samples = array("h")
        for frame in range(duration_seconds * sample_rate):
            if frame >= active_frames:
                left = right = 0
            else:
                seconds = frame / sample_rate
                left = round(10000 * math.sin(math.tau * 220 * seconds))
                right = round(10000 * math.sin(math.tau * 220 * seconds + 0.4))
            samples.extend((left, right))
        with wave.open(str(path), "wb") as audio:
            audio.setnchannels(2)
            audio.setsampwidth(2)
            audio.setframerate(sample_rate)
            audio.writeframes(samples.tobytes())

    clean_path = tmp_path / "clean.wav"
    silent_path = tmp_path / "silent-tail.wav"
    write_tone(clean_path, silent_tail_seconds=0.5)
    write_tone(silent_path, silent_tail_seconds=3)

    clean = _score_music_candidate(
        clean_path,
        index=0,
        seed=1,
        expected_duration=5,
        profile="cinematic",
    )
    silent = _score_music_candidate(
        silent_path,
        index=1,
        seed=2,
        expected_duration=5,
        profile="cinematic",
    )

    assert clean.total_score > silent.total_score
    assert any(note.startswith("silent tail") for note in silent.notes)


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
        music_quality="draft",
    )
    montage = QuickMontagePlan(
        created_at=datetime.now(UTC),
        settings=settings,
        total_duration_seconds=5,
    )
    generator = FakeMusicGenerator()
    output = tmp_path / "soundtrack.wav"
    first_execution = MusicPlanExecution()
    first = build_music_plan(
        [],
        [],
        settings,
        tmp_path / "library",
        output,
        montage,
        neural_generator=generator,
        execution=first_execution,
    )
    (tmp_path / "music_plan.json").write_text(
        first.model_dump_json(),
        encoding="utf-8",
    )

    second_execution = MusicPlanExecution()
    second = build_music_plan(
        [],
        [],
        settings,
        tmp_path / "library",
        output,
        montage,
        neural_generator=generator,
        execution=second_execution,
    )

    assert generator.calls == 1
    assert first_execution.cache_hit is False
    assert second_execution.cache_hit is True
    assert second.cache_key == first.cache_key
    assert "cache" in second.reasoning

    output.write_bytes(b"replaced soundtrack content")
    third_execution = MusicPlanExecution()
    third = build_music_plan(
        [],
        [],
        settings,
        tmp_path / "library",
        output,
        montage,
        neural_generator=generator,
        execution=third_execution,
    )

    assert generator.calls == 2
    assert third_execution.cache_hit is False
    assert third.source_content_sha256 == hashlib.sha256(output.read_bytes()).hexdigest()


def test_generated_music_cache_invalidates_when_reference_audio_changes(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"reference-v1")
    settings = QuickMontageSettings(
        target_duration_seconds=5,
        music_engine="ace-step",
        music_profile="lounge",
        music_quality="draft",
        music_reference_path=reference,
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
    (tmp_path / "music_plan.json").write_text(first.model_dump_json(), encoding="utf-8")

    cached_execution = MusicPlanExecution()
    build_music_plan(
        [],
        [],
        settings,
        tmp_path / "library",
        output,
        montage,
        neural_generator=generator,
        execution=cached_execution,
    )
    reference.write_bytes(b"reference-v2")
    invalidated_execution = MusicPlanExecution()
    build_music_plan(
        [],
        [],
        settings,
        tmp_path / "library",
        output,
        montage,
        neural_generator=generator,
        execution=invalidated_execution,
    )

    assert cached_execution.cache_hit is True
    assert invalidated_execution.cache_hit is False
    assert generator.calls == 2


def test_procedural_fallback_is_not_reused_as_success(tmp_path: Path) -> None:
    settings = QuickMontageSettings(
        target_duration_seconds=5,
        music_engine="auto",
        music_profile="lounge",
    )
    montage = QuickMontagePlan(
        created_at=datetime.now(UTC),
        settings=settings,
        total_duration_seconds=5,
    )
    output = tmp_path / "fallback.wav"
    first_execution = MusicPlanExecution()
    first = build_music_plan(
        [],
        [],
        settings,
        tmp_path / "library",
        output,
        montage,
        execution=first_execution,
    )
    (tmp_path / "music_plan.json").write_text(first.model_dump_json(), encoding="utf-8")

    second_execution = MusicPlanExecution()
    second = build_music_plan(
        [],
        [],
        settings,
        tmp_path / "library",
        output,
        montage,
        execution=second_execution,
    )

    assert first.generator == "procedural"
    assert first.fallback_used is True
    assert first_execution.cache_hit is False
    assert second.generator == "procedural"
    assert second.fallback_used is True
    assert second_execution.cache_hit is False
    assert "reused from cache" not in second.reasoning.casefold()
