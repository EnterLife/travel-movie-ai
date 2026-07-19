import subprocess
import wave
from pathlib import Path

import pytest

from travelmovieai.core.exceptions import DependencyUnavailableError, PipelineStageError
from travelmovieai.infrastructure.voice import PiperVoiceProvider


def test_piper_provider_writes_valid_voice_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "voice.onnx"
    model_path.write_bytes(b"model")
    output_path = tmp_path / "narration.wav"
    captured: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["command"] = command
        captured["input"] = kwargs["input"]
        temporary_output = Path(command[command.index("--output_file") + 1])
        with wave.open(str(temporary_output), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(16_000)
            audio.writeframes(b"\x00\x00" * 8_000)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = PiperVoiceProvider(
        executable="piper",
        model_path=model_path,
    ).synthesize("  A local   journey.  ", output_path)

    assert result.output_path == output_path.resolve()
    assert result.duration_seconds == 0.5
    assert result.sample_rate == 16_000
    assert result.channels == 1
    assert result.provider == "piper"
    assert captured["input"] == "A local journey."
    assert output_path.is_file()
    assert not list(tmp_path.glob(".*.tmp.wav"))


def test_piper_provider_rejects_missing_model_without_starting_process(tmp_path: Path) -> None:
    provider = PiperVoiceProvider(
        executable="piper",
        model_path=tmp_path / "missing.onnx",
    )

    with pytest.raises(DependencyUnavailableError, match="voice model is unavailable"):
        provider.synthesize("Narration", tmp_path / "narration.wav")


def test_piper_provider_rejects_empty_narration(tmp_path: Path) -> None:
    model_path = tmp_path / "voice.onnx"
    model_path.write_bytes(b"model")
    provider = PiperVoiceProvider(executable="piper", model_path=model_path)

    with pytest.raises(PipelineStageError, match="non-empty narration"):
        provider.synthesize("  \n ", tmp_path / "narration.wav")


def test_piper_provider_redacts_private_paths_from_process_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "private voice.onnx"
    model_path.write_bytes(b"model")

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        temporary_output = command[command.index("--output_file") + 1]
        return subprocess.CompletedProcess(
            command,
            1,
            "",
            f"cannot read {model_path.resolve()} or write {temporary_output}",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    provider = PiperVoiceProvider(executable="piper", model_path=model_path)

    with pytest.raises(PipelineStageError) as raised:
        provider.synthesize("Narration", tmp_path / "narration.wav")

    message = str(raised.value)
    assert str(model_path.resolve()) not in message
    assert "<local-path>" in message
