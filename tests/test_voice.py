import subprocess
import wave
from pathlib import Path

import pytest

from travelmovieai.core.exceptions import DependencyUnavailableError, PipelineStageError
from travelmovieai.infrastructure.voice import PiperVoiceProvider


def _request_cancel() -> bool:
    return True


def _raise_web_cancel() -> bool:
    raise PipelineStageError("web full stop")


class _SuccessfulProcess:
    returncode: int | None = None

    def __init__(self, command: list[str], **_: object) -> None:
        self.command = command
        self.inputs: list[str | None] = []

    def communicate(
        self,
        input: str | None = None,
        timeout: float | None = None,
    ) -> tuple[str, str]:
        del timeout
        self.inputs.append(input)
        temporary_output = Path(self.command[self.command.index("--output_file") + 1])
        with wave.open(str(temporary_output), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(16_000)
            audio.writeframes(b"\x00\x00" * 8_000)
        self.returncode = 0
        return "", ""


def test_piper_provider_writes_valid_voice_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "voice.onnx"
    model_path.write_bytes(b"model")
    output_path = tmp_path / "narration.wav"
    processes: list[_SuccessfulProcess] = []

    def fake_popen(command: list[str], **kwargs: object) -> _SuccessfulProcess:
        process = _SuccessfulProcess(command, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr("travelmovieai.infrastructure.voice.subprocess.Popen", fake_popen)
    result = PiperVoiceProvider(
        executable="piper",
        model_path=model_path,
    ).synthesize("  A local   journey.  ", output_path)

    assert result.output_path == output_path.resolve()
    assert result.duration_seconds == 0.5
    assert result.sample_rate == 16_000
    assert result.channels == 1
    assert result.provider == "piper"
    assert processes[0].inputs == ["A local journey."]
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

    class FailedProcess:
        returncode: int | None = None

        def __init__(self, command: list[str], **_: object) -> None:
            self.command = command

        def communicate(
            self,
            input: str | None = None,
            timeout: float | None = None,
        ) -> tuple[str, str]:
            del input, timeout
            self.returncode = 1
            temporary_output = self.command[self.command.index("--output_file") + 1]
            return "", f"cannot read {model_path.resolve()} or write {temporary_output}"

        def poll(self) -> int:
            return 1

    monkeypatch.setattr("travelmovieai.infrastructure.voice.subprocess.Popen", FailedProcess)
    provider = PiperVoiceProvider(executable="piper", model_path=model_path)

    with pytest.raises(PipelineStageError) as raised:
        provider.synthesize("Narration", tmp_path / "narration.wav")

    message = str(raised.value)
    assert str(model_path.resolve()) not in message
    assert "<local-path>" in message


def test_piper_provider_polls_heartbeat_before_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "voice.onnx"
    model_path.write_bytes(b"model")
    output_path = tmp_path / "narration.wav"
    heartbeat_calls = 0

    class PollingProcess(_SuccessfulProcess):
        calls = 0

        def communicate(
            self,
            input: str | None = None,
            timeout: float | None = None,
        ) -> tuple[str, str]:
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired(self.command, timeout or 0)
            return super().communicate(input=input, timeout=timeout)

    def heartbeat() -> bool:
        nonlocal heartbeat_calls
        heartbeat_calls += 1
        return False

    monkeypatch.setattr("travelmovieai.infrastructure.voice.subprocess.Popen", PollingProcess)

    result = PiperVoiceProvider(executable="piper", model_path=model_path).synthesize(
        "Journey",
        output_path,
        heartbeat=heartbeat,
    )

    assert result.duration_seconds == pytest.approx(0.5)
    assert heartbeat_calls == 2
    assert output_path.is_file()


@pytest.mark.parametrize("failure_mode", ["cancel", "cancel_exception", "timeout"])
def test_piper_provider_stops_tree_and_removes_partial_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
) -> None:
    model_path = tmp_path / "voice.onnx"
    model_path.write_bytes(b"model")
    output_path = tmp_path / "narration.wav"
    stopped: list[object] = []

    class HungProcess:
        returncode: int | None = None

        def __init__(self, command: list[str], **_: object) -> None:
            temporary_output = Path(command[command.index("--output_file") + 1])
            temporary_output.write_bytes(b"partial")

        def communicate(self, **_: object) -> tuple[str, str]:
            raise AssertionError("cancellation or deadline must be checked before polling")

    monkeypatch.setattr("travelmovieai.infrastructure.voice.subprocess.Popen", HungProcess)
    monkeypatch.setattr(
        "travelmovieai.infrastructure.voice.terminate_process_tree",
        lambda process: stopped.append(process),
    )
    if failure_mode == "timeout":
        clock = iter((0.0, 1.0))
        monkeypatch.setattr(
            "travelmovieai.infrastructure.voice.time.monotonic",
            lambda: next(clock),
        )
        heartbeat = None
        expected = "timed out after 0.1s.*process tree was stopped"
    elif failure_mode == "cancel":
        heartbeat = _request_cancel
        expected = "cancelled.*process tree was stopped"
    else:
        heartbeat = _raise_web_cancel
        expected = "web full stop"

    with pytest.raises(PipelineStageError, match=expected):
        PiperVoiceProvider(
            executable="piper",
            model_path=model_path,
            timeout_seconds=0.1,
        ).synthesize("Journey", output_path, heartbeat=heartbeat)

    assert len(stopped) == 1
    assert not output_path.exists()
    assert not list(tmp_path.glob(".*.tmp.wav"))
