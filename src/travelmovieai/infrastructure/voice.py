"""Local voice-synthesis providers with lazy optional dependencies."""

import os
import subprocess
import time
import wave
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from travelmovieai.core.exceptions import DependencyUnavailableError, PipelineStageError
from travelmovieai.core.security import sanitize_process_error
from travelmovieai.infrastructure.processes import (
    release_process_resources,
    start_process,
    terminate_process_tree,
)


@dataclass(frozen=True, slots=True)
class SynthesizedVoice:
    output_path: Path
    duration_seconds: float
    sample_rate: int
    channels: int
    provider: str
    model: str


class PiperVoiceProvider:
    """Invoke a local Piper executable without importing model-heavy packages."""

    provider = "piper"

    def __init__(
        self,
        *,
        executable: str,
        model_path: Path,
        timeout_seconds: float = 600,
    ) -> None:
        self.executable = executable
        self.model_path = model_path.expanduser().resolve()
        self.timeout_seconds = timeout_seconds

    @property
    def model(self) -> str:
        return self.model_path.name

    def synthesize(
        self,
        text: str,
        output_path: Path,
        *,
        heartbeat: Callable[[], bool] | None = None,
    ) -> SynthesizedVoice:
        normalized_text = " ".join(text.split())
        if not normalized_text:
            raise PipelineStageError("Voice synthesis needs non-empty narration text.")
        if not self.model_path.is_file():
            raise DependencyUnavailableError(
                "The configured local Piper voice model is unavailable."
            )

        resolved_output = output_path.expanduser().resolve()
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        temporary_output = resolved_output.with_name(
            f".{resolved_output.stem}.{uuid4().hex}.tmp.wav"
        )
        command = [
            self.executable,
            "--model",
            str(self.model_path),
            "--output_file",
            str(temporary_output),
        ]
        process: subprocess.Popen[str] | None = None
        try:
            process = start_process(
                command,
                popen_factory=subprocess.Popen,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding="utf-8",
                errors="replace",
                text=True,
            )
        except FileNotFoundError as error:
            temporary_output.unlink(missing_ok=True)
            raise DependencyUnavailableError(
                f"Piper executable was not found: {self.executable}"
            ) from error
        except OSError as error:
            temporary_output.unlink(missing_ok=True)
            raise DependencyUnavailableError("The Piper process could not be started.") from error

        stderr = ""
        process_input: str | None = normalized_text
        deadline = time.monotonic() + self.timeout_seconds
        try:
            while True:
                if heartbeat is not None and heartbeat():
                    raise PipelineStageError(
                        "Piper voice synthesis was cancelled; the process tree was stopped."
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise PipelineStageError(
                        f"Piper voice synthesis timed out after {self.timeout_seconds:g}s; "
                        "the process tree was stopped."
                    )
                try:
                    _, stderr = process.communicate(
                        input=process_input,
                        timeout=min(0.2, remaining),
                    )
                    break
                except subprocess.TimeoutExpired:
                    process_input = None
            if process.returncode != 0:
                detail = sanitize_process_error(
                    stderr,
                    private_paths=[self.model_path, temporary_output],
                    fallback="unknown local Piper error",
                )
                raise PipelineStageError(f"Piper voice synthesis failed: {detail}")
            duration, sample_rate, channels = _probe_wave(temporary_output)
            os.replace(temporary_output, resolved_output)
        except BaseException:
            terminate_process_tree(process)
            raise
        finally:
            release_process_resources(process)
            temporary_output.unlink(missing_ok=True)

        return SynthesizedVoice(
            output_path=resolved_output,
            duration_seconds=duration,
            sample_rate=sample_rate,
            channels=channels,
            provider=self.provider,
            model=self.model,
        )


def _probe_wave(path: Path) -> tuple[float, int, int]:
    try:
        with wave.open(str(path), "rb") as audio:
            frame_count = audio.getnframes()
            sample_rate = audio.getframerate()
            channels = audio.getnchannels()
    except (OSError, EOFError, wave.Error) as error:
        raise PipelineStageError("Piper produced an invalid WAV file.") from error
    if frame_count <= 0 or sample_rate <= 0 or channels <= 0:
        raise PipelineStageError("Piper produced an empty WAV file.")
    return frame_count / sample_rate, sample_rate, channels
