"""Adapters for specialized local music generation models."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from travelmovieai.core.exceptions import MusicGenerationError
from travelmovieai.domain.models import MusicCueSection

LOCAL_MUSIC_MODELS = ("ACE-Step/acestep-v15-turbo",)
ACE_STEP_REPOSITORY = "https://github.com/ACE-Step/ACE-Step-1.5.git"
ACE_STEP_MODEL_REPOSITORY = "ACE-Step/Ace-Step1.5"
ACE_STEP_MAX_GENERATION_SECONDS = 90.0
ACE_STEP_REQUIRED_CONFIGS = (
    "Qwen3-Embedding-0.6B/config.json",
    "vae/config.json",
)

MusicProgress = Callable[[int, int, str], None]


def resolve_local_music_model(
    configured_model: str | None,
    *,
    gpu_memory_mb: int | None,
) -> str:
    """Resolve the default model while preserving explicit selections."""

    if configured_model and configured_model != "auto":
        return configured_model
    return LOCAL_MUSIC_MODELS[0]


class AceStepMusicGenerator:
    """Run ACE-Step in an isolated local runtime and normalize its output."""

    name: Literal["ace-step"] = "ace-step"

    def __init__(
        self,
        model: str,
        *,
        runtime_dir: Path,
        model_cache: Path,
        ffmpeg_binary: str,
        allow_download: bool,
        device: str,
        gpu_memory_mb: int | None,
        ffmpeg_timeout_seconds: float = 7200,
    ) -> None:
        self.model = model
        self.runtime_dir = runtime_dir
        self.model_cache = model_cache
        self.ffmpeg_binary = ffmpeg_binary
        self.allow_download = allow_download
        self.device = device
        self.gpu_memory_mb = gpu_memory_mb
        self.ffmpeg_timeout_seconds = ffmpeg_timeout_seconds if ffmpeg_timeout_seconds > 0 else 7200

    def generate(
        self,
        output_path: Path,
        *,
        prompt: str,
        cue_sheet: list[MusicCueSection],
        duration_seconds: float,
        bpm: int,
        seed: int,
        progress: MusicProgress | None = None,
    ) -> None:
        self._ensure_runtime(progress)
        self._ensure_model_configs(progress)
        python_executable = self.runtime_dir / ".venv" / "Scripts" / "python.exe"
        cli_path = self.runtime_dir / "cli.py"
        if not python_executable.is_file() or not cli_path.is_file():
            raise MusicGenerationError("The isolated ACE-Step environment is incomplete.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".ace-step-",
            dir=output_path.parent,
        ) as temporary:
            temporary_dir = Path(temporary)
            model_output = temporary_dir / "output"
            model_output.mkdir()
            config_path = temporary_dir / "generation.toml"
            generation_duration = min(
                ACE_STEP_MAX_GENERATION_SECONDS,
                max(10.0, duration_seconds),
            )
            config_path.write_text(
                self._configuration(
                    prompt=_prompt_with_cue_sheet(prompt, cue_sheet),
                    duration_seconds=generation_duration,
                    bpm=bpm,
                    seed=seed,
                    output_dir=model_output,
                ),
                encoding="utf-8",
            )
            environment = os.environ.copy()
            environment.update(
                {
                    "ACESTEP_PROJECT_ROOT": str(self.runtime_dir),
                    "ACESTEP_CHECKPOINTS_DIR": str(self.model_cache),
                    "PYTHONUTF8": "1",
                }
            )
            if not self.allow_download:
                environment.update(
                    {
                        "HF_HUB_OFFLINE": "1",
                        "TRANSFORMERS_OFFLINE": "1",
                    }
                )
            if progress:
                progress(0, 4, f"ACE-Step: preparing {self.model}")
            lines = self._run_streaming(
                [
                    str(python_executable),
                    str(cli_path),
                    "--config",
                    str(config_path),
                    "--backend",
                    "pt",
                    "--log-level",
                    "INFO",
                ],
                cwd=self.runtime_dir,
                environment=environment,
                progress=progress,
            )
            candidates = sorted(
                model_output.rglob("*.wav"),
                key=lambda path: path.stat().st_mtime_ns,
                reverse=True,
            )
            if not candidates:
                detail = _ace_step_error_detail(lines)
                raise MusicGenerationError(f"ACE-Step did not create a WAV file. {detail}".strip())
            if progress:
                progress(3, 4, "ACE-Step: normalizing output")
            self._normalize(candidates[0], output_path, duration_seconds)
            if progress:
                progress(4, 4, "ACE-Step: music created")

    def _ensure_runtime(self, progress: MusicProgress | None) -> None:
        executable = self.runtime_dir / ".venv" / "Scripts" / "python.exe"
        if executable.is_file():
            return
        if not self.allow_download:
            raise MusicGenerationError("Auto-download is disabled and ACE-Step runtime is missing.")
        setup_script = Path(__file__).resolve().parents[3] / "scripts" / "setup_windows.bat"
        if not setup_script.is_file():
            raise MusicGenerationError("scripts\\setup_windows.bat was not found.")
        if progress:
            progress(0, 4, "ACE-Step: installing isolated runtime")
        command_processor = os.environ.get("COMSPEC", "cmd.exe")
        self._run_streaming(
            [
                command_processor,
                "/d",
                "/c",
                str(setup_script),
                "--music-ai-only",
                str(self.runtime_dir),
            ],
            cwd=setup_script.parent.parent,
            environment=os.environ.copy(),
            progress=progress,
        )
        if not executable.is_file():
            raise MusicGenerationError("Could not install ACE-Step runtime.")

    def _ensure_model_configs(self, progress: MusicProgress | None) -> None:
        required = [*ACE_STEP_REQUIRED_CONFIGS]
        if self.model.split("/")[-1] == "acestep-v15-turbo":
            required.append("acestep-v15-turbo/config.json")
        missing = [name for name in required if not _valid_ace_step_config(self.model_cache / name)]
        if not missing:
            return
        if not self.allow_download:
            joined = ", ".join(missing)
            raise MusicGenerationError(
                f"ACE-Step model cache is incomplete ({joined}) and downloads are disabled."
            )

        python_executable = self.runtime_dir / ".venv" / "Scripts" / "python.exe"
        if not python_executable.is_file():
            raise MusicGenerationError("The isolated ACE-Step Python runtime is missing.")
        if progress:
            progress(1, 4, "ACE-Step: repairing incomplete model metadata")
        environment = os.environ.copy()
        environment["PYTHONUTF8"] = "1"
        download_script = (
            "from huggingface_hub import hf_hub_download; import sys; "
            "hf_hub_download(repo_id=sys.argv[1], filename=sys.argv[2], local_dir=sys.argv[3])"
        )
        for relative_path in missing:
            try:
                self._run_streaming(
                    [
                        str(python_executable),
                        "-c",
                        download_script,
                        ACE_STEP_MODEL_REPOSITORY,
                        relative_path,
                        str(self.model_cache),
                    ],
                    cwd=self.runtime_dir,
                    environment=environment,
                    progress=progress,
                )
            except MusicGenerationError as error:
                raise MusicGenerationError(
                    f"Could not repair ACE-Step model metadata ({relative_path}). {error}"
                ) from error
        unresolved = [
            name for name in missing if not _valid_ace_step_config(self.model_cache / name)
        ]
        if unresolved:
            raise MusicGenerationError(
                "ACE-Step model metadata is still incomplete after repair: " + ", ".join(unresolved)
            )

    def _configuration(
        self,
        *,
        prompt: str,
        duration_seconds: float,
        bpm: int,
        seed: int,
        output_dir: Path,
    ) -> str:
        device = "cpu" if self.device == "cpu" else "auto"
        low_vram = (self.gpu_memory_mb or 0) <= 8 * 1024
        return "\n".join(
            (
                f"project_root = {_toml_string(self.runtime_dir)}",
                f"checkpoint_dir = {_toml_string(self.model_cache)}",
                f"config_path = {_toml_string(self.model.split('/')[-1])}",
                f"device = {_toml_string(device)}",
                f"save_dir = {_toml_string(output_dir)}",
                'task_type = "text2music"',
                f"caption = {_toml_string(prompt)}",
                'lyrics = "[Instrumental]"',
                "instrumental = true",
                f"duration = {duration_seconds:.3f}",
                f"bpm = {bpm}",
                'keyscale = "C Major"',
                'timesignature = "4"',
                "thinking = false",
                "use_cot_metas = false",
                "use_cot_caption = false",
                "use_cot_lyrics = false",
                "use_cot_language = false",
                "inference_steps = 16",
                "guidance_scale = 1.35",
                f"seed = {seed}",
                "batch_size = 1",
                "use_random_seed = false",
                'audio_format = "wav"',
                f"offload_to_cpu = {_toml_bool(low_vram)}",
                f"offload_dit_to_cpu = {_toml_bool(low_vram)}",
                "",
            )
        )

    def _run_streaming(
        self,
        command: list[str],
        *,
        cwd: Path,
        environment: dict[str, str],
        progress: MusicProgress | None,
    ) -> list[str]:
        try:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                encoding="utf-8",
                errors="replace",
                text=True,
            )
        except OSError as error:
            raise MusicGenerationError("Could not start ACE-Step.") from error

        lines: list[str] = []
        line_count = 0
        assert process.stdout is not None
        try:
            for raw_line in process.stdout:
                line = raw_line.strip()
                if not line:
                    continue
                line_count += 1
                lines.append(line)
                lines = lines[-100:]
                if progress:
                    lowered = line.casefold()
                    if "download" in lowered:
                        progress(1, 4, "ACE-Step: downloading model weights")
                    elif "initializ" in lowered or "loading" in lowered:
                        progress(2, 4, "ACE-Step: loading model into GPU/RAM")
                    elif "generat" in lowered:
                        progress(2, 4, "ACE-Step: generating composition")
                    elif line_count % 20 == 0:
                        progress(1, 4, "ACE-Step: preparing local components")
            return_code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            raise
        if return_code != 0:
            detail = _ace_step_error_detail(lines)
            raise MusicGenerationError(f"ACE-Step exited with code {return_code}. {detail}".strip())
        return lines

    def _normalize(
        self,
        source_path: Path,
        output_path: Path,
        duration_seconds: float,
    ) -> None:
        temporary_path = output_path.with_name(f".{output_path.stem}.ace-step.wav")
        try:
            completed = subprocess.run(
                [
                    self.ffmpeg_binary,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-stream_loop",
                    "-1",
                    "-i",
                    str(source_path),
                    "-t",
                    f"{duration_seconds:.3f}",
                    "-ar",
                    "44100",
                    "-ac",
                    "2",
                    "-c:a",
                    "pcm_s16le",
                    str(temporary_path),
                ],
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=self.ffmpeg_timeout_seconds,
            )
        except FileNotFoundError as error:
            temporary_path.unlink(missing_ok=True)
            raise MusicGenerationError(
                f"FFmpeg executable was not found: {self.ffmpeg_binary}"
            ) from error
        except subprocess.TimeoutExpired as error:
            temporary_path.unlink(missing_ok=True)
            raise MusicGenerationError(
                "FFmpeg timed out while normalizing ACE-Step output after "
                f"{self.ffmpeg_timeout_seconds:g}s."
            ) from error
        if completed.returncode != 0 or not temporary_path.is_file():
            temporary_path.unlink(missing_ok=True)
            detail = completed.stderr.strip()
            suffix = f" {detail}" if detail else ""
            raise MusicGenerationError(f"FFmpeg could not normalize ACE-Step output.{suffix}")
        os.replace(temporary_path, output_path)


def _toml_string(value: str | Path) -> str:
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _toml_bool(value: bool) -> str:
    return "true" if value else "false"


def _ace_step_error_detail(lines: list[str]) -> str:
    markers = ("error", "failed", "traceback", "not fully initialized", "unrecognized model")
    diagnostic = [line for line in lines if any(marker in line.casefold() for marker in markers)]
    selected = diagnostic[-8:] if diagnostic else lines[-8:]
    return " ".join(selected)[-2000:]


def _valid_ace_step_config(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict) or not payload:
        return False
    if path.parent.name == "Qwen3-Embedding-0.6B":
        return isinstance(payload.get("model_type"), str)
    return True


def _prompt_with_cue_sheet(prompt: str, cue_sheet: list[MusicCueSection]) -> str:
    if not cue_sheet:
        return prompt
    cues = "; ".join(
        (
            f"{section.role} {section.start_seconds:.1f}-{section.end_seconds:.1f}s "
            f"intensity {section.intensity:.2f}"
        )
        for section in cue_sheet[:8]
    )
    if len(cue_sheet) > 8:
        cues = f"{cues}; {len(cue_sheet) - 8} additional gentle sections"
    return f"{prompt} Cue sheet: {cues}"[:900]
