"""Adapters for specialized local music generation models."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from travelmovieai.core.exceptions import MusicGenerationError
from travelmovieai.core.security import sanitize_process_error
from travelmovieai.domain.models import MusicCueSection
from travelmovieai.infrastructure.processes import (
    release_process_resources,
    start_process,
    terminate_process_tree,
)

LOCAL_MUSIC_MODELS = (
    "ACE-Step/acestep-v15-turbo",
    "ACE-Step/acestep-v15-sft",
    "ACE-Step/acestep-v15-base",
    "ACE-Step/acestep-v15-xl-turbo",
    "ACE-Step/acestep-v15-xl-sft",
    "ACE-Step/acestep-v15-xl-base",
)
ACE_STEP_REPOSITORY = "https://github.com/ACE-Step/ACE-Step-1.5.git"
ACE_STEP_MODEL_REPOSITORY = "ACE-Step/Ace-Step1.5"
ACE_STEP_MAX_GENERATION_SECONDS = 600.0
ACE_STEP_REQUIRED_CONFIGS = (
    "Qwen3-Embedding-0.6B/config.json",
    "vae/config.json",
)

MusicProgress = Callable[[int, int, str], None]
CancelCheck = Callable[[], bool]


def resolve_local_music_model(
    configured_model: str | None,
    *,
    gpu_memory_mb: int | None,
    quality: Literal["draft", "balanced", "studio"] = "balanced",
) -> str:
    """Resolve the default model while preserving explicit selections."""

    if configured_model and configured_model != "auto":
        return configured_model
    memory = gpu_memory_mb or 0
    if quality == "studio":
        if memory >= 20 * 1024:
            return "ACE-Step/acestep-v15-xl-sft"
        if memory >= 12 * 1024:
            return "ACE-Step/acestep-v15-sft"
    if quality == "balanced" and memory >= 20 * 1024:
        return "ACE-Step/acestep-v15-xl-turbo"
    return "ACE-Step/acestep-v15-turbo"


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
        process_timeout_seconds: float | None = None,
        cancel_requested: CancelCheck | None = None,
        quality: Literal["draft", "balanced", "studio"] = "balanced",
        reference_audio: Path | None = None,
        reference_strength: float = 0.2,
        lora_path: Path | None = None,
        lora_strength: float = 0.7,
    ) -> None:
        self.model = model
        self.runtime_dir = runtime_dir
        self.model_cache = model_cache
        self.ffmpeg_binary = ffmpeg_binary
        self.allow_download = allow_download
        self.device = device
        self.gpu_memory_mb = gpu_memory_mb
        self.ffmpeg_timeout_seconds = ffmpeg_timeout_seconds if ffmpeg_timeout_seconds > 0 else 7200
        self.process_timeout_seconds = (
            process_timeout_seconds
            if process_timeout_seconds is not None and process_timeout_seconds > 0
            else self.ffmpeg_timeout_seconds
        )
        self.cancel_requested = cancel_requested
        self.quality = quality
        self.reference_audio = reference_audio
        self.reference_strength = reference_strength
        self.lora_path = lora_path
        self.lora_strength = lora_strength

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
        generated = self.generate_candidates(
            output_path.parent / f".{output_path.stem}-candidates",
            prompt=prompt,
            cue_sheet=cue_sheet,
            duration_seconds=duration_seconds,
            bpm=bpm,
            keyscale="C Major",
            seeds=[seed],
            progress=progress,
        )
        os.replace(generated[0][0], output_path)

    def generate_candidates(
        self,
        output_dir: Path,
        *,
        prompt: str,
        cue_sheet: list[MusicCueSection],
        duration_seconds: float,
        bpm: int,
        keyscale: str,
        seeds: list[int],
        progress: MusicProgress | None = None,
    ) -> list[tuple[Path, int]]:
        if not seeds:
            raise MusicGenerationError("ACE-Step requires at least one candidate seed.")
        if duration_seconds > ACE_STEP_MAX_GENERATION_SECONDS:
            raise MusicGenerationError(
                "ACE-Step supports native compositions up to 600 seconds. "
                "Use AI Auto for procedural fallback or provide a manual/library soundtrack."
            )
        self._ensure_runtime(progress)
        self._ensure_model_configs(progress)
        python_executable = self.runtime_dir / ".venv" / "Scripts" / "python.exe"
        if not python_executable.is_file():
            raise MusicGenerationError("The isolated ACE-Step environment is incomplete.")

        output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".ace-step-",
            dir=output_dir,
        ) as temporary:
            temporary_dir = Path(temporary)
            model_output = temporary_dir / "output"
            model_output.mkdir()
            request_path = temporary_dir / "generation.json"
            generation_duration = max(10.0, duration_seconds)
            request_path.write_text(
                json.dumps(
                    self._worker_request(
                        prompt=_prompt_with_cue_sheet(prompt, cue_sheet),
                        duration_seconds=generation_duration,
                        bpm=bpm,
                        keyscale=keyscale,
                        seeds=seeds,
                        output_dir=model_output,
                    ),
                    ensure_ascii=False,
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
                    str(Path(__file__).with_name("ace_step_worker.py")),
                    "--request",
                    str(request_path),
                ],
                cwd=self.runtime_dir,
                environment=environment,
                progress=progress,
            )
            worker_candidates = _worker_candidates(lines, model_output)
            if len(worker_candidates) != len(seeds):
                detail = _ace_step_error_detail(lines)
                raise MusicGenerationError(
                    "ACE-Step did not create all requested candidates. " + detail
                )
            if progress:
                progress(3, 4, "ACE-Step: preparing lossless 48 kHz candidates")
            prepared: list[tuple[Path, int]] = []
            for index, (source, seed) in enumerate(worker_candidates):
                destination = output_dir / f"candidate-{index + 1:02d}-{seed}.wav"
                self._normalize(source, destination, duration_seconds)
                prepared.append((destination, seed))
            if progress:
                progress(4, 4, f"ACE-Step: created {len(prepared)} candidate(s)")
            return prepared

    def _ensure_runtime(self, progress: MusicProgress | None) -> None:
        executable = self.runtime_dir / ".venv" / "Scripts" / "python.exe"
        if executable.is_file():
            return
        if not self.allow_download:
            raise MusicGenerationError("Auto-download is disabled and ACE-Step runtime is missing.")
        setup_script = _find_setup_script()
        if setup_script is None:
            packaged = bool(getattr(sys, "frozen", False))
            detail = (
                "The packaged application does not include the ACE-Step setup runtime. "
                "Reinstall with AI music support or configure an existing .cache\\ace-step runtime."
                if packaged
                else "scripts\\setup_windows.bat was not found. Reinstall the project files."
            )
            raise MusicGenerationError(detail)
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
                "--non-interactive",
            ],
            cwd=setup_script.parent.parent,
            environment=os.environ.copy(),
            progress=progress,
        )
        if not executable.is_file():
            raise MusicGenerationError("Could not install ACE-Step runtime.")

    def _ensure_model_configs(self, progress: MusicProgress | None) -> None:
        required = [*ACE_STEP_REQUIRED_CONFIGS]
        required.append(f"{self.model.split('/')[-1]}/config.json")
        lm_model = self._lm_model()
        if lm_model is not None:
            required.append(f"{lm_model}/config.json")
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

    def _worker_request(
        self,
        *,
        prompt: str,
        duration_seconds: float,
        bpm: int,
        keyscale: str,
        seeds: list[int],
        output_dir: Path,
    ) -> dict[str, object]:
        turbo = "turbo" in self.model.casefold()
        inference_steps = 8 if turbo else (64 if self.quality == "studio" else 48)
        if turbo and self.quality == "studio":
            inference_steps = 12
        lm_model = self._lm_model()
        low_vram = (self.gpu_memory_mb or 0) <= 8 * 1024
        return {
            "project_root": str(self.runtime_dir),
            "checkpoint_dir": str(self.model_cache),
            "config_path": self.model.split("/")[-1],
            "device": "cpu" if self.device == "cpu" else "auto",
            "offload_to_cpu": low_vram,
            "offload_dit_to_cpu": low_vram,
            "prompt": prompt,
            "duration_seconds": duration_seconds,
            "bpm": bpm,
            "keyscale": keyscale,
            "timesignature": "4",
            "inference_steps": inference_steps,
            "guidance_scale": 1.0 if turbo else 7.5,
            "use_adg": not turbo and self.quality == "studio",
            "shift": 3.0,
            "seeds": seeds,
            "output_dir": str(output_dir),
            "thinking": lm_model is not None,
            "lm_model": lm_model,
            "lm_backend": "pt" if (self.gpu_memory_mb or 0) < 12 * 1024 else "vllm",
            "reference_audio": str(self.reference_audio) if self.reference_audio else None,
            "reference_strength": self.reference_strength,
            "lora_path": str(self.lora_path) if self.lora_path else None,
            "lora_strength": self.lora_strength,
        }

    def _lm_model(self) -> str | None:
        memory = self.gpu_memory_mb or 0
        if self.quality != "studio" or memory < 8 * 1024:
            return None
        if memory >= 24 * 1024:
            return "acestep-5Hz-lm-4B"
        if memory >= 12 * 1024:
            return "acestep-5Hz-lm-1.7B"
        return "acestep-5Hz-lm-0.6B"

    def _run_streaming(
        self,
        command: list[str],
        *,
        cwd: Path,
        environment: dict[str, str],
        progress: MusicProgress | None,
    ) -> list[str]:
        try:
            process = start_process(
                command,
                popen_factory=subprocess.Popen,
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
        reader: threading.Thread | None = None
        try:
            stdout = process.stdout
            assert stdout is not None
            output_queue: queue.Queue[str | None] = queue.Queue()

            def read_output() -> None:
                try:
                    for raw_line in stdout:
                        output_queue.put(raw_line)
                finally:
                    output_queue.put(None)

            reader = threading.Thread(
                target=read_output,
                name="travelmovieai-ace-step-output",
                daemon=True,
            )
            reader.start()
            deadline = time.monotonic() + self.process_timeout_seconds
            while True:
                if self.cancel_requested is not None and self.cancel_requested():
                    raise MusicGenerationError("ACE-Step generation was cancelled.")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise MusicGenerationError(
                        "ACE-Step timed out after "
                        f"{self.process_timeout_seconds:g}s; the process tree was stopped."
                    )
                try:
                    raw_line = output_queue.get(timeout=min(0.25, remaining))
                except queue.Empty:
                    if process.poll() is not None and not reader.is_alive():
                        break
                    continue
                if raw_line is None:
                    break
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
            return_code = process.wait(timeout=5)
        except BaseException:
            terminate_process_tree(process)
            raise
        finally:
            if reader is not None:
                reader.join(timeout=1)
            release_process_resources(process)
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
        command = [
            self.ffmpeg_binary,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source_path),
            "-af",
            "apad",
            "-t",
            f"{duration_seconds:.3f}",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-c:a",
            "pcm_s24le",
            str(temporary_path),
        ]
        try:
            process = start_process(
                command,
                popen_factory=subprocess.Popen,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding="utf-8",
                errors="replace",
                text=True,
            )
        except FileNotFoundError as error:
            temporary_path.unlink(missing_ok=True)
            raise MusicGenerationError(
                f"FFmpeg executable was not found: {self.ffmpeg_binary}"
            ) from error

        deadline = time.monotonic() + self.ffmpeg_timeout_seconds
        stderr = ""
        try:
            while process.poll() is None:
                if self.cancel_requested is not None and self.cancel_requested():
                    raise MusicGenerationError(
                        "ACE-Step output normalization was cancelled; the FFmpeg process "
                        "tree was stopped."
                    )
                if time.monotonic() >= deadline:
                    raise MusicGenerationError(
                        "FFmpeg timed out while normalizing ACE-Step output after "
                        f"{self.ffmpeg_timeout_seconds:g}s; the process tree was stopped."
                    )
                time.sleep(0.1)
            _, stderr = process.communicate(timeout=5)
        except BaseException:
            terminate_process_tree(process)
            temporary_path.unlink(missing_ok=True)
            raise
        finally:
            release_process_resources(process)
        if process.returncode != 0 or not temporary_path.is_file():
            temporary_path.unlink(missing_ok=True)
            detail = sanitize_process_error(
                stderr,
                private_paths=[source_path, temporary_path, output_path],
                fallback="",
            )
            suffix = f" {detail}" if detail else ""
            raise MusicGenerationError(f"FFmpeg could not normalize ACE-Step output.{suffix}")
        os.replace(temporary_path, output_path)


def _toml_string(value: str | Path) -> str:
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _find_setup_script() -> Path | None:
    candidates = [Path(__file__).resolve().parents[3] / "scripts" / "setup_windows.bat"]
    bundle_root = getattr(sys, "_MEIPASS", None)
    if isinstance(bundle_root, str):
        candidates.append(Path(bundle_root) / "scripts" / "setup_windows.bat")
    candidates.append(Path(sys.executable).resolve().parent / "scripts" / "setup_windows.bat")
    return next((path for path in candidates if path.is_file()), None)


def _toml_bool(value: bool) -> str:
    return "true" if value else "false"


def _ace_step_error_detail(lines: list[str]) -> str:
    markers = ("error", "failed", "traceback", "not fully initialized", "unrecognized model")
    diagnostic = [line for line in lines if any(marker in line.casefold() for marker in markers)]
    selected = diagnostic[-8:] if diagnostic else lines[-8:]
    return " ".join(selected)[-2000:]


def _worker_candidates(lines: list[str], output_dir: Path) -> list[tuple[Path, int]]:
    prefix = "TRAVELMOVIEAI_RESULT="
    marker = next((line for line in reversed(lines) if line.startswith(prefix)), None)
    if marker is None:
        return []
    try:
        payload = json.loads(marker[len(prefix) :])
        raw_candidates = payload["candidates"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return []
    resolved_output = output_dir.resolve()
    candidates: list[tuple[Path, int]] = []
    for item in raw_candidates:
        try:
            path = Path(item["path"]).resolve()
            seed = int(item["seed"])
        except (KeyError, TypeError, ValueError):
            return []
        if seed < 0 or not path.is_relative_to(resolved_output) or not path.is_file():
            return []
        candidates.append((path, seed))
    return candidates


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
    if not cue_sheet or "macro arrangement:" in prompt.casefold():
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
    return f"{prompt} Macro arrangement: {cues}"[:1400]
