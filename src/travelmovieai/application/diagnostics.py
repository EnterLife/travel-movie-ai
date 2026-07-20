"""Non-mutating local runtime and model diagnostics."""

import shutil
import subprocess
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Literal

from travelmovieai.core.config import Settings
from travelmovieai.infrastructure.system import (
    CudaStatus,
    ExecutableStatus,
    ResourceProfile,
    check_cuda,
    check_executable,
    detect_resource_profile,
)
from travelmovieai.infrastructure.vision import resolve_local_vision_model

DiagnosticLevel = Literal["ok", "warning", "error"]


@dataclass(frozen=True, slots=True)
class DiagnosticCheck:
    name: str
    level: DiagnosticLevel
    message: str


@dataclass(frozen=True, slots=True)
class SystemDiagnosticReport:
    ready: bool
    checks: tuple[DiagnosticCheck, ...]


def run_system_diagnostics(settings: Settings) -> SystemDiagnosticReport:
    checks: list[DiagnosticCheck] = []
    ffmpeg = check_executable(settings.ffmpeg_binary)
    ffprobe = check_executable(settings.ffprobe_binary)
    checks.extend([_executable_check(ffmpeg), _executable_check(ffprobe)])
    checks.append(_ffmpeg_filter_check(settings.ffmpeg_binary, ffmpeg.available))
    cuda = check_cuda(settings.ffmpeg_binary)
    checks.append(_cuda_check(cuda, settings.device))
    resources = detect_resource_profile(
        settings.ffmpeg_binary,
        cuda=cuda,
        worker_override=settings.workers,
        batch_override=settings.batch_size,
        resource_mode=settings.resource_mode,
        gpu_memory_reserve_mb=settings.gpu_memory_reserve_mb,
        max_gpu_processes=settings.max_gpu_processes,
    )
    checks.extend(_vision_checks(settings))
    checks.extend(_embedding_checks(settings))
    checks.extend(_speech_checks(settings))
    checks.extend(_voice_checks(settings))
    checks.append(_model_cache_check(settings, resources))
    return SystemDiagnosticReport(
        ready=not any(check.level == "error" for check in checks),
        checks=tuple(checks),
    )


def _executable_check(status: ExecutableStatus) -> DiagnosticCheck:
    if status.available:
        version = status.version or "version available"
        return DiagnosticCheck(status.name, "ok", version[:300])
    return DiagnosticCheck(
        status.name,
        "error",
        status.error or "Required executable is unavailable.",
    )


def _ffmpeg_filter_check(binary: str, ffmpeg_available: bool) -> DiagnosticCheck:
    if not ffmpeg_available:
        return DiagnosticCheck(
            "FFmpeg filters",
            "warning",
            "Filter capabilities cannot be checked until FFmpeg is available.",
        )
    filters = _ffmpeg_filters(binary)
    if filters is None:
        return DiagnosticCheck(
            "FFmpeg filters",
            "warning",
            "Could not inspect optional title, HDR, and tone-mapping filters.",
        )
    required = {"drawtext", "zscale", "tonemap"}
    missing = sorted(required - filters)
    if missing:
        return DiagnosticCheck(
            "FFmpeg filters",
            "warning",
            f"Optional render filters are unavailable: {', '.join(missing)}.",
        )
    return DiagnosticCheck(
        "FFmpeg filters",
        "ok",
        "Title, HDR conversion, and tone-mapping filters are available.",
    )


def _ffmpeg_filters(binary: str) -> set[str] | None:
    resolved = shutil.which(binary) or binary
    try:
        completed = subprocess.run(
            [resolved, "-hide_banner", "-filters"],
            capture_output=True,
            check=False,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    filters: set[str] = set()
    for line in completed.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and len(parts[0]) <= 4:
            filters.add(parts[1])
    return filters


def _cuda_check(status: CudaStatus, configured_device: str) -> DiagnosticCheck:
    if status.ffmpeg_nvenc and status.torch_cuda:
        return DiagnosticCheck(
            "CUDA",
            "ok",
            "CUDA inference and NVENC rendering are available.",
        )
    if configured_device == "cuda":
        return DiagnosticCheck(
            "CUDA",
            "error",
            "CUDA was required by configuration but the complete runtime is unavailable.",
        )
    return DiagnosticCheck(
        "CUDA",
        "warning",
        "Complete CUDA acceleration is unavailable; CPU fallback remains active.",
    )


def _vision_checks(settings: Settings) -> list[DiagnosticCheck]:
    packages = ("torch", "transformers", "accelerate")
    missing = [package for package in packages if find_spec(package) is None]
    if missing:
        return [
            DiagnosticCheck(
                "Vision AI",
                "warning",
                f"Optional vision packages are missing: {', '.join(missing)}.",
            )
        ]
    return [DiagnosticCheck("Vision AI", "ok", "Local vision runtime packages are installed.")]


def _embedding_checks(settings: Settings) -> list[DiagnosticCheck]:
    if settings.embedding_backend == "feature-hash":
        level: DiagnosticLevel = "ok"
        message = "Deterministic model-free embeddings are configured."
    elif find_spec("sentence_transformers") is None:
        level = "error"
        message = "Configured sentence-transformers backend is not installed."
    else:
        level = "ok"
        message = "Sentence-transformers backend is installed."
    checks = [DiagnosticCheck("Embeddings", level, message)]
    if settings.embedding_index != "disabled" and find_spec("faiss") is None:
        index_level: DiagnosticLevel = "error" if settings.embedding_index == "faiss" else "warning"
        checks.append(
            DiagnosticCheck(
                "FAISS",
                index_level,
                "FAISS is unavailable; auto mode will skip archive indexing.",
            )
        )
    else:
        checks.append(DiagnosticCheck("FAISS", "ok", "FAISS archive indexing is available."))
    return checks


def _speech_checks(settings: Settings) -> list[DiagnosticCheck]:
    if find_spec("faster_whisper") is None:
        return [
            DiagnosticCheck(
                "Faster Whisper",
                "warning",
                "Optional speech recognition dependency is not installed.",
            )
        ]
    return [DiagnosticCheck("Faster Whisper", "ok", "Local speech runtime is installed.")]


def _voice_checks(settings: Settings) -> list[DiagnosticCheck]:
    if settings.voice_provider == "disabled":
        return [DiagnosticCheck("Piper", "warning", "Voice synthesis is disabled.")]
    if shutil.which(settings.piper_binary) is None and not settings.piper_model:
        return [
            DiagnosticCheck(
                "Piper",
                "error",
                "Piper executable and local voice model must be configured.",
            )
        ]
    if shutil.which(settings.piper_binary) is None:
        return [DiagnosticCheck("Piper", "error", "Configured Piper executable was not found.")]
    if settings.piper_model is None or not settings.piper_model.expanduser().is_file():
        return [DiagnosticCheck("Piper", "error", "Configured Piper voice model was not found.")]
    return [DiagnosticCheck("Piper", "ok", "Local Piper executable and voice model are ready.")]


def _model_cache_check(settings: Settings, resources: ResourceProfile) -> DiagnosticCheck:
    cache = settings.model_cache.expanduser()
    if settings.allow_model_download:
        return DiagnosticCheck(
            "Model cache",
            "ok",
            "Missing local models may be downloaded on first explicit use.",
        )
    required_models = _configured_offline_models(settings, resources)
    missing_models = [
        model for model in required_models if not model_snapshot_present(cache, model)
    ]
    if required_models and not missing_models:
        return DiagnosticCheck(
            "Model cache",
            "ok",
            "Offline cache-only mode contains every explicitly configured model.",
        )
    if missing_models:
        return DiagnosticCheck(
            "Model cache",
            "error",
            "Offline cache is missing configured snapshot(s): " + ", ".join(missing_models),
        )
    return DiagnosticCheck(
        "Model cache",
        "warning",
        "Offline cache-only mode is enabled, but the model cache is empty.",
    )


def _configured_offline_models(
    settings: Settings,
    resources: ResourceProfile,
) -> list[str]:
    models: list[str] = [_resolved_vision_model(settings, resources)]
    if settings.embedding_backend == "sentence-transformers":
        models.append(settings.embedding_model)
    if settings.story_provider == "local":
        models.append(settings.story_model)
    return list(dict.fromkeys(models))


def model_snapshot_present(cache: Path, model: str) -> bool:
    """Recognize complete direct or Hugging Face snapshots in supported cache layouts."""

    normalized = f"models--{model.replace('/', '--')}"
    cache_roots = tuple(
        cache / namespace for namespace in ("", "sentence-transformers", "faster-whisper", "story")
    )
    direct_candidates = tuple(
        candidate
        for root in cache_roots
        for candidate in (root / model, root / model.split("/")[-1])
    )
    snapshot_roots = tuple(root / normalized / "snapshots" for root in cache_roots)
    try:
        if any(_looks_like_model_directory(candidate) for candidate in direct_candidates):
            return True
        return any(
            root.is_dir()
            and any(_looks_like_model_directory(snapshot) for snapshot in root.iterdir())
            for root in snapshot_roots
        )
    except OSError:
        return False


def _looks_like_model_directory(path: Path) -> bool:
    if not path.is_dir() or not (path / "config.json").is_file():
        return False
    weight_patterns = ("*.safetensors", "*.bin", "*.onnx", "model.bin")
    return any(any(path.glob(pattern)) for pattern in weight_patterns)


def _resolved_vision_model(settings: Settings, resources: ResourceProfile) -> str:
    if settings.vision_provider == "florence":
        return (
            settings.vision_model
            if settings.vision_model != "auto"
            else "microsoft/Florence-2-large"
        )
    return resolve_local_vision_model(
        settings.vision_model,
        gpu_memory_mb=resources.gpu_memory_mb,
        system_memory_mb=resources.memory_mb,
    )
