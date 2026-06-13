"""Local executable readiness checks."""

import importlib
import os
import shutil
import subprocess
import sys
from ctypes import Structure, byref, c_ulong, c_ulonglong, sizeof
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ExecutableStatus:
    name: str
    configured_value: str
    available: bool
    resolved_path: Path | None = None
    version: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class CudaStatus:
    available: bool
    gpu_name: str | None = None
    driver_version: str | None = None
    memory_mb: int | None = None
    compute_capability: str | None = None
    ffmpeg_nvenc: bool = False
    opencv_cuda_devices: int = 0
    torch_cuda: bool = False
    torch_version: str | None = None
    note: str | None = None


@dataclass(frozen=True, slots=True)
class ResourceProfile:
    logical_cores: int
    memory_mb: int | None
    gpu_name: str | None
    gpu_memory_mb: int | None
    nvenc: bool
    frame_workers: int
    analysis_workers: int
    render_workers: int
    ffmpeg_threads: int
    model_batch_size: int
    summary: str


class _MemoryStatus(Structure):
    _fields_ = [
        ("length", c_ulong),
        ("memory_load", c_ulong),
        ("total_physical", c_ulonglong),
        ("available_physical", c_ulonglong),
        ("total_page_file", c_ulonglong),
        ("available_page_file", c_ulonglong),
        ("total_virtual", c_ulonglong),
        ("available_virtual", c_ulonglong),
        ("available_extended_virtual", c_ulonglong),
    ]


def check_executable(binary: str, *, timeout_seconds: float = 5) -> ExecutableStatus:
    resolved = shutil.which(binary)
    if resolved is None:
        configured_path = Path(binary).expanduser()
        if configured_path.is_file():
            resolved = str(configured_path.resolve())

    if resolved is None:
        return ExecutableStatus(
            name=Path(binary).name,
            configured_value=binary,
            available=False,
            error="Исполняемый файл не найден.",
        )

    try:
        completed = subprocess.run(
            [resolved, "-version"],
            capture_output=True,
            check=False,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return ExecutableStatus(
            name=Path(binary).name,
            configured_value=binary,
            available=False,
            resolved_path=Path(resolved),
            error=str(error),
        )

    version = (completed.stdout or completed.stderr).splitlines()
    if completed.returncode != 0:
        return ExecutableStatus(
            name=Path(binary).name,
            configured_value=binary,
            available=False,
            resolved_path=Path(resolved),
            error=version[0] if version else "Не удалось получить версию.",
        )

    return ExecutableStatus(
        name=Path(binary).name,
        configured_value=binary,
        available=True,
        resolved_path=Path(resolved),
        version=version[0] if version else None,
    )


def check_cuda(ffmpeg_binary: str = "ffmpeg") -> CudaStatus:
    gpu = _nvidia_gpu()
    nvenc = _ffmpeg_has_nvenc(ffmpeg_binary)
    opencv_devices = 0
    torch_cuda = False
    torch_version = None

    try:
        cv2 = importlib.import_module("cv2")
        opencv_devices = int(cv2.cuda.getCudaEnabledDeviceCount())
    except (ImportError, AttributeError, RuntimeError):
        pass

    try:
        torch = importlib.import_module("torch")
        torch_version = str(torch.__version__)
        torch_cuda = bool(torch.cuda.is_available())
    except (ImportError, AttributeError, RuntimeError):
        pass

    note = None
    if gpu and not torch_cuda:
        note = (
            "NVIDIA GPU доступен. LM Studio использует GPU независимо; "
            "локальный PyTorch установлен без CUDA."
        )
    return CudaStatus(
        available=gpu is not None,
        gpu_name=gpu.get("name") if gpu else None,
        driver_version=gpu.get("driver") if gpu else None,
        memory_mb=int(gpu["memory"]) if gpu else None,
        compute_capability=gpu.get("compute") if gpu else None,
        ffmpeg_nvenc=nvenc,
        opencv_cuda_devices=opencv_devices,
        torch_cuda=torch_cuda,
        torch_version=torch_version,
        note=note,
    )


def detect_resource_profile(
    ffmpeg_binary: str = "ffmpeg",
    *,
    cuda: CudaStatus | None = None,
    worker_override: int = 0,
    batch_override: int = 0,
) -> ResourceProfile:
    logical_cores = max(1, os.cpu_count() or 1)
    memory_mb = _system_memory_mb()
    resolved_cuda = cuda or check_cuda(ffmpeg_binary)

    memory_factor = 1.0
    if memory_mb is not None:
        if memory_mb < 8 * 1024:
            memory_factor = 0.45
        elif memory_mb < 16 * 1024:
            memory_factor = 0.7

    automatic_frames = max(1, min(12, round(logical_cores * 0.6 * memory_factor)))
    automatic_analysis = max(1, min(16, round(logical_cores * 0.8 * memory_factor)))
    automatic_render = max(
        1,
        min(
            4 if resolved_cuda.ffmpeg_nvenc else 3,
            round(logical_cores / 4 * memory_factor),
        ),
    )
    frame_workers = worker_override or automatic_frames
    analysis_workers = worker_override or automatic_analysis
    render_workers = min(worker_override, 6) if worker_override else automatic_render
    render_workers = max(1, render_workers)
    ffmpeg_threads = max(1, logical_cores // render_workers)

    gpu_memory = resolved_cuda.memory_mb or 0
    automatic_batch = (
        16
        if gpu_memory >= 16 * 1024
        else 8
        if gpu_memory >= 10 * 1024
        else 4
        if gpu_memory >= 6 * 1024
        else max(1, min(4, logical_cores // 4))
    )
    model_batch_size = batch_override or automatic_batch
    accelerator = (
        f"{resolved_cuda.gpu_name}, NVENC"
        if resolved_cuda.available and resolved_cuda.ffmpeg_nvenc
        else resolved_cuda.gpu_name
        if resolved_cuda.available
        else "CPU"
    )
    memory_label = f"{memory_mb // 1024} GB RAM" if memory_mb else "RAM unknown"
    summary = (
        f"{logical_cores} CPU threads, {memory_label}, {accelerator}; "
        f"frames {frame_workers}x, analysis {analysis_workers}x, "
        f"render {render_workers}x/{ffmpeg_threads} threads"
    )
    return ResourceProfile(
        logical_cores=logical_cores,
        memory_mb=memory_mb,
        gpu_name=resolved_cuda.gpu_name,
        gpu_memory_mb=resolved_cuda.memory_mb,
        nvenc=resolved_cuda.ffmpeg_nvenc,
        frame_workers=frame_workers,
        analysis_workers=analysis_workers,
        render_workers=render_workers,
        ffmpeg_threads=ffmpeg_threads,
        model_batch_size=model_batch_size,
        summary=summary,
    )


def _system_memory_mb() -> int | None:
    if sys.platform != "win32":
        return None
    import ctypes

    status = _MemoryStatus()
    status.length = sizeof(_MemoryStatus)
    try:
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(byref(status)):
            return None
    except (AttributeError, OSError):
        return None
    return int(status.total_physical // (1024 * 1024))


def _nvidia_gpu() -> dict[str, str] | None:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    try:
        completed = subprocess.run(
            [
                executable,
                "--query-gpu=name,driver_version,memory.total,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=False,
            encoding="utf-8",
            errors="replace",
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    values = [value.strip() for value in completed.stdout.splitlines()[0].split(",")]
    if len(values) != 4:
        return None
    return dict(zip(("name", "driver", "memory", "compute"), values, strict=True))


def _ffmpeg_has_nvenc(ffmpeg_binary: str) -> bool:
    resolved = shutil.which(ffmpeg_binary) or ffmpeg_binary
    try:
        completed = subprocess.run(
            [resolved, "-hide_banner", "-encoders"],
            capture_output=True,
            check=False,
            encoding="utf-8",
            errors="replace",
            timeout=8,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0 and "h264_nvenc" in completed.stdout
