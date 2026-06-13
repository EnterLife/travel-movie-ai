"""Local executable readiness checks."""

import importlib
import shutil
import subprocess
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
