"""Local executable readiness checks."""

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
