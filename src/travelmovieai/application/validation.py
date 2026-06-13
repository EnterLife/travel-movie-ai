"""Shared validation for project input and workspace paths."""

import os
from dataclasses import dataclass
from pathlib import Path

from travelmovieai.core.exceptions import InvalidProjectPathError


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    input_path: Path
    workspace: Path


def validate_project_paths(input_path: Path, workspace: Path) -> ProjectPaths:
    resolved_input = input_path.expanduser().resolve()
    resolved_workspace = workspace.expanduser().resolve()

    if not resolved_input.exists():
        raise InvalidProjectPathError("Исходная папка не существует.")
    if not resolved_input.is_dir():
        raise InvalidProjectPathError("Исходный путь должен быть папкой.")
    if not os.access(resolved_input, os.R_OK):
        raise InvalidProjectPathError("Нет прав на чтение исходной папки.")
    if resolved_workspace.exists() and not resolved_workspace.is_dir():
        raise InvalidProjectPathError("Workspace должен быть папкой.")
    writable_root = _nearest_existing_parent(resolved_workspace)
    if not os.access(writable_root, os.W_OK):
        raise InvalidProjectPathError("Нет прав на запись в каталог workspace.")
    if resolved_input.is_relative_to(resolved_workspace):
        raise InvalidProjectPathError(
            "Workspace не может совпадать с исходной папкой или быть её родителем."
        )

    return ProjectPaths(input_path=resolved_input, workspace=resolved_workspace)


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate
