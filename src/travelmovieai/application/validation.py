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
        raise InvalidProjectPathError("The source folder does not exist.")
    if not resolved_input.is_dir():
        raise InvalidProjectPathError("The source path must be a folder.")
    if not os.access(resolved_input, os.R_OK):
        raise InvalidProjectPathError("The source folder is not readable.")
    if resolved_workspace.exists() and not resolved_workspace.is_dir():
        raise InvalidProjectPathError("Workspace must be a folder.")
    writable_root = _nearest_existing_parent(resolved_workspace)
    if not os.access(writable_root, os.W_OK):
        raise InvalidProjectPathError("Workspace is not writable.")
    if resolved_input.is_relative_to(resolved_workspace):
        raise InvalidProjectPathError(
            "Workspace cannot be the source folder or one of its parents."
        )

    return ProjectPaths(input_path=resolved_input, workspace=resolved_workspace)


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate
