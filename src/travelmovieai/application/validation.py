"""Shared validation for project input and workspace paths."""

import os
from dataclasses import dataclass
from pathlib import Path

from travelmovieai.core.exceptions import InvalidProjectPathError


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    input_path: Path
    workspace: Path


def validate_output_path(
    output_path: Path,
    input_path: Path,
    *,
    workspace: Path | None = None,
    database_path: Path | None = None,
) -> Path:
    resolved_output = output_path.expanduser().resolve()
    resolved_input = input_path.expanduser().resolve()
    if resolved_output.suffix.casefold() != ".mp4":
        raise InvalidProjectPathError("Movie output must use the .mp4 extension.")
    if resolved_output.is_relative_to(resolved_input):
        raise InvalidProjectPathError("Movie output must be outside the source media folder.")
    if database_path is not None and resolved_output == database_path.expanduser().resolve():
        raise InvalidProjectPathError("Movie output must not overwrite the project database.")
    if workspace is not None:
        resolved_workspace = workspace.expanduser().resolve()
        reserved_roots = (resolved_workspace / "cache", resolved_workspace / "frames")
        if any(resolved_output.is_relative_to(root) for root in reserved_roots):
            raise InvalidProjectPathError(
                "Movie output must be outside workspace cache and frame folders."
            )
    if resolved_output.exists() and not resolved_output.is_file():
        raise InvalidProjectPathError("Movie output path must be a file.")
    return resolved_output


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
    if resolved_input.is_relative_to(resolved_workspace) or resolved_workspace.is_relative_to(
        resolved_input
    ):
        raise InvalidProjectPathError(
            "Workspace and source folder cannot be the same or nested inside each other."
        )

    return ProjectPaths(input_path=resolved_input, workspace=resolved_workspace)


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate
