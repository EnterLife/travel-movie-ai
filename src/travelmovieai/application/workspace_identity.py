"""Stable workspace naming and source ownership checks."""

from __future__ import annotations

import hashlib
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, ValidationError

from travelmovieai.application.workspace_lease import (
    WORKSPACE_LEASE_FILENAME,
    WORKSPACE_LEASE_METADATA_FILENAME,
)
from travelmovieai.core.exceptions import WorkspaceIdentityError
from travelmovieai.domain.models import MediaScanReport

WORKSPACE_IDENTITY_FILENAME = ".travelmovieai-project.json"
_PROJECT_NAME_PATTERN = re.compile(r"[^\w.-]+", flags=re.UNICODE)


class ProjectWorkspaceIdentity(BaseModel):
    """Persistent binding between one workspace and one media source root."""

    schema_version: Literal[1] = 1
    source_root: Path
    source_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: datetime


def default_workspace_path(workspace_root: Path, input_path: Path) -> Path:
    """Return a deterministic workspace path unique to the resolved source root."""
    resolved_input = input_path.expanduser().resolve()
    fingerprint = source_fingerprint(resolved_input)
    source_name = _safe_project_name(resolved_input.name)
    return (workspace_root.expanduser().resolve() / f"{source_name}-{fingerprint[:16]}").resolve()


def legacy_workspace_path(workspace_root: Path, input_path: Path) -> Path:
    """Return the pre-identity basename-only workspace location."""
    resolved_input = input_path.expanduser().resolve()
    return (workspace_root.expanduser().resolve() / resolved_input.name).resolve()


def workspace_proves_source(workspace: Path, input_path: Path) -> bool:
    """Return whether durable workspace artifacts identify the requested source."""
    resolved_workspace = workspace.expanduser().resolve()
    if not resolved_workspace.is_dir():
        return False
    identity_path = resolved_workspace / WORKSPACE_IDENTITY_FILENAME
    if identity_path.is_file():
        try:
            identity = _read_identity(identity_path)
        except WorkspaceIdentityError:
            return False
        return _identity_matches(identity, input_path)
    return _legacy_analysis_matches(resolved_workspace, input_path)


def validate_existing_workspace_identity(input_path: Path, workspace: Path) -> None:
    """Reject an existing non-empty workspace that cannot prove source ownership."""
    resolved_workspace = workspace.expanduser().resolve()
    if not resolved_workspace.exists():
        return
    identity_path = resolved_workspace / WORKSPACE_IDENTITY_FILENAME
    if identity_path.is_file():
        identity = _read_identity(identity_path)
        if not _identity_matches(identity, input_path):
            raise WorkspaceIdentityError("Workspace is already bound to a different source folder.")
        return
    try:
        lease_files = {WORKSPACE_LEASE_FILENAME, WORKSPACE_LEASE_METADATA_FILENAME}
        has_entries = any(
            entry.name not in lease_files
            and not entry.name.startswith(f"{WORKSPACE_LEASE_METADATA_FILENAME}.")
            for entry in resolved_workspace.iterdir()
        )
    except OSError as error:
        raise WorkspaceIdentityError("Could not inspect the workspace identity.") from error
    if has_entries and not _legacy_analysis_matches(resolved_workspace, input_path):
        raise WorkspaceIdentityError(
            "Existing workspace has no valid identity for this source folder."
        )


def ensure_workspace_identity(input_path: Path, workspace: Path) -> ProjectWorkspaceIdentity:
    """Validate or atomically create the workspace-to-source binding."""
    resolved_workspace = workspace.expanduser().resolve()
    validate_existing_workspace_identity(input_path, resolved_workspace)
    identity_path = resolved_workspace / WORKSPACE_IDENTITY_FILENAME
    if identity_path.is_file():
        return _read_identity(identity_path)

    resolved_workspace.mkdir(parents=True, exist_ok=True)
    # Recheck after mkdir so another local caller cannot silently rebind a workspace.
    validate_existing_workspace_identity(input_path, resolved_workspace)
    if identity_path.is_file():
        return _read_identity(identity_path)

    resolved_input = input_path.expanduser().resolve()
    identity = ProjectWorkspaceIdentity(
        source_root=resolved_input,
        source_fingerprint=source_fingerprint(resolved_input),
        created_at=datetime.now(UTC),
    )
    persisted = _write_identity_once(identity_path, identity)
    if not _identity_matches(persisted, resolved_input):
        raise WorkspaceIdentityError(
            "Workspace identity changed while the project was being prepared."
        )
    return persisted


def source_fingerprint(input_path: Path) -> str:
    """Hash a canonical source path using Windows-safe normalization."""
    normalized = _normalized_path(input_path)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _read_identity(path: Path) -> ProjectWorkspaceIdentity:
    try:
        identity = ProjectWorkspaceIdentity.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as error:
        raise WorkspaceIdentityError("Workspace identity manifest is invalid.") from error
    if identity.source_fingerprint != source_fingerprint(identity.source_root):
        raise WorkspaceIdentityError("Workspace identity manifest is inconsistent.")
    return identity


def _write_identity_once(
    path: Path,
    identity: ProjectWorkspaceIdentity,
) -> ProjectWorkspaceIdentity:
    """Publish a complete manifest without replacing a concurrent source claim."""
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary_path.open("x", encoding="utf-8") as file:
            file.write(identity.model_dump_json(indent=2))
            file.flush()
            os.fsync(file.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            return _read_identity(path)
        except OSError as error:
            raise WorkspaceIdentityError(
                "Could not create the workspace identity manifest atomically."
            ) from error
        return _read_identity(path)
    except WorkspaceIdentityError:
        raise
    except OSError as error:
        raise WorkspaceIdentityError("Could not persist the workspace identity.") from error
    finally:
        temporary_path.unlink(missing_ok=True)


def _identity_matches(identity: ProjectWorkspaceIdentity, input_path: Path) -> bool:
    resolved_input = input_path.expanduser().resolve()
    return _normalized_path(identity.source_root) == _normalized_path(
        resolved_input
    ) and identity.source_fingerprint == source_fingerprint(resolved_input)


def _legacy_analysis_matches(workspace: Path, input_path: Path) -> bool:
    analysis_path = workspace / "artifacts" / "analysis.json"
    try:
        report = MediaScanReport.model_validate_json(analysis_path.read_text(encoding="utf-8"))
    except (OSError, ValidationError):
        return False
    return _normalized_path(report.input_path) == _normalized_path(input_path)


def _normalized_path(path: Path) -> str:
    normalized = os.path.normpath(str(path.expanduser().resolve()))
    if os.name == "nt":
        normalized = os.path.normcase(normalized).casefold()
    return normalized


def _safe_project_name(value: str) -> str:
    normalized = _PROJECT_NAME_PATTERN.sub("-", value).strip(" .-_")
    return normalized[:80] or "project"
