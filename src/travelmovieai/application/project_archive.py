"""Safe local project backup, export, and restore archives."""

import hashlib
import os
import shutil
import sqlite3
import stat
import unicodedata
import zipfile
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Final, Literal, Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from travelmovieai.application.context import ProjectContext
from travelmovieai.application.workspace_identity import (
    WORKSPACE_IDENTITY_FILENAME,
    ProjectWorkspaceIdentity,
    source_fingerprint,
)
from travelmovieai.core.exceptions import (
    InvalidProjectPathError,
    ProjectArchiveError,
)
from travelmovieai.infrastructure.database import MediaAssetRepository

ARCHIVE_SCHEMA_VERSION: Final[Literal[1]] = 1
MAX_ARCHIVE_ENTRIES = 100_000
MAX_ARCHIVE_MANIFEST_BYTES = 64 * 1024 * 1024
MAX_RESTORE_BYTES = 100 * 1024 * 1024 * 1024
MEDIA_OUTPUT_SUFFIXES = {".mp4", ".mkv", ".mov", ".wav", ".mp3", ".m4a", ".flac"}


class ProjectArchiveFile(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    path: str = Field(min_length=1, max_length=4096)
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("path")
    @classmethod
    def validate_safe_path(cls, value: str) -> str:
        archive_path = PurePosixPath(value)
        if (
            archive_path.is_absolute()
            or ".." in archive_path.parts
            or "\\" in value
            or ":" in value
            or value == "."
            or value != archive_path.as_posix()
            or value.casefold() == "backup_manifest.json"
            or any(ord(character) < 32 for character in value)
        ):
            raise ValueError("archive file path must be canonical and relative")
        return value


class ProjectArchiveManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[1]
    created_at: datetime
    project_name: str = Field(min_length=1, max_length=255)
    includes_rendered_media: bool
    files: list[ProjectArchiveFile] = Field(min_length=1, max_length=MAX_ARCHIVE_ENTRIES)

    @field_validator("created_at")
    @classmethod
    def require_aware_created_at(cls, value: datetime) -> datetime:
        if value.utcoffset() is None:
            raise ValueError("archive creation time must include a timezone")
        return value

    @model_validator(mode="after")
    def require_unique_paths(self) -> Self:
        paths = [_archive_path_key(item.path) for item in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("archive manifest file paths must be unique")
        return self


def _archive_path_key(value: str) -> str:
    return unicodedata.normalize("NFC", value).casefold()


@dataclass(frozen=True, slots=True)
class ProjectArchiveResult:
    archive_path: Path
    file_count: int
    total_bytes: int
    includes_rendered_media: bool


def export_project_archive(
    context: ProjectContext,
    output_path: Path,
    *,
    include_rendered_media: bool = False,
    overwrite: bool = False,
) -> ProjectArchiveResult:
    context.prepare()
    repository = MediaAssetRepository(context.database_path)
    repository.initialize()
    resolved_output = _validate_archive_output(context, output_path, overwrite=overwrite)
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_archive = resolved_output.with_name(f".{resolved_output.stem}.{uuid4().hex}.tmp.zip")
    database_snapshot = resolved_output.with_name(
        f".{resolved_output.stem}.{uuid4().hex}.project.tmp.db"
    )
    file_count = 0
    total_bytes = 0
    try:
        _snapshot_database(context.database_path, database_snapshot)
        sources = [(database_snapshot, PurePosixPath(context.settings.database_filename))]
        sources.extend(
            _archive_sources(
                context,
                include_rendered_media=include_rendered_media,
            )
        )
        manifest_files = [
            ProjectArchiveFile(
                path=archive_path.as_posix(),
                size_bytes=source.stat().st_size,
                sha256=_sha256(source),
            )
            for source, archive_path in sources
        ]
        file_count = len(manifest_files)
        total_bytes = sum(source.stat().st_size for source, _ in sources)
        manifest = ProjectArchiveManifest(
            schema_version=ARCHIVE_SCHEMA_VERSION,
            created_at=datetime.now(UTC),
            project_name=context.input_path.name or "project",
            includes_rendered_media=include_rendered_media,
            files=manifest_files,
        )
        with zipfile.ZipFile(
            temporary_archive,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as archive:
            archive.writestr(
                "backup_manifest.json",
                manifest.model_dump_json(indent=2),
            )
            for source, archive_path in sources:
                archive.write(source, archive_path.as_posix())
        os.replace(temporary_archive, resolved_output)
    except ProjectArchiveError:
        raise
    except (OSError, sqlite3.Error, ValidationError, zipfile.BadZipFile) as error:
        raise ProjectArchiveError(
            "Could not create the local project archive; verify the output path and free space."
        ) from error
    finally:
        temporary_archive.unlink(missing_ok=True)
        database_snapshot.unlink(missing_ok=True)
    return ProjectArchiveResult(
        archive_path=resolved_output,
        file_count=file_count,
        total_bytes=total_bytes,
        includes_rendered_media=include_rendered_media,
    )


def restore_project_archive(archive_path: Path, workspace: Path) -> Path:
    resolved_archive = archive_path.expanduser().resolve()
    resolved_workspace = workspace.expanduser().resolve()
    if not resolved_archive.is_file() or resolved_archive.suffix.casefold() != ".zip":
        raise InvalidProjectPathError("Project archive must be an existing .zip file.")
    if resolved_workspace.exists() and any(resolved_workspace.iterdir()):
        raise InvalidProjectPathError("Restore workspace must be absent or empty.")
    if resolved_workspace == resolved_archive.parent or resolved_archive.is_relative_to(
        resolved_workspace
    ):
        raise InvalidProjectPathError("Restore workspace conflicts with the archive path.")
    temporary_workspace = resolved_workspace.with_name(
        f".{resolved_workspace.name}.restore.{uuid4().hex}"
    )
    try:
        with zipfile.ZipFile(resolved_archive) as archive:
            manifest = _validated_manifest(archive)
            _extract_validated(archive, manifest, temporary_workspace)
            _validate_restored_identity(temporary_workspace)
        if resolved_workspace.exists():
            resolved_workspace.rmdir()
        os.replace(temporary_workspace, resolved_workspace)
    except ProjectArchiveError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise ProjectArchiveError(
            "Could not restore the local project archive; it may be corrupt or unsafe."
        ) from error
    finally:
        if temporary_workspace.exists():
            shutil.rmtree(temporary_workspace, ignore_errors=True)
    return resolved_workspace


def _validate_archive_output(
    context: ProjectContext,
    output_path: Path,
    *,
    overwrite: bool,
) -> Path:
    resolved = output_path.expanduser().resolve()
    if resolved.suffix.casefold() != ".zip":
        raise InvalidProjectPathError("Project archive output must use the .zip extension.")
    if resolved.is_relative_to(context.input_path.resolve()) or resolved.is_relative_to(
        context.workspace.resolve()
    ):
        raise InvalidProjectPathError(
            "Project archive output must be outside the source and workspace folders."
        )
    if resolved.exists() and (not resolved.is_file() or not overwrite):
        raise InvalidProjectPathError(
            "Project archive output already exists; enable overwrite explicitly to replace it."
        )
    return resolved


def _snapshot_database(source: Path, destination: Path) -> None:
    with (
        closing(sqlite3.connect(source)) as source_connection,
        closing(sqlite3.connect(destination)) as destination_connection,
    ):
        source_connection.backup(destination_connection)


def _archive_sources(
    context: ProjectContext,
    *,
    include_rendered_media: bool,
) -> list[tuple[Path, PurePosixPath]]:
    sources: list[tuple[Path, PurePosixPath]] = []
    identity_path = context.workspace / WORKSPACE_IDENTITY_FILENAME
    if identity_path.exists():
        if identity_path.is_symlink() or not identity_path.is_file():
            raise ProjectArchiveError("Project identity manifest must be a regular file.")
        sources.append((identity_path, PurePosixPath(WORKSPACE_IDENTITY_FILENAME)))
    roots = [
        (context.artifacts_dir, PurePosixPath("artifacts")),
        (context.workspace / ".web", PurePosixPath(".web")),
    ]
    for root, prefix in roots:
        if not root.is_dir() or root.is_symlink():
            continue
        for path in sorted(root.rglob("*"), key=lambda item: item.as_posix().casefold()):
            if path.is_symlink() or not path.is_file():
                continue
            if not include_rendered_media and path.suffix.casefold() in MEDIA_OUTPUT_SUFFIXES:
                continue
            relative = path.relative_to(root)
            sources.append((path, prefix / PurePosixPath(relative.as_posix())))
    return sources


def _validate_restored_identity(workspace: Path) -> None:
    identity_path = workspace / WORKSPACE_IDENTITY_FILENAME
    if not identity_path.exists():
        return
    if identity_path.is_symlink() or not identity_path.is_file():
        raise ValueError("Project identity manifest must be a regular file.")
    try:
        identity = ProjectWorkspaceIdentity.model_validate_json(
            identity_path.read_text(encoding="utf-8")
        )
    except OSError as error:
        raise ValueError("Project identity manifest could not be read.") from error
    if not identity.source_root.is_absolute() or identity.source_fingerprint != source_fingerprint(
        identity.source_root
    ):
        raise ValueError("Project identity manifest is inconsistent.")


def _validated_manifest(archive: zipfile.ZipFile) -> ProjectArchiveManifest:
    infos = archive.infolist()
    if len(infos) > MAX_ARCHIVE_ENTRIES:
        raise ProjectArchiveError("Could not restore project archive: too many entries.")
    names = [info.filename for info in infos]
    if len(names) != len(set(names)):
        raise ProjectArchiveError(
            "Could not restore project archive: duplicate entry names are unsafe."
        )
    if "backup_manifest.json" not in names:
        raise ProjectArchiveError(
            "Could not restore project archive: typed backup manifest is missing."
        )
    manifest_info = archive.getinfo("backup_manifest.json")
    if manifest_info.file_size > MAX_ARCHIVE_MANIFEST_BYTES:
        raise ProjectArchiveError(
            "Could not restore project archive: typed manifest is unreasonably large."
        )
    for info in infos:
        _validate_zip_entry_type(info)
    try:
        manifest = ProjectArchiveManifest.model_validate_json(archive.read("backup_manifest.json"))
    except ValidationError as error:
        raise ProjectArchiveError(
            "Could not restore project archive: manifest is malformed or uses an "
            "unsupported schema."
        ) from error
    expected = {item.path for item in manifest.files}
    actual = {name for name in names if name != "backup_manifest.json"}
    if actual != expected:
        raise ProjectArchiveError(
            "Could not restore project archive: entries do not match the typed manifest."
        )
    return manifest


def _extract_validated(
    archive: zipfile.ZipFile,
    manifest: ProjectArchiveManifest,
    destination: Path,
) -> None:
    total_size = 0
    destination.mkdir(parents=True, exist_ok=False)
    for item in manifest.files:
        name = item.path
        info = archive.getinfo(name)
        archive_path = PurePosixPath(name)
        expected_size = item.size_bytes
        total_size += expected_size
        if expected_size != info.file_size or total_size > MAX_RESTORE_BYTES:
            raise ProjectArchiveError(
                "Could not restore project archive: size contract is invalid."
            )
        target = destination.joinpath(*archive_path.parts).resolve()
        if not target.is_relative_to(destination.resolve()):
            raise ProjectArchiveError(
                "Could not restore project archive: a target escapes the workspace."
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        with archive.open(info) as source, target.open("wb") as output:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
                output.write(chunk)
        if digest.hexdigest() != item.sha256:
            raise ProjectArchiveError(
                f"Could not restore project archive: checksum failed for {item.path}."
            )


def _validate_zip_entry_type(info: zipfile.ZipInfo) -> None:
    if info.flag_bits & 0x1:
        raise ProjectArchiveError(
            "Could not restore project archive: encrypted entries are not supported."
        )
    mode = info.external_attr >> 16
    file_type = stat.S_IFMT(mode)
    if stat.S_ISLNK(mode) or file_type not in {0, stat.S_IFREG}:
        raise ProjectArchiveError(
            "Could not restore project archive: entries must be regular files, not links "
            "or special files."
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
