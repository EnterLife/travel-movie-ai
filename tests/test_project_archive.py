import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from travelmovieai.application.context import ProjectContext
from travelmovieai.application.project_archive import (
    export_project_archive,
    restore_project_archive,
)
from travelmovieai.application.workspace_identity import WORKSPACE_IDENTITY_FILENAME
from travelmovieai.core.config import Settings
from travelmovieai.core.exceptions import (
    InvalidProjectPathError,
    PipelineStageError,
    ProjectArchiveError,
)


def test_project_archive_round_trip_excludes_large_media_by_default(tmp_path: Path) -> None:
    context = _context(tmp_path)
    identity_payload = (context.workspace / WORKSPACE_IDENTITY_FILENAME).read_text(encoding="utf-8")
    metadata = context.artifacts_dir / "analysis.json"
    movie = context.artifacts_dir / "final.mp4"
    metadata.write_text('{"count": 1}', encoding="utf-8")
    movie.write_bytes(b"rendered movie")
    archive_path = tmp_path / "backup.zip"

    result = export_project_archive(context, archive_path)
    restored = restore_project_archive(archive_path, tmp_path / "restored")

    assert result.archive_path == archive_path.resolve()
    assert result.includes_rendered_media is False
    assert (restored / context.settings.database_filename).is_file()
    assert (restored / WORKSPACE_IDENTITY_FILENAME).read_text(encoding="utf-8") == identity_payload
    assert (restored / "artifacts" / "analysis.json").read_text(encoding="utf-8") == (
        '{"count": 1}'
    )
    assert not (restored / ".travelmovieai.lock").exists()
    assert not (restored / ".travelmovieai.lock.json").exists()
    assert not (restored / "artifacts" / "final.mp4").exists()
    with zipfile.ZipFile(archive_path) as archive:
        assert WORKSPACE_IDENTITY_FILENAME in archive.namelist()
        manifest = json.loads(archive.read("backup_manifest.json"))
        assert str(context.input_path.resolve()) not in json.dumps(manifest)


def test_project_archive_can_include_rendered_media_explicitly(tmp_path: Path) -> None:
    context = _context(tmp_path)
    movie = context.artifacts_dir / "final.mp4"
    movie.write_bytes(b"rendered movie")

    result = export_project_archive(
        context,
        tmp_path / "full-backup.zip",
        include_rendered_media=True,
    )

    with zipfile.ZipFile(result.archive_path) as archive:
        assert "artifacts/final.mp4" in archive.namelist()


def test_project_archive_rejects_output_inside_workspace(tmp_path: Path) -> None:
    context = _context(tmp_path)

    with pytest.raises(InvalidProjectPathError, match="outside"):
        export_project_archive(context, context.workspace / "backup.zip")


def test_project_archive_requires_explicit_overwrite(tmp_path: Path) -> None:
    context = _context(tmp_path)
    archive_path = tmp_path / "backup.zip"
    archive_path.write_bytes(b"existing")

    with pytest.raises(InvalidProjectPathError, match="overwrite explicitly"):
        export_project_archive(context, archive_path)

    result = export_project_archive(context, archive_path, overwrite=True)
    assert zipfile.is_zipfile(result.archive_path)


def test_restore_rejects_path_traversal_even_when_manifest_matches(tmp_path: Path) -> None:
    archive_path = tmp_path / "malicious.zip"
    payload = b"escape"
    manifest = {
        "schema_version": 1,
        "created_at": "2026-01-01T00:00:00+00:00",
        "project_name": "project",
        "includes_rendered_media": False,
        "files": [
            {
                "path": "../escape.txt",
                "size_bytes": len(payload),
                "sha256": ("b93f9d65f0b53f2f7c48e85ed3c5ed7c55856c4cbbd0f10b7c37d7909a16e38a"),
            }
        ],
    }
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("backup_manifest.json", json.dumps(manifest))
        archive.writestr("../escape.txt", payload)

    with pytest.raises(PipelineStageError, match="restore"):
        restore_project_archive(archive_path, tmp_path / "restored")

    assert not (tmp_path / "escape.txt").exists()


def test_restore_rejects_inconsistent_project_identity(tmp_path: Path) -> None:
    archive_path = tmp_path / "invalid-identity.zip"
    identity = {
        "schema_version": 1,
        "source_root": str((tmp_path / "source").resolve()),
        "source_fingerprint": "0" * 64,
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    payload = json.dumps(identity).encode("utf-8")
    manifest = {
        "schema_version": 1,
        "created_at": "2026-01-01T00:00:00+00:00",
        "project_name": "project",
        "includes_rendered_media": False,
        "files": [
            {
                "path": WORKSPACE_IDENTITY_FILENAME,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        ],
    }
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("backup_manifest.json", json.dumps(manifest))
        archive.writestr(WORKSPACE_IDENTITY_FILENAME, payload)

    restored = tmp_path / "restored"
    with pytest.raises(PipelineStageError, match="restore"):
        restore_project_archive(archive_path, restored)

    assert not restored.exists()


def test_restore_rejects_malformed_typed_manifest_fields(tmp_path: Path) -> None:
    archive_path = tmp_path / "malformed-manifest.zip"
    manifest = {
        "schema_version": 1,
        "created_at": "2026-01-01T00:00:00+00:00",
        "project_name": "project",
        "includes_rendered_media": False,
        "files": [{"path": "project.db", "size_bytes": "not-an-integer"}],
    }
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("backup_manifest.json", json.dumps(manifest))
        archive.writestr("project.db", b"database")

    with pytest.raises(ProjectArchiveError, match="manifest is malformed"):
        restore_project_archive(archive_path, tmp_path / "restored")


def test_restore_rejects_duplicate_zip_member_names(tmp_path: Path) -> None:
    archive_path = tmp_path / "duplicate-members.zip"
    payload = b"database"
    manifest = {
        "schema_version": 1,
        "created_at": "2026-01-01T00:00:00+00:00",
        "project_name": "project",
        "includes_rendered_media": False,
        "files": [
            {
                "path": "project.db",
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        ],
    }
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("backup_manifest.json", json.dumps(manifest))
        archive.writestr("project.db", payload)
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("project.db", payload)

    with pytest.raises(ProjectArchiveError, match="duplicate entry"):
        restore_project_archive(archive_path, tmp_path / "restored")


def _context(tmp_path: Path) -> ProjectContext:
    input_path = tmp_path / "input"
    input_path.mkdir()
    context = ProjectContext(
        input_path=input_path,
        workspace=tmp_path / "workspace",
        settings=Settings(),
    )
    context.prepare()
    return context
