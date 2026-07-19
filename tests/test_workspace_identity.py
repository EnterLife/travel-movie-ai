from datetime import UTC, datetime
from pathlib import Path

import pytest

from travelmovieai.application.context import ProjectContext
from travelmovieai.application.service import TravelMovieService
from travelmovieai.application.workspace_identity import (
    WORKSPACE_IDENTITY_FILENAME,
    ProjectWorkspaceIdentity,
)
from travelmovieai.core.config import Settings
from travelmovieai.core.exceptions import WorkspaceIdentityError
from travelmovieai.domain.models import MediaScanReport


def test_default_workspace_is_deterministic_and_isolates_equal_basenames(
    tmp_path: Path,
) -> None:
    first_source = tmp_path / "first" / "Trip"
    second_source = tmp_path / "second" / "Trip"
    first_source.mkdir(parents=True)
    second_source.mkdir(parents=True)
    service = TravelMovieService(Settings(workspace=tmp_path / "projects"))

    first_workspace = service.resolve_workspace(first_source, None)
    second_workspace = service.resolve_workspace(second_source, None)

    assert first_workspace == service.resolve_workspace(first_source, None)
    assert first_workspace != second_workspace
    assert first_workspace.name.startswith("Trip-")
    assert second_workspace.name.startswith("Trip-")


def test_matching_legacy_analysis_is_reused_and_migrated_to_identity(
    tmp_path: Path,
) -> None:
    source = tmp_path / "media" / "Trip"
    source.mkdir(parents=True)
    workspace_root = tmp_path / "projects"
    legacy_workspace = workspace_root / source.name
    _write_analysis(legacy_workspace, source)
    service = TravelMovieService(Settings(workspace=workspace_root))

    project_paths = service.resolve_project_paths(source, None)
    context = ProjectContext(
        input_path=project_paths.input_path,
        workspace=project_paths.workspace,
        settings=service.settings,
    )
    context.prepare()

    identity_path = legacy_workspace / WORKSPACE_IDENTITY_FILENAME
    identity = ProjectWorkspaceIdentity.model_validate_json(
        identity_path.read_text(encoding="utf-8")
    )
    assert project_paths.workspace == legacy_workspace.resolve()
    assert identity.source_root == source.resolve()
    assert (legacy_workspace / "artifacts" / "analysis.json").is_file()
    assert list(legacy_workspace.glob(f".{WORKSPACE_IDENTITY_FILENAME}.*.tmp")) == []


def test_legacy_workspace_is_not_reused_for_same_basename_from_another_root(
    tmp_path: Path,
) -> None:
    first_source = tmp_path / "first" / "Trip"
    second_source = tmp_path / "second" / "Trip"
    first_source.mkdir(parents=True)
    second_source.mkdir(parents=True)
    workspace_root = tmp_path / "projects"
    legacy_workspace = workspace_root / "Trip"
    _write_analysis(legacy_workspace, first_source)
    service = TravelMovieService(Settings(workspace=workspace_root))

    assert service.resolve_workspace(first_source, None) == legacy_workspace.resolve()
    assert service.resolve_workspace(second_source, None) != legacy_workspace.resolve()


def test_explicit_workspace_reuse_for_another_source_is_rejected_before_pipeline_work(
    tmp_path: Path,
) -> None:
    first_source = tmp_path / "first"
    second_source = tmp_path / "second"
    first_source.mkdir()
    second_source.mkdir()
    workspace = tmp_path / "workspace"
    first_context = ProjectContext(
        input_path=first_source,
        workspace=workspace,
        settings=Settings(),
    )
    first_context.prepare()
    sentinel = first_context.cache_dir / "keep.bin"
    sentinel.write_bytes(b"private cache")
    identity_before = (workspace / WORKSPACE_IDENTITY_FILENAME).read_bytes()
    service = TravelMovieService(Settings())

    with pytest.raises(WorkspaceIdentityError, match="different source"):
        service.analyze(input_path=second_source, workspace=workspace)

    assert sentinel.read_bytes() == b"private cache"
    assert (workspace / WORKSPACE_IDENTITY_FILENAME).read_bytes() == identity_before


def test_nonempty_unidentified_explicit_workspace_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "media"
    source.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    existing = workspace / "project.db"
    existing.write_bytes(b"unidentified project")
    service = TravelMovieService(Settings())

    with pytest.raises(WorkspaceIdentityError, match="no valid identity"):
        service.resolve_project_paths(source, workspace)

    assert existing.read_bytes() == b"unidentified project"
    assert not (workspace / WORKSPACE_IDENTITY_FILENAME).exists()


def _write_analysis(workspace: Path, source: Path) -> None:
    artifacts = workspace / "artifacts"
    artifacts.mkdir(parents=True)
    report = MediaScanReport(
        input_path=source.resolve(),
        scanned_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    (artifacts / "analysis.json").write_text(
        report.model_dump_json(indent=2),
        encoding="utf-8",
    )
