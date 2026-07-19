from pathlib import Path

import pytest

from travelmovieai.application.validation import validate_output_path, validate_project_paths
from travelmovieai.core.exceptions import InvalidProjectPathError


def test_project_paths_reject_workspace_parent_of_source(tmp_path: Path) -> None:
    media = tmp_path / "media"
    media.mkdir()

    with pytest.raises(InvalidProjectPathError, match="cannot be the same or nested"):
        validate_project_paths(media, tmp_path)


def test_project_paths_reject_workspace_inside_source(tmp_path: Path) -> None:
    media = tmp_path / "media"
    media.mkdir()

    with pytest.raises(InvalidProjectPathError, match="cannot be the same or nested"):
        validate_project_paths(media, media / "workspace")


def test_project_paths_allow_separate_workspace(tmp_path: Path) -> None:
    media = tmp_path / "media"
    workspace = tmp_path / "workspace"
    media.mkdir()

    paths = validate_project_paths(media, workspace)

    assert paths.input_path == media.resolve()
    assert paths.workspace == workspace.resolve()


def test_movie_output_must_be_outside_source_folder(tmp_path: Path) -> None:
    media = tmp_path / "media"
    media.mkdir()

    with pytest.raises(InvalidProjectPathError, match="outside the source media folder"):
        validate_output_path(media / "rendered.mp4", media)

    output = validate_output_path(tmp_path / "output" / "rendered.mp4", media)
    assert output == (tmp_path / "output" / "rendered.mp4").resolve()


def test_movie_output_rejects_non_mp4_and_workspace_runtime_targets(tmp_path: Path) -> None:
    media = tmp_path / "media"
    workspace = tmp_path / "workspace"
    media.mkdir()

    with pytest.raises(InvalidProjectPathError, match=".mp4 extension"):
        validate_output_path(tmp_path / "movie.avi", media)
    with pytest.raises(InvalidProjectPathError, match="project database"):
        validate_output_path(
            workspace / "project.mp4",
            media,
            workspace=workspace,
            database_path=workspace / "project.mp4",
        )
    with pytest.raises(InvalidProjectPathError, match="cache and frame folders"):
        validate_output_path(
            workspace / "cache" / "movie.mp4",
            media,
            workspace=workspace,
        )

    output = validate_output_path(
        workspace / "artifacts" / "final.mp4",
        media,
        workspace=workspace,
    )
    assert output == (workspace / "artifacts" / "final.mp4").resolve()
