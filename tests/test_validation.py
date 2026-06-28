from pathlib import Path

import pytest

from travelmovieai.application.validation import validate_project_paths
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
