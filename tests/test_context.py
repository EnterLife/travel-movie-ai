from pathlib import Path

from travelmovieai.application.context import ProjectContext
from travelmovieai.core.config import Settings


def test_prepare_creates_runtime_directories(tmp_path: Path) -> None:
    context = ProjectContext(
        input_path=tmp_path / "media",
        workspace=tmp_path / "workspace",
        settings=Settings(),
    )

    context.prepare()

    assert context.frames_dir.is_dir()
    assert context.cache_dir.is_dir()
    assert context.artifacts_dir.is_dir()
