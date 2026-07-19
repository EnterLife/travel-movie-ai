import os
import time
from pathlib import Path

import pytest

from travelmovieai.application.cache import cleanup_cache_roots, cleanup_context_cache
from travelmovieai.application.context import ProjectContext
from travelmovieai.core.config import Settings
from travelmovieai.domain.enums import PipelineStage
from travelmovieai.domain.models import StageResult
from travelmovieai.pipeline.base import Stage
from travelmovieai.pipeline.runner import PipelineRunner


def test_cache_cleanup_removes_oldest_files_until_target(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    frames = tmp_path / "frames"
    cache.mkdir()
    frames.mkdir()
    oldest = cache / "old.bin"
    middle = frames / "middle.bin"
    newest = cache / "new.bin"
    for path in (oldest, middle, newest):
        path.write_bytes(b"x" * 10)
    now = time.time_ns()
    os.utime(oldest, ns=(now - 3_000_000_000, now - 3_000_000_000))
    os.utime(middle, ns=(now - 2_000_000_000, now - 2_000_000_000))
    os.utime(newest, ns=(now - 1_000_000_000, now - 1_000_000_000))

    result = cleanup_cache_roots(
        [cache, frames],
        limit_bytes=20,
        target_ratio=0.5,
    )

    assert result.before_bytes == 30
    assert result.after_bytes == 10
    assert result.removed_files == 2
    assert not oldest.exists()
    assert not middle.exists()
    assert newest.is_file()


def test_zero_cache_limit_disables_cleanup(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    artifact = cache / "keep.bin"
    artifact.write_bytes(b"cache")

    result = cleanup_cache_roots([cache], limit_bytes=0)

    assert result.removed_files == 0
    assert artifact.is_file()


def test_context_cache_cleanup_uses_typed_mebibyte_limit(tmp_path: Path) -> None:
    context = ProjectContext(
        input_path=tmp_path,
        workspace=tmp_path / "workspace",
        settings=Settings(project_cache_limit_mb=1, project_cache_target_ratio=0.5),
    )
    context.prepare()
    first = context.cache_dir / "first.bin"
    second = context.frames_dir / "second.bin"
    first.write_bytes(b"x" * 700_000)
    second.write_bytes(b"y" * 700_000)
    now = time.time_ns()
    os.utime(first, ns=(now - 2_000_000_000, now - 2_000_000_000))
    os.utime(second, ns=(now - 1_000_000_000, now - 1_000_000_000))

    result = cleanup_context_cache(context)

    assert result.limit_bytes == 1024 * 1024
    assert result.after_bytes <= result.target_bytes
    assert not first.exists()


def test_cache_cleanup_does_not_follow_external_symlink(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    external = tmp_path / "private.bin"
    external.write_bytes(b"private")
    link = cache / "linked.bin"
    try:
        link.symlink_to(external)
    except OSError:
        pytest.skip("File symlinks are unavailable in this Windows environment")
    local = cache / "local.bin"
    local.write_bytes(b"x" * 20)

    cleanup_cache_roots([cache], limit_bytes=10, target_ratio=0.5)

    assert external.read_bytes() == b"private"
    assert link.is_symlink()


def test_cache_cleanup_rejects_invalid_target_ratio(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="target ratio"):
        cleanup_cache_roots([tmp_path], limit_bytes=10, target_ratio=0)


def test_pipeline_runner_enforces_cache_limit_before_stage_work(tmp_path: Path) -> None:
    context = ProjectContext(
        input_path=tmp_path,
        workspace=tmp_path / "workspace",
        settings=Settings(project_cache_limit_mb=1, project_cache_target_ratio=0.5),
    )
    context.prepare()
    stale = context.cache_dir / "stale.bin"
    stale.write_bytes(b"x" * 1_200_000)

    class AssertCleanStage(Stage):
        name = PipelineStage.MEDIA_SCAN

        def run(self, context: ProjectContext) -> StageResult:
            assert not stale.exists()
            return StageResult(stage=self.name)

    PipelineRunner([AssertCleanStage()]).run_until(context, PipelineStage.MEDIA_SCAN)
