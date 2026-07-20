import json
import multiprocessing
import os
import threading
from pathlib import Path
from typing import Any

import pytest

from travelmovieai.application.context import ProjectContext
from travelmovieai.application.project_archive import restore_project_archive
from travelmovieai.application.service import TravelMovieService
from travelmovieai.application.workspace_lease import (
    WORKSPACE_LEASE_METADATA_FILENAME,
    WorkspaceLease,
    WorkspaceTargetLease,
    workspace_target_lease_paths,
)
from travelmovieai.core.config import Settings
from travelmovieai.core.exceptions import WorkspaceBusyError
from travelmovieai.domain.enums import PipelineStage
from travelmovieai.domain.models import QuickMontageSettings, StageResult
from travelmovieai.pipeline.base import Stage
from travelmovieai.pipeline.runner import PipelineRunner


def _hold_workspace_lease(
    workspace: Path,
    ready: Any,
    release: Any,
) -> None:
    with WorkspaceLease(workspace, operation="process-test"):
        ready.set()
        release.wait(30)


def _hold_workspace_target_lease(
    workspace: Path,
    ready: Any,
    release: Any,
) -> None:
    with WorkspaceTargetLease(workspace, operation="restore:first"):
        ready.set()
        release.wait(30)


def test_workspace_lease_is_reentrant_only_in_owner_thread(tmp_path: Path) -> None:
    workspace = tmp_path / "рабочая папка"
    thread_errors: list[BaseException] = []

    with WorkspaceLease(workspace, operation="outer"):
        with WorkspaceLease(workspace, operation="nested"):
            pass

        def compete() -> None:
            try:
                with WorkspaceLease(workspace, operation="thread-competitor"):
                    pass
            except BaseException as error:
                thread_errors.append(error)

        thread = threading.Thread(target=compete)
        thread.start()
        thread.join(timeout=5)

    assert not thread.is_alive()
    assert len(thread_errors) == 1
    assert isinstance(thread_errors[0], WorkspaceBusyError)


def test_workspace_lease_blocks_another_process_and_reports_owner(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    workspace = tmp_path / "workspace with spaces"
    process = context.Process(
        target=_hold_workspace_lease,
        args=(workspace, ready, release),
    )
    process.start()
    try:
        assert ready.wait(timeout=15)
        with (
            pytest.raises(WorkspaceBusyError) as captured,
            WorkspaceLease(workspace, operation="competitor"),
        ):
            pass

        message = str(captured.value)
        assert f"PID {process.pid}" in message
        assert "operation process-test" in message
        metadata = json.loads(
            (workspace / WORKSPACE_LEASE_METADATA_FILENAME).read_text(encoding="utf-8")
        )
        assert metadata["pid"] == process.pid
        assert metadata["started_at"]
    finally:
        release.set()
        process.join(timeout=15)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
    assert process.exitcode == 0


def test_workspace_lease_recovers_after_owner_process_crashes(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    workspace = tmp_path / "workspace"
    process = context.Process(
        target=_hold_workspace_lease,
        args=(workspace, ready, release),
    )
    process.start()
    assert ready.wait(timeout=15)
    process.terminate()
    process.join(timeout=15)
    assert not process.is_alive()

    with WorkspaceLease(workspace, operation="recovered"):
        metadata = json.loads(
            (workspace / WORKSPACE_LEASE_METADATA_FILENAME).read_text(encoding="utf-8")
        )
        assert metadata["pid"] == os.getpid()
        assert metadata["operation"] == "recovered"


def test_target_sidecar_serializes_restore_and_workspace_operations_without_touching_target(
    tmp_path: Path,
) -> None:
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    workspace = tmp_path / "absent restore target"
    archive = tmp_path / "backup.zip"
    archive.write_bytes(b"validation must not start while the target is leased")
    process = context.Process(
        target=_hold_workspace_target_lease,
        args=(workspace, ready, release),
    )
    process.start()
    try:
        assert ready.wait(timeout=15)
        lock_path, metadata_path = workspace_target_lease_paths(workspace)
        assert lock_path.is_file()
        assert metadata_path.is_file()
        assert lock_path.parent == workspace.parent
        assert not lock_path.is_relative_to(workspace)
        assert not workspace.exists()

        with pytest.raises(WorkspaceBusyError, match="operation restore:first"):
            restore_project_archive(archive, workspace)
        with (
            pytest.raises(WorkspaceBusyError, match="operation restore:first"),
            WorkspaceLease(workspace, operation="pipeline"),
        ):
            pass
        assert not workspace.exists()
    finally:
        release.set()
        process.join(timeout=15)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
    assert process.exitcode == 0


def test_restore_is_blocked_by_running_workspace_operation_before_target_validation(
    tmp_path: Path,
) -> None:
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    workspace = tmp_path / "workspace"
    archive = tmp_path / "backup.zip"
    archive.write_bytes(b"validation must wait for the running pipeline")
    process = context.Process(
        target=_hold_workspace_lease,
        args=(workspace, ready, release),
    )
    process.start()
    try:
        assert ready.wait(timeout=15)
        assert workspace.is_dir()
        with pytest.raises(WorkspaceBusyError, match="operation process-test"):
            restore_project_archive(archive, workspace)
    finally:
        release.set()
        process.join(timeout=15)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
    assert process.exitcode == 0


def test_pipeline_runner_holds_process_lease_and_ignores_lock_for_identity(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "media"
    input_path.mkdir()
    workspace = tmp_path / "workspace"
    context = ProjectContext(
        input_path=input_path,
        workspace=workspace,
        settings=Settings(),
    )
    process_context = multiprocessing.get_context("spawn")
    ready = process_context.Event()
    release = process_context.Event()
    process = process_context.Process(
        target=_hold_workspace_lease,
        args=(workspace, ready, release),
    )
    process.start()
    try:
        assert ready.wait(timeout=15)
        with pytest.raises(WorkspaceBusyError):
            PipelineRunner([_SuccessfulStage()]).run_until(
                context,
                PipelineStage.MEDIA_SCAN,
            )
        with pytest.raises(WorkspaceBusyError):
            TravelMovieService(Settings()).create_quick_montage(
                input_path=input_path,
                workspace=workspace,
                settings=QuickMontageSettings(),
            )
    finally:
        release.set()
        process.join(timeout=15)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)

    result = PipelineRunner([_SuccessfulStage()]).run_until(
        context,
        PipelineStage.MEDIA_SCAN,
    )

    assert result.stage is PipelineStage.MEDIA_SCAN
    assert (workspace / ".travelmovieai-project.json").is_file()


class _SuccessfulStage(Stage):
    name = PipelineStage.MEDIA_SCAN

    def run(self, context: ProjectContext) -> StageResult:
        assert context.workspace.is_dir()
        return StageResult(stage=self.name)
