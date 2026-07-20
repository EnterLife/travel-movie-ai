import ctypes
import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

from travelmovieai.infrastructure import processes


class _FakeProcess:
    pid = 4321

    def __init__(self) -> None:
        self.returncode: int | None = None
        self.terminate_calls = 0
        self.kill_calls = 0

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.returncode = -1

    def wait(self, timeout: float) -> int:
        del timeout
        return self.returncode or 0

    def kill(self) -> None:
        self.kill_calls += 1
        self.returncode = -9


class _FakeWindowsApi:
    def __init__(self) -> None:
        self.closed: list[int] = []
        self.terminated: list[tuple[int, int]] = []

    def CloseHandle(self, handle: int) -> bool:
        self.closed.append(handle)
        return True

    def TerminateJobObject(self, handle: int, exit_code: int) -> bool:
        self.terminated.append((handle, exit_code))
        return True


def test_windows_start_uses_suspended_job_then_releases_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess()
    process._handle = 123  # type: ignore[attr-defined]
    options: dict[str, object] = {}
    api = _FakeWindowsApi()
    job = processes._WindowsKillJob(api, 456)
    resumed: list[int] = []

    def factory(command: object, **kwargs: object) -> object:
        del command
        options.update(kwargs)
        return process

    monkeypatch.setattr(processes.os, "name", "nt")
    monkeypatch.setattr(processes, "_create_windows_kill_job", lambda _: job)
    monkeypatch.setattr(processes, "_resume_windows_process", resumed.append)

    started = processes.start_process(
        ["example.exe"],
        popen_factory=cast(Callable[..., subprocess.Popen[str]], factory),
        text=True,
    )

    flags = cast(int, options["creationflags"])
    assert flags & getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
    assert flags & getattr(subprocess, "CREATE_SUSPENDED", 0x00000004)
    assert resumed == [process.pid]
    processes.release_process_resources(started)
    assert api.closed == [456]


def test_windows_job_termination_precedes_parent_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess()
    api = _FakeWindowsApi()
    setattr(process, processes._PROCESS_JOB_ATTRIBUTE, processes._WindowsKillJob(api, 789))
    monkeypatch.setattr(processes.os, "name", "nt")
    monkeypatch.setattr(
        processes,
        "_terminate_windows_tree_with_taskkill",
        lambda _: (_ for _ in ()).throw(AssertionError("taskkill fallback was used")),
    )

    processes.terminate_process_tree(cast(subprocess.Popen[str], process))

    assert api.terminated == [(789, 1)]
    assert api.closed == [789]
    assert process.terminate_calls == 0


def test_windows_tree_kill_retries_nonzero_taskkill_then_terminates_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess()
    commands: list[list[str]] = []

    def failed_taskkill(command: list[str], **kwargs: object) -> object:
        del kwargs
        commands.append(command)
        return type("Completed", (), {"returncode": 5})()

    monkeypatch.setattr(processes.os, "name", "nt")
    monkeypatch.setattr(processes.subprocess, "run", failed_taskkill)

    processes.terminate_process_tree(cast(subprocess.Popen[str], process))

    assert len(commands) == 2
    assert all(command[0] == "taskkill" and "/T" in command for command in commands)
    assert process.terminate_calls == 1


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Objects are required")
def test_windows_job_handle_is_released_after_normal_process_exit() -> None:
    process = processes.start_process(
        [sys.executable, "-c", "pass"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )

    assert process.wait(timeout=5) == 0
    assert getattr(process, processes._PROCESS_JOB_ATTRIBUTE, None) is not None

    processes.release_process_resources(process)

    assert getattr(process, processes._PROCESS_JOB_ATTRIBUTE, None) is None
    processes.release_process_resources(process)


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Objects are required")
def test_windows_job_kills_real_parent_and_long_lived_child(tmp_path: Path) -> None:
    child_pid_path = tmp_path / "child.pid"
    parent_script = (
        "import pathlib,subprocess,sys,time; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid),encoding='ascii'); "
        "time.sleep(60)"
    )
    process = processes.start_process(
        [sys.executable, "-c", parent_script, str(child_pid_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        errors="replace",
        text=True,
    )
    try:
        deadline = time.monotonic() + 10
        while not child_pid_path.is_file() and time.monotonic() < deadline:
            if process.poll() is not None:
                stderr = process.stderr.read() if process.stderr is not None else ""
                raise AssertionError(f"parent exited before creating child: {stderr}")
            time.sleep(0.05)
        assert child_pid_path.is_file()
        child_pid = int(child_pid_path.read_text(encoding="ascii"))

        processes.terminate_process_tree(process)

        assert process.poll() is not None
        assert _wait_for_windows_process_exit(child_pid, timeout_seconds=5)
    finally:
        if process.poll() is None:
            processes.terminate_process_tree(process)
        else:
            processes.release_process_resources(process)


def _wait_for_windows_process_exit(pid: int, *, timeout_seconds: float) -> bool:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    kernel32.WaitForSingleObject.restype = ctypes.c_ulong
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_int
    synchronize = 0x00100000
    wait_object_0 = 0
    handle = kernel32.OpenProcess(synchronize, False, pid)
    if not handle:
        return True
    try:
        return kernel32.WaitForSingleObject(handle, round(timeout_seconds * 1000)) == wait_object_0
    finally:
        kernel32.CloseHandle(handle)
