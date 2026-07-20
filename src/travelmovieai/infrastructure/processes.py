"""Cross-platform helpers for cancellable child process trees."""

from __future__ import annotations

import ctypes
import os
import signal
import subprocess
from collections.abc import Callable, Sequence
from contextlib import suppress
from ctypes import wintypes
from functools import lru_cache
from typing import Any

_PROCESS_JOB_ATTRIBUTE = "_travelmovieai_windows_job"
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_TH32CS_SNAPTHREAD = 0x00000004
_THREAD_SUSPEND_RESUME = 0x0002
_ERROR_NO_MORE_FILES = 18
_INVALID_DWORD = 0xFFFFFFFF
_INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value


class _JobObjectBasicLimitInformation(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _IoCounters(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class _JobObjectExtendedLimitInformation(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JobObjectBasicLimitInformation),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _ThreadEntry32(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ThreadID", wintypes.DWORD),
        ("th32OwnerProcessID", wintypes.DWORD),
        ("tpBasePri", wintypes.LONG),
        ("tpDeltaPri", wintypes.LONG),
        ("dwFlags", wintypes.DWORD),
    ]


class _WindowsKillJob:
    def __init__(self, api: Any, handle: int) -> None:
        self._api = api
        self._handle: int | None = handle

    def terminate(self) -> bool:
        return bool(self._handle is not None and self._api.TerminateJobObject(self._handle, 1))

    def close(self) -> None:
        handle = self._handle
        if handle is None:
            return
        if not self._api.CloseHandle(handle):
            error_code = ctypes.get_last_error()
            raise OSError(error_code, "Could not close the Windows process Job Object.")
        self._handle = None


def process_group_popen_kwargs() -> dict[str, Any]:
    """Legacy process-group options for short-lived direct ``Popen`` callers."""

    if os.name == "nt":
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
    return {"start_new_session": True}


def start_process(
    command: Sequence[str],
    *,
    popen_factory: Callable[..., subprocess.Popen[str]] | None = None,
    **kwargs: Any,
) -> subprocess.Popen[str]:
    """Start a long-running process inside an OS-owned termination boundary."""

    factory = popen_factory or subprocess.Popen
    options = dict(kwargs)
    if os.name != "nt":
        options.setdefault("start_new_session", True)
        return factory(command, **options)

    creation_flags = int(options.pop("creationflags", 0))
    creation_flags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
    creation_flags |= getattr(subprocess, "CREATE_SUSPENDED", 0x00000004)
    options["creationflags"] = creation_flags
    process = factory(command, **options)
    process_handle = getattr(process, "_handle", None)
    if process_handle is None:
        # Test doubles do not own a native Windows process and need no Job Object.
        return process

    job: _WindowsKillJob | None = None
    try:
        job = _create_windows_kill_job(int(process_handle))
        setattr(process, _PROCESS_JOB_ATTRIBUTE, job)
        _resume_windows_process(process.pid)
    except BaseException:
        if job is not None:
            with suppress(OSError):
                job.close()
        if process.poll() is None:
            with suppress(OSError):
                process.kill()
        with suppress(OSError, subprocess.TimeoutExpired):
            process.wait(timeout=5)
        raise
    return process


def release_process_resources(process: subprocess.Popen[str]) -> None:
    """Release a process Job handle after normal completion, killing stray descendants."""

    job = getattr(process, _PROCESS_JOB_ATTRIBUTE, None)
    if not isinstance(job, _WindowsKillJob):
        return
    job.close()
    setattr(process, _PROCESS_JOB_ATTRIBUTE, None)


def terminate_process_tree(process: subprocess.Popen[str]) -> None:
    """Terminate a process and every descendant started inside its boundary."""

    if process.poll() is not None:
        release_process_resources(process)
        return
    if os.name == "nt":
        job = getattr(process, _PROCESS_JOB_ATTRIBUTE, None)
        tree_stopped = isinstance(job, _WindowsKillJob) and job.terminate()
        if not tree_stopped:
            tree_stopped = _terminate_windows_tree_with_taskkill(process.pid)
        if not tree_stopped and process.poll() is None:
            process.terminate()
    else:
        try:
            _kill_posix_process_group(process.pid, signal.SIGTERM)
        except OSError:
            process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            try:
                _kill_posix_process_group(
                    process.pid,
                    getattr(signal, "SIGKILL", signal.SIGTERM),
                )
            except OSError:
                process.kill()
        else:
            process.kill()
        process.wait(timeout=5)
    finally:
        release_process_resources(process)


def _terminate_windows_tree_with_taskkill(pid: int) -> bool:
    command = ["taskkill", "/PID", str(pid), "/T", "/F"]
    for _ in range(2):
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                check=False,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if completed.returncode == 0:
            return True
    return False


@lru_cache(maxsize=1)
def _windows_api() -> Any:
    loader = getattr(ctypes, "WinDLL", None)
    if loader is None:
        raise OSError("Windows process Job APIs are unavailable.")
    api: Any = loader("kernel32", use_last_error=True)
    api.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    api.CreateJobObjectW.restype = wintypes.HANDLE
    api.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    api.SetInformationJobObject.restype = wintypes.BOOL
    api.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    api.AssignProcessToJobObject.restype = wintypes.BOOL
    api.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    api.TerminateJobObject.restype = wintypes.BOOL
    api.CloseHandle.argtypes = [wintypes.HANDLE]
    api.CloseHandle.restype = wintypes.BOOL
    api.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    api.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    api.Thread32First.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ThreadEntry32)]
    api.Thread32First.restype = wintypes.BOOL
    api.Thread32Next.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ThreadEntry32)]
    api.Thread32Next.restype = wintypes.BOOL
    api.OpenThread.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    api.OpenThread.restype = wintypes.HANDLE
    api.ResumeThread.argtypes = [wintypes.HANDLE]
    api.ResumeThread.restype = wintypes.DWORD
    return api


def _create_windows_kill_job(process_handle: int) -> _WindowsKillJob:
    api = _windows_api()
    handle = api.CreateJobObjectW(None, None)
    if not handle:
        _raise_windows_api_error("Could not create a Windows process Job Object")
    job = _WindowsKillJob(api, int(handle))
    limits = _JobObjectExtendedLimitInformation()
    limits.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not api.SetInformationJobObject(
        handle,
        _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
        ctypes.byref(limits),
        ctypes.sizeof(limits),
    ):
        with suppress(OSError):
            job.close()
        _raise_windows_api_error("Could not configure a Windows process Job Object")
    if not api.AssignProcessToJobObject(handle, process_handle):
        with suppress(OSError):
            job.close()
        _raise_windows_api_error("Could not assign a process to its Windows Job Object")
    return job


def _resume_windows_process(pid: int) -> None:
    api = _windows_api()
    snapshot = api.CreateToolhelp32Snapshot(_TH32CS_SNAPTHREAD, 0)
    if not snapshot or int(snapshot) == _INVALID_HANDLE_VALUE:
        _raise_windows_api_error("Could not inspect the suspended Windows process")
    try:
        entry = _ThreadEntry32()
        entry.dwSize = ctypes.sizeof(entry)
        if not api.Thread32First(snapshot, ctypes.byref(entry)):
            _raise_windows_api_error("Could not inspect the suspended Windows process")
        while True:
            if entry.th32OwnerProcessID == pid:
                thread = api.OpenThread(_THREAD_SUSPEND_RESUME, False, entry.th32ThreadID)
                if not thread:
                    _raise_windows_api_error("Could not open the suspended Windows process thread")
                try:
                    if api.ResumeThread(thread) == _INVALID_DWORD:
                        _raise_windows_api_error("Could not resume the Windows process thread")
                finally:
                    api.CloseHandle(thread)
                return
            if not api.Thread32Next(snapshot, ctypes.byref(entry)):
                error_code = ctypes.get_last_error()
                if error_code != _ERROR_NO_MORE_FILES:
                    _raise_windows_api_error("Could not inspect the suspended Windows process")
                break
    finally:
        api.CloseHandle(snapshot)
    raise OSError(f"Could not find the primary thread for suspended process PID {pid}.")


def _raise_windows_api_error(message: str) -> None:
    error_code = ctypes.get_last_error()
    raise OSError(error_code, f"{message} (Windows error {error_code}).")


def _kill_posix_process_group(pid: int, signal_number: int) -> None:
    get_process_group = getattr(os, "getpgid", None)
    kill_process_group = getattr(os, "killpg", None)
    if not callable(get_process_group) or not callable(kill_process_group):
        raise OSError("POSIX process-group operations are unavailable")
    kill_process_group(get_process_group(pid), signal_number)
