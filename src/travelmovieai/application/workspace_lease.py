"""Cross-process exclusive lease for mutations inside one project workspace."""

from __future__ import annotations

import errno
import hashlib
import importlib
import json
import os
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, BinaryIO, Self

from travelmovieai.core.exceptions import WorkspaceBusyError

WORKSPACE_LEASE_FILENAME = ".travelmovieai.lock"
WORKSPACE_LEASE_METADATA_FILENAME = ".travelmovieai.lock.json"
WORKSPACE_TARGET_LEASE_PREFIX = ".travelmovieai-target-"
_METADATA_LIMIT_BYTES = 8192


@dataclass(slots=True)
class _HeldLease:
    file: BinaryIO
    owner_thread_id: int
    owner_pid: int
    references: int
    metadata: dict[str, Any]


_REGISTRY_LOCK = threading.RLock()
_HELD_LEASES: dict[str, _HeldLease] = {}


class WorkspaceTargetLease:
    """Lock a workspace target without creating or writing inside that target."""

    def __init__(self, workspace: Path, *, operation: str) -> None:
        self.workspace = workspace.expanduser().resolve()
        self.operation = operation.strip() or "workspace-operation"
        self.path, self.metadata_path = workspace_target_lease_paths(self.workspace)
        self._key: str | None = None
        self._entered = False

    def __enter__(self) -> Self:
        if self._entered:
            raise RuntimeError("A workspace target lease instance cannot be entered twice.")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._key = _acquire_file_lease(self.path, self.metadata_path, self.operation)
        self._entered = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        if not self._entered or self._key is None:
            return
        try:
            _release_file_lease(self._key)
        finally:
            self._entered = False
            self._key = None


class WorkspaceLease:
    """Hold an OS-released, non-blocking exclusive workspace lock.

    An adjacent target lease coordinates with atomic restore before this class
    creates the workspace. Lock files deliberately remain after release. The
    operating-system locks, not file existence, establish ownership, so a crashed
    process cannot leave the workspace permanently locked.
    """

    def __init__(self, workspace: Path, *, operation: str) -> None:
        self.workspace = workspace.expanduser().resolve()
        self.operation = operation.strip() or "pipeline"
        self.path = self.workspace / WORKSPACE_LEASE_FILENAME
        self.metadata_path = self.workspace / WORKSPACE_LEASE_METADATA_FILENAME
        self._key: str | None = None
        self._entered = False
        self._target_lease: WorkspaceTargetLease | None = None

    def __enter__(self) -> Self:
        if self._entered:
            raise RuntimeError("A workspace lease instance cannot be entered twice.")
        target_lease = WorkspaceTargetLease(self.workspace, operation=self.operation)
        target_lease.__enter__()
        self._target_lease = target_lease
        try:
            self.workspace.mkdir(parents=True, exist_ok=True)
            self._key = _acquire_file_lease(self.path, self.metadata_path, self.operation)
            self._entered = True
            return self
        except BaseException:
            target_lease.__exit__(None, None, None)
            self._target_lease = None
            raise

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        try:
            if self._entered and self._key is not None:
                _release_file_lease(self._key)
        finally:
            self._entered = False
            self._key = None
            if self._target_lease is not None:
                self._target_lease.__exit__(None, None, None)
                self._target_lease = None


def workspace_target_lease_paths(workspace: Path) -> tuple[Path, Path]:
    resolved = workspace.expanduser().resolve()
    key = os.path.normcase(str(resolved))
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    stem = f"{WORKSPACE_TARGET_LEASE_PREFIX}{digest}"
    return resolved.parent / f"{stem}.lock", resolved.parent / f"{stem}.lock.json"


def _acquire_file_lease(path: Path, metadata_path: Path, operation: str) -> str:
    key = os.path.normcase(str(path.resolve()))
    thread_id = threading.get_ident()
    process_id = os.getpid()
    with _REGISTRY_LOCK:
        _discard_inherited_leases(process_id)
        held = _HELD_LEASES.get(key)
        if held is not None:
            if held.owner_thread_id != thread_id:
                raise _busy_error(held.metadata)
            held.references += 1
            return key

        lock_file = _open_lock_file(path)
        try:
            _ensure_lock_byte(lock_file)
            _try_lock(lock_file)
        except OSError as error:
            lock_file.close()
            if error.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                raise
            metadata = _read_metadata(metadata_path)
            raise _busy_error(metadata) from error

        metadata = {
            "schema_version": 1,
            "pid": process_id,
            "thread_id": thread_id,
            "started_at": datetime.now(UTC).isoformat(),
            "operation": operation,
        }
        try:
            _write_metadata(metadata_path, metadata)
        except OSError:
            _unlock(lock_file)
            lock_file.close()
            raise
        _HELD_LEASES[key] = _HeldLease(
            file=lock_file,
            owner_thread_id=thread_id,
            owner_pid=process_id,
            references=1,
            metadata=metadata,
        )
        return key


def _release_file_lease(key: str) -> None:
    with _REGISTRY_LOCK:
        held = _HELD_LEASES.get(key)
        if held is None:
            return
        held.references -= 1
        if held.references == 0:
            try:
                _unlock(held.file)
            finally:
                held.file.close()
                _HELD_LEASES.pop(key, None)


def _open_lock_file(path: Path) -> BinaryIO:
    try:
        return path.open("x+b")
    except FileExistsError:
        return path.open("r+b")


def _ensure_lock_byte(file: BinaryIO) -> None:
    file.seek(0, os.SEEK_END)
    if file.tell() == 0:
        file.write(b"\0")
        file.flush()
        os.fsync(file.fileno())
    file.seek(0)


def _try_lock(file: BinaryIO) -> None:
    file.seek(0)
    if os.name == "nt":
        msvcrt = importlib.import_module("msvcrt")
        msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
        return
    fcntl = importlib.import_module("fcntl")
    fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(file: BinaryIO) -> None:
    file.seek(0)
    if os.name == "nt":
        msvcrt = importlib.import_module("msvcrt")
        msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
        return
    fcntl = importlib.import_module("fcntl")
    fcntl.flock(file.fileno(), fcntl.LOCK_UN)


def _write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    payload = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
    temporary_path = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as metadata_file:
            metadata_file.write(payload)
            metadata_file.flush()
            os.fsync(metadata_file.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _read_metadata(path: Path) -> dict[str, Any]:
    try:
        payload = path.read_text(encoding="utf-8")[:_METADATA_LIMIT_BYTES]
        value = json.loads(payload)
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _busy_error(metadata: dict[str, Any]) -> WorkspaceBusyError:
    pid = metadata.get("pid")
    started_at = metadata.get("started_at")
    operation = metadata.get("operation")
    details: list[str] = []
    if isinstance(pid, int) and pid > 0:
        details.append(f"PID {pid}")
    if isinstance(started_at, str) and started_at:
        details.append(f"started {started_at}")
    if isinstance(operation, str) and operation:
        details.append(f"operation {operation}")
    owner = f" ({', '.join(details)})" if details else ""
    return WorkspaceBusyError(
        "This project workspace is already being updated by another TravelMovieAI "
        f"job{owner}. Wait for it to finish, then retry."
    )


def _discard_inherited_leases(process_id: int) -> None:
    inherited = [key for key, held in _HELD_LEASES.items() if held.owner_pid != process_id]
    for key in inherited:
        held = _HELD_LEASES.pop(key)
        held.file.close()
