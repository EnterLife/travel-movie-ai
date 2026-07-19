"""Small process-local pool for expensive, reusable model runtimes."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from threading import RLock


@dataclass(slots=True)
class _Entry[T]:
    value: T
    disposer: Callable[[T], None]
    leases: int = 0


@dataclass(frozen=True, slots=True)
class ModelPoolStats:
    loaded_entries: int
    active_leases: int
    idle_entries: int
    max_idle_entries: int


class ModelLease[T]:
    def __init__(self, pool: BoundedModelPool[T], key: Hashable, value: T) -> None:
        self._pool = pool
        self._key = key
        self.value = value
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._pool._release(self._key)

    def __enter__(self) -> T:
        return self.value

    def __exit__(self, *_: object) -> None:
        self.release()


class BoundedModelPool[T]:
    """Retain a bounded LRU of idle models while never evicting active leases."""

    def __init__(self, max_idle_entries: int = 1) -> None:
        if max_idle_entries < 0:
            raise ValueError("max_idle_entries cannot be negative")
        self._max_idle_entries = max_idle_entries
        self._entries: OrderedDict[Hashable, _Entry[T]] = OrderedDict()
        self._lock = RLock()

    def configure(self, max_idle_entries: int) -> None:
        if max_idle_entries < 0:
            raise ValueError("max_idle_entries cannot be negative")
        with self._lock:
            self._max_idle_entries = max_idle_entries
            evicted = self._collect_evictions()
        _dispose(evicted)

    def acquire(
        self,
        key: Hashable,
        loader: Callable[[], T],
        disposer: Callable[[T], None],
    ) -> ModelLease[T]:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                entry = _Entry(value=loader(), disposer=disposer)
                self._entries[key] = entry
            entry.leases += 1
            self._entries.move_to_end(key)
            evicted = self._collect_evictions()
            lease = ModelLease(self, key, entry.value)
        _dispose(evicted)
        return lease

    def clear_idle(self) -> None:
        with self._lock:
            evicted = [
                self._entries.pop(key)
                for key, entry in list(self._entries.items())
                if entry.leases == 0
            ]
        _dispose(evicted)

    def stats(self) -> ModelPoolStats:
        with self._lock:
            active = sum(entry.leases for entry in self._entries.values())
            idle = sum(entry.leases == 0 for entry in self._entries.values())
            return ModelPoolStats(
                loaded_entries=len(self._entries),
                active_leases=active,
                idle_entries=idle,
                max_idle_entries=self._max_idle_entries,
            )

    def _release(self, key: Hashable) -> None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None or entry.leases == 0:
                return
            entry.leases -= 1
            self._entries.move_to_end(key)
            evicted = self._collect_evictions()
        _dispose(evicted)

    def _collect_evictions(self) -> list[_Entry[T]]:
        idle_keys = [key for key, entry in self._entries.items() if entry.leases == 0]
        excess = max(0, len(idle_keys) - self._max_idle_entries)
        return [self._entries.pop(key) for key in idle_keys[:excess]]


def _dispose[T](entries: list[_Entry[T]]) -> None:
    for entry in entries:
        entry.disposer(entry.value)
