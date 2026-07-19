from pathlib import Path

import pytest

from travelmovieai.infrastructure.model_pool import BoundedModelPool
from travelmovieai.infrastructure.vision import (
    LocalQwenVisionProvider,
    _VisionRuntime,
    clear_idle_vision_models,
    vision_model_pool_stats,
)


def test_model_pool_reuses_idle_value_and_evicts_least_recently_used() -> None:
    pool: BoundedModelPool[str] = BoundedModelPool(max_idle_entries=1)
    loaded: list[str] = []
    disposed: list[str] = []

    first = pool.acquire("a", lambda: _load("a", loaded), disposed.append)
    first.release()
    reused = pool.acquire("a", lambda: _load("unexpected", loaded), disposed.append)
    reused.release()
    second = pool.acquire("b", lambda: _load("b", loaded), disposed.append)
    second.release()

    assert loaded == ["a", "b"]
    assert disposed == ["a"]
    assert pool.stats().idle_entries == 1
    pool.clear_idle()
    assert disposed == ["a", "b"]


def test_model_pool_never_evicts_an_active_lease() -> None:
    pool: BoundedModelPool[str] = BoundedModelPool(max_idle_entries=0)
    disposed: list[str] = []

    lease = pool.acquire("active", lambda: "model", disposed.append)
    pool.configure(0)

    assert pool.stats().loaded_entries == 1
    assert pool.stats().active_leases == 1
    assert disposed == []
    lease.release()
    lease.release()
    assert disposed == ["model"]


def test_vision_stage_release_retains_runtime_for_next_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_idle_vision_models()
    load_count = 0

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def empty_cache() -> None:
            raise AssertionError("CPU runtime must not clear a CUDA cache")

    class FakeTorch:
        cuda = FakeCuda()

    class FakeModel:
        device = "cpu"

    def fake_load(_: LocalQwenVisionProvider) -> _VisionRuntime:
        nonlocal load_count
        load_count += 1
        return _VisionRuntime(FakeTorch(), object(), FakeModel())

    monkeypatch.setattr(LocalQwenVisionProvider, "_load_runtime", fake_load)
    first = LocalQwenVisionProvider("test/model", cache_dir=tmp_path, model_pool_size=1)
    second = LocalQwenVisionProvider("test/model", cache_dir=tmp_path, model_pool_size=1)

    first.prepare()
    first.release()
    assert vision_model_pool_stats().idle_entries == 1
    second.prepare()
    second.release()

    assert load_count == 1
    assert vision_model_pool_stats().loaded_entries == 1
    clear_idle_vision_models()
    assert vision_model_pool_stats().loaded_entries == 0


def _load(value: str, calls: list[str]) -> str:
    calls.append(value)
    return value
