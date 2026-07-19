import pytest

from travelmovieai.application.scale_benchmark import (
    build_synthetic_assets,
    run_synthetic_metadata_benchmark,
)


def test_synthetic_benchmark_handles_500_assets_and_100_gib_without_media() -> None:
    source_bytes = 128 * 1024**3

    first = run_synthetic_metadata_benchmark(
        asset_count=512,
        total_source_bytes=source_bytes,
    )
    second = run_synthetic_metadata_benchmark(
        asset_count=512,
        total_source_bytes=source_bytes,
    )

    assert first.asset_count == 512
    assert first.source_bytes == source_bytes
    assert first.fingerprint == second.fingerprint
    assert first.metadata_json_bytes == second.metadata_json_bytes
    assert first.metadata_json_bytes < first.asset_count * 2_000
    assert first.scene_count >= first.asset_count * 0.9
    assert first.sqlite_bytes > first.metadata_json_bytes
    assert first.sqlite_write_seconds >= 0
    assert first.sqlite_read_seconds >= 0
    assert first.asset_throughput_per_second > 0
    assert first.peak_traced_memory_bytes > 0
    assert first.estimate.workload.proxy_candidate_count >= 450
    assert first.estimate.workload.estimated_scene_count >= first.asset_count
    assert first.estimate.estimated_analysis_workspace_bytes < source_bytes


def test_synthetic_asset_builder_rejects_invalid_metadata_scale() -> None:
    with pytest.raises(ValueError, match="asset_count"):
        build_synthetic_assets(asset_count=0, total_source_bytes=1)
    with pytest.raises(ValueError, match="total_source_bytes"):
        build_synthetic_assets(asset_count=10, total_source_bytes=9)
