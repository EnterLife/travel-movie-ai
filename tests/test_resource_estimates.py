from datetime import UTC, datetime
from pathlib import Path

import pytest

from travelmovieai.application.resource_estimates import estimate_project_resources
from travelmovieai.core.config import Settings
from travelmovieai.domain.enums import MediaType
from travelmovieai.domain.models import MediaAsset, QuickMontageSettings


def test_project_resource_estimate_accounts_for_4k_proxies_and_ai_runtime() -> None:
    assets = [_asset("4k.mp4", size_bytes=10 * 1024**3, duration_seconds=600, width=3840)]
    settings = Settings(analysis_proxy_mode="auto", analysis_proxy_max_dimension=1920)
    basic = QuickMontageSettings(semantic_analysis=False, analysis_quality_mode="fast")
    semantic = basic.model_copy(update={"semantic_analysis": True, "analysis_quality_mode": "deep"})

    basic_estimate = estimate_project_resources(
        assets,
        settings=settings,
        montage_settings=basic,
    )
    semantic_estimate = estimate_project_resources(
        assets,
        settings=settings,
        montage_settings=semantic,
    )

    assert basic_estimate.workload.proxy_candidate_count == 1
    assert basic_estimate.estimated_proxy_bytes > 0
    assert basic_estimate.estimated_peak_workspace_bytes > (
        basic_estimate.estimated_analysis_workspace_bytes
    )
    assert semantic_estimate.estimated_frame_cache_bytes > (
        basic_estimate.estimated_frame_cache_bytes
    )
    assert semantic_estimate.runtime.likely_seconds > basic_estimate.runtime.likely_seconds
    assert (
        semantic_estimate.runtime.lower_seconds
        <= semantic_estimate.runtime.likely_seconds
        <= semantic_estimate.runtime.upper_seconds
    )


def test_project_resource_estimate_uses_known_scene_count() -> None:
    estimate = estimate_project_resources(
        [_asset("clip.mp4", size_bytes=1000, duration_seconds=60, width=1920)],
        settings=Settings(),
        montage_settings=QuickMontageSettings(),
        known_scene_count=123,
    )

    assert estimate.workload.estimated_scene_count == 123
    with pytest.raises(ValueError, match="known_scene_count"):
        estimate_project_resources(
            [],
            settings=Settings(),
            montage_settings=QuickMontageSettings(),
            known_scene_count=-1,
        )


def _asset(
    filename: str,
    *,
    size_bytes: int,
    duration_seconds: float,
    width: int,
) -> MediaAsset:
    return MediaAsset(
        path=Path("synthetic") / filename,
        relative_path=Path(filename),
        media_type=MediaType.VIDEO,
        extension=".mp4",
        size_bytes=size_bytes,
        modified_at=datetime(2026, 1, 1, tzinfo=UTC),
        modified_ns=1,
        duration_seconds=duration_seconds,
        width=width,
        height=width * 9 // 16,
        fps=30,
    )
