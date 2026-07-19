from pathlib import Path
from uuid import uuid4

import pytest

from travelmovieai.core.config import Settings, load_settings
from travelmovieai.core.exceptions import ConfigurationError
from travelmovieai.domain.models import QuickMontageSettings, TimelineItem


def test_load_settings_reads_toml(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.toml"
    config_path.write_text(
        "\n".join(
            [
                'workspace = "projects"',
                'vision_provider = "florence"',
                'vision_model = "microsoft/Florence-2-large"',
                'story_provider = "local"',
                'story_model = "local-story-model"',
                "story_max_new_tokens = 512",
                "allow_model_download = false",
                "frame_extraction_timeout_seconds = 45",
                'analysis_proxy_mode = "always"',
                "analysis_proxy_max_dimension = 1280",
                "analysis_proxy_video_bitrate_mbps = 4.5",
                "analysis_proxy_timeout_seconds = 900",
                "vision_model_pool_size = 2",
                "render_timeout_seconds = 600",
                "render_disk_reserve_mb = 2048",
                "render_disk_safety_factor = 4.0",
                "web_port = 8123",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_settings(config_path)

    assert settings.workspace == Path("projects")
    assert settings.vision_provider == "florence"
    assert settings.story_provider == "local"
    assert settings.story_model == "local-story-model"
    assert settings.story_max_new_tokens == 512
    assert settings.allow_model_download is False
    assert settings.frame_extraction_timeout_seconds == 45
    assert settings.analysis_proxy_mode == "always"
    assert settings.analysis_proxy_max_dimension == 1280
    assert settings.analysis_proxy_video_bitrate_mbps == 4.5
    assert settings.analysis_proxy_timeout_seconds == 900
    assert settings.vision_model_pool_size == 2
    assert settings.render_timeout_seconds == 600
    assert settings.render_disk_reserve_mb == 2048
    assert settings.render_disk_safety_factor == 4.0
    assert settings.web_port == 8123


def test_load_settings_uses_defaults_when_file_is_missing(tmp_path: Path) -> None:
    settings = load_settings(tmp_path / "missing.toml")

    assert settings == Settings()
    assert settings.device == "auto"
    assert settings.resource_mode == "auto"
    assert settings.gpu_memory_reserve_mb == 1536
    assert settings.max_gpu_processes == 2
    assert settings.workers == 0
    assert settings.batch_size == 0
    assert settings.render_disk_reserve_mb == 1024
    assert settings.render_disk_safety_factor == 3.0
    assert settings.story_provider == "deterministic"
    assert settings.analysis_proxy_mode == "auto"
    assert settings.analysis_proxy_max_dimension == 1920
    assert settings.vision_model_pool_size == 1


def test_load_settings_rejects_unknown_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.toml"
    config_path.write_text('remote_api_key = "unexpected"\n', encoding="utf-8")

    with pytest.raises(ConfigurationError, match="remote_api_key"):
        load_settings(config_path)


def test_settings_reject_unsafe_gpu_resource_limits() -> None:
    with pytest.raises(ValueError, match="gpu_memory_reserve_mb"):
        Settings(gpu_memory_reserve_mb=128)
    with pytest.raises(ValueError, match="max_gpu_processes"):
        Settings(max_gpu_processes=0)


def test_settings_reject_unsafe_render_disk_limits() -> None:
    with pytest.raises(ValueError, match="render_disk_reserve_mb"):
        Settings(render_disk_reserve_mb=-1)
    with pytest.raises(ValueError, match="render_disk_safety_factor"):
        Settings(render_disk_safety_factor=0.5)


def test_settings_reject_public_web_bind_without_authentication() -> None:
    assert Settings(web_host="localhost").web_host == "localhost"
    with pytest.raises(ValueError, match="loopback"):
        Settings(web_host="0.0.0.0")


def test_settings_reject_unsafe_proxy_and_model_pool_limits() -> None:
    with pytest.raises(ValueError, match="analysis_proxy_max_dimension"):
        Settings(analysis_proxy_max_dimension=320)
    with pytest.raises(ValueError, match="analysis_proxy_video_bitrate_mbps"):
        Settings(analysis_proxy_video_bitrate_mbps=0)
    with pytest.raises(ValueError, match="vision_model_pool_size"):
        Settings(vision_model_pool_size=5)


def test_settings_validate_local_story_model_limits() -> None:
    settings = Settings(story_provider="local", story_model="local-model", story_max_new_tokens=256)

    assert settings.story_model == "local-model"
    with pytest.raises(ValueError, match="story_model"):
        Settings(story_model="")
    with pytest.raises(ValueError, match="story_max_new_tokens"):
        Settings(story_max_new_tokens=64)


def test_quick_montage_settings_validate_analysis_quality_mode() -> None:
    settings = QuickMontageSettings(analysis_quality_mode="deep")

    assert settings.analysis_quality_mode == "deep"
    with pytest.raises(ValueError, match="analysis_quality_mode"):
        QuickMontageSettings(analysis_quality_mode="extreme")


def test_quick_montage_settings_default_to_safe_cinematic_transitions() -> None:
    settings = QuickMontageSettings()

    assert settings.transition == "cinematic"


@pytest.mark.parametrize("transition", ["cinematic", "fade", "wipeleft", "slideright"])
def test_quick_montage_settings_preserve_safe_requested_transition(
    transition: str,
) -> None:
    settings = QuickMontageSettings.model_validate({"transition": transition})

    assert settings.transition == transition


@pytest.mark.parametrize("transition", ["dissolve", "soft"])
def test_quick_montage_settings_reject_pixel_dissolve_presets(transition: str) -> None:
    with pytest.raises(ValueError, match="transition"):
        QuickMontageSettings.model_validate({"transition": transition})


def test_timeline_item_accepts_fade_and_rejects_pixel_dissolve() -> None:
    payload = {
        "scene_id": uuid4(),
        "source_start_seconds": 0,
        "source_end_seconds": 1,
    }

    item = TimelineItem.model_validate({**payload, "transition": "fade"})

    assert item.transition == "fade"
    with pytest.raises(ValueError, match="transition"):
        TimelineItem.model_validate({**payload, "transition": "dissolve"})


def test_quick_montage_settings_use_full_music_volume_by_default() -> None:
    settings = QuickMontageSettings()

    assert settings.music_volume == 1.0


def test_quick_montage_settings_default_to_auto_rendering() -> None:
    settings = QuickMontageSettings()

    assert settings.render_device == "auto"


def test_quick_montage_settings_require_even_render_dimensions() -> None:
    settings = QuickMontageSettings(width=1280, height=720)

    assert (settings.width, settings.height) == (1280, 720)
    with pytest.raises(ValueError, match="multiple of 2"):
        QuickMontageSettings(width=1279)
    with pytest.raises(ValueError, match="multiple of 2"):
        QuickMontageSettings(height=719)
