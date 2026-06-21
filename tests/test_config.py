from pathlib import Path

import pytest

from travelmovieai.core.config import Settings, load_settings
from travelmovieai.core.exceptions import ConfigurationError
from travelmovieai.domain.models import QuickMontageSettings


def test_load_settings_reads_toml(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.toml"
    config_path.write_text(
        "\n".join(
            [
                'workspace = "projects"',
                'vision_provider = "florence"',
                'vision_model = "microsoft/Florence-2-large"',
                "allow_model_download = false",
                "web_port = 8123",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_settings(config_path)

    assert settings.workspace == Path("projects")
    assert settings.vision_provider == "florence"
    assert settings.allow_model_download is False
    assert settings.web_port == 8123


def test_load_settings_uses_defaults_when_file_is_missing(tmp_path: Path) -> None:
    assert load_settings(tmp_path / "missing.toml") == Settings()


def test_load_settings_rejects_unknown_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.toml"
    config_path.write_text('remote_api_key = "unexpected"\n', encoding="utf-8")

    with pytest.raises(ConfigurationError, match="remote_api_key"):
        load_settings(config_path)


def test_quick_montage_settings_validate_analysis_quality_mode() -> None:
    settings = QuickMontageSettings(analysis_quality_mode="deep")

    assert settings.analysis_quality_mode == "deep"
    with pytest.raises(ValueError, match="analysis_quality_mode"):
        QuickMontageSettings(analysis_quality_mode="extreme")
