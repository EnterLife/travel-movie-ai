import importlib
import socket
import sys
from pathlib import Path

import pytest

from travelmovieai import desktop
from travelmovieai.core.config import Settings
from travelmovieai.core.exceptions import DependencyUnavailableError
from travelmovieai.desktop import (
    APP_MUTEX_NAME,
    _application_mutex,
    _configure_frozen_startup_logging,
    _desktop_settings,
    _desktop_url,
    _ensure_port_available,
    _load_desktop_settings,
    _load_qt,
)


def test_desktop_module_keeps_pyside_optional(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = importlib.import_module

    def missing(name: str) -> object:
        if name.startswith("PySide6"):
            raise ImportError(name)
        return original_import(name)

    monkeypatch.setattr(importlib, "import_module", missing)

    with pytest.raises(DependencyUnavailableError, match="desktop"):
        _load_qt()


def test_desktop_url_is_always_loopback() -> None:
    assert _desktop_url(8000) == "http://127.0.0.1:8000/"


def test_desktop_url_rejects_invalid_port() -> None:
    with pytest.raises(ValueError, match="between 1 and 65535"):
        _desktop_url(0)


def test_frozen_desktop_uses_persistent_local_app_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = tmp_path / "bundle"
    local_data = tmp_path / "local"
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle), raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(local_data))

    settings = _desktop_settings(Settings(piper_model=Path("models") / "voice.onnx"))

    assert settings.workspace == local_data / "TravelMovieAI" / "workspace"
    assert settings.model_cache == local_data / "TravelMovieAI" / "models"
    assert settings.music_library == bundle / "assets" / "music"
    assert settings.piper_model == local_data / "TravelMovieAI" / "models" / "voice.onnx"
    assert (local_data / "TravelMovieAI").is_dir()


def test_frozen_desktop_bootstraps_and_preserves_editable_user_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = tmp_path / "bundle"
    bundled_config = bundle / "configs" / "settings.toml"
    bundled_config.parent.mkdir(parents=True)
    bundled_config.write_text("web_port = 8123\n", encoding="utf-8")
    local_data = tmp_path / "local"
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle), raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(local_data))

    first = _load_desktop_settings()
    user_config = local_data / "TravelMovieAI" / "settings.toml"
    user_config.write_text("web_port = 8124\n", encoding="utf-8")
    second = _load_desktop_settings()

    assert first.web_port == 8123
    assert second.web_port == 8124
    assert user_config.read_text(encoding="utf-8") == "web_port = 8124\n"


def test_frozen_desktop_rejects_missing_bundled_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path / "bundle"), raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))

    with pytest.raises(DependencyUnavailableError, match="bundled settings"):
        _load_desktop_settings()


def test_frozen_desktop_configures_log_before_loading_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Path, tuple[Path, ...]]] = []
    local_data = tmp_path / "local"
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(local_data))

    def configure(path: Path, *, private_paths: tuple[Path, ...]) -> Path:
        calls.append((path, private_paths))
        return path

    monkeypatch.setattr(desktop, "configure_local_logging", configure)

    path = _configure_frozen_startup_logging()

    user_root = local_data / "TravelMovieAI"
    assert path == user_root / "logs" / "travelmovieai.log"
    assert calls == [(path, (Path.home(), user_root))]


def test_desktop_reports_port_conflict_before_server_start() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
        occupied.bind(("127.0.0.1", 0))
        port = occupied.getsockname()[1]
        with pytest.raises(DependencyUnavailableError, match="already in use"):
            _ensure_port_available(port)


def test_desktop_holds_installer_mutex_for_application_lifetime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    released: list[str] = []
    monkeypatch.setattr(
        desktop,
        "_acquire_application_mutex",
        lambda: lambda: released.append(APP_MUTEX_NAME),
    )

    with pytest.raises(RuntimeError, match="stop"), _application_mutex():
        assert released == []
        raise RuntimeError("stop")

    assert released == [APP_MUTEX_NAME]
