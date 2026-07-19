import importlib
import socket
import sys
from pathlib import Path

import pytest

from travelmovieai.core.config import Settings
from travelmovieai.core.exceptions import DependencyUnavailableError
from travelmovieai.desktop import (
    _desktop_settings,
    _desktop_url,
    _ensure_port_available,
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


def test_desktop_reports_port_conflict_before_server_start() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
        occupied.bind(("127.0.0.1", 0))
        port = occupied.getsockname()[1]
        with pytest.raises(DependencyUnavailableError, match="already in use"):
            _ensure_port_available(port)
