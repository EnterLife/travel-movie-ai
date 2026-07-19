"""Optional PySide6 launcher for the package-local web interface."""

import importlib
import os
import socket
import sys
from pathlib import Path
from threading import Thread
from typing import Any

import uvicorn

from travelmovieai.core.config import Settings, load_settings
from travelmovieai.core.exceptions import DependencyUnavailableError, TravelMovieError
from travelmovieai.web.app import create_app


class _WebServerThread(Thread):
    def __init__(self, settings: Settings) -> None:
        super().__init__(name="travelmovieai-desktop-web", daemon=True)
        local_settings = settings.model_copy(update={"web_host": "127.0.0.1"})
        self.server = uvicorn.Server(
            uvicorn.Config(
                create_app(local_settings),
                host="127.0.0.1",
                port=local_settings.web_port,
                log_level="warning",
                access_log=False,
            )
        )

    def run(self) -> None:
        self.server.run()

    def stop(self) -> None:
        self.server.should_exit = True
        if self.is_alive():
            self.join(timeout=5)


def main(settings: Settings | None = None) -> int:
    qt_core, qt_gui, qt_widgets = _load_qt()
    resolved_settings = _desktop_settings(settings or load_settings())
    _ensure_port_available(resolved_settings.web_port)
    application = qt_widgets.QApplication.instance() or qt_widgets.QApplication(sys.argv)
    server = _WebServerThread(resolved_settings)
    server.start()

    window = qt_widgets.QMainWindow()
    window.setWindowTitle("TravelMovieAI")
    window.resize(560, 300)
    central = qt_widgets.QWidget()
    layout = qt_widgets.QVBoxLayout(central)
    title = qt_widgets.QLabel("TravelMovieAI")
    title.setStyleSheet("font-size: 30px; font-weight: 700;")
    description = qt_widgets.QLabel(
        "The local movie workspace is running on this computer. "
        "Open it in your browser to scan media and build a movie."
    )
    description.setWordWrap(True)
    open_button = qt_widgets.QPushButton("Open local workspace")
    open_button.setMinimumHeight(44)
    open_button.setEnabled(False)
    url = _desktop_url(resolved_settings.web_port)
    open_button.clicked.connect(lambda: qt_gui.QDesktopServices.openUrl(qt_core.QUrl(url)))
    layout.addStretch(1)
    layout.addWidget(title)
    layout.addWidget(description)
    layout.addSpacing(18)
    layout.addWidget(open_button)
    layout.addStretch(1)
    window.setCentralWidget(central)
    application.aboutToQuit.connect(server.stop)
    attempts = 0

    def wait_for_server() -> None:
        nonlocal attempts
        attempts += 1
        if server.server.started:
            open_button.setEnabled(True)
            description.setText(
                "The local movie workspace is ready. All project data stays on this computer."
            )
            qt_gui.QDesktopServices.openUrl(qt_core.QUrl(url))
            return
        if not server.is_alive() or attempts >= 100:
            description.setText(
                "The local server could not start. Close this window, run diagnostics, "
                "and check whether the configured port is already in use."
            )
            return
        qt_core.QTimer.singleShot(100, wait_for_server)

    qt_core.QTimer.singleShot(100, wait_for_server)
    window.show()
    return int(application.exec())


def run() -> None:
    try:
        raise SystemExit(main())
    except TravelMovieError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from error


def _load_qt() -> tuple[Any, Any, Any]:
    try:
        return (
            importlib.import_module("PySide6.QtCore"),
            importlib.import_module("PySide6.QtGui"),
            importlib.import_module("PySide6.QtWidgets"),
        )
    except ImportError as error:
        raise DependencyUnavailableError(
            'The desktop shell requires the optional "desktop" dependency group.'
        ) from error


def _desktop_url(port: int) -> str:
    if port < 1 or port > 65535:
        raise ValueError("Desktop web port must be between 1 and 65535.")
    return f"http://127.0.0.1:{port}/"


def _desktop_settings(settings: Settings) -> Settings:
    """Move mutable frozen-app data out of the installer directory."""

    if not bool(getattr(sys, "frozen", False)):
        return settings
    local_app_data = os.environ.get("LOCALAPPDATA")
    if not local_app_data:
        raise DependencyUnavailableError(
            "LOCALAPPDATA is unavailable; TravelMovieAI cannot select a safe user data folder."
        )
    user_root = Path(local_app_data) / "TravelMovieAI"
    bundled_root_value = getattr(sys, "_MEIPASS", None)
    bundled_root = Path(bundled_root_value) if isinstance(bundled_root_value, str) else None
    updates: dict[str, object] = {}
    if not settings.workspace.is_absolute():
        updates["workspace"] = user_root / "workspace"
    if not settings.model_cache.is_absolute():
        updates["model_cache"] = user_root / "models"
    if not settings.music_library.is_absolute() and bundled_root is not None:
        updates["music_library"] = bundled_root / settings.music_library
    if settings.piper_model is not None and not settings.piper_model.is_absolute():
        relative_voice = settings.piper_model
        if relative_voice.parts and relative_voice.parts[0].casefold() == "models":
            relative_voice = Path(*relative_voice.parts[1:])
        updates["piper_model"] = user_root / "models" / relative_voice
    user_root.mkdir(parents=True, exist_ok=True)
    return settings.model_copy(update=updates)


def _ensure_port_available(port: int) -> None:
    """Fail before launching Qt when another local process owns the configured port."""

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", port))
    except OSError as error:
        raise DependencyUnavailableError(
            f"Local web port {port} is already in use. Change web_port in settings.toml."
        ) from error


if __name__ == "__main__":
    run()
