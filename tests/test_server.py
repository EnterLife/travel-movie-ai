import sys
from typing import Any

import pytest

from travelmovieai.core.config import Settings, validate_loopback_web_host
from travelmovieai.web import server


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_loopback_host_helper_accepts_local_addresses(host: str) -> None:
    assert validate_loopback_web_host(f" {host} ") == host


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.10", "example.test"])
def test_server_cli_rejects_non_loopback_host(
    host: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(sys, "argv", ["travelmovieai-web", "--host", host, "--no-browser"])
    monkeypatch.setattr(server.uvicorn, "run", lambda *args, **kwargs: calls.append(kwargs))

    with pytest.raises(SystemExit) as error:
        server.main()

    assert error.value.code == 2
    assert calls == []


def test_server_cli_passes_validated_loopback_to_uvicorn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, dict[str, Any]]] = []
    application = object()
    monkeypatch.setattr(
        sys,
        "argv",
        ["travelmovieai-web", "--host", "::1", "--port", "8123", "--no-browser"],
    )
    monkeypatch.setattr(server, "load_settings", lambda: Settings())
    monkeypatch.setattr(server, "create_app", lambda settings: application)
    monkeypatch.setattr(
        server.uvicorn,
        "run",
        lambda app, **kwargs: calls.append((app, kwargs)),
    )

    server.main()

    assert calls == [
        (
            application,
            {"host": "::1", "port": 8123, "log_level": "info"},
        )
    ]


def test_ipv6_loopback_browser_url_uses_brackets() -> None:
    assert server._browser_url("::1", 8000) == "http://[::1]:8000"
    assert server._browser_url("127.0.0.1", 8000) == "http://127.0.0.1:8000"
