"""Uvicorn launcher for the local web interface."""

import argparse
import webbrowser
from threading import Timer

import uvicorn

from travelmovieai.core.config import load_settings, validate_loopback_web_host
from travelmovieai.web.app import create_app


def main() -> None:
    settings = load_settings()
    parser = argparse.ArgumentParser(description="Run the TravelMovieAI web interface.")
    parser.add_argument(
        "--host",
        default=settings.web_host,
        type=validate_loopback_web_host,
    )
    parser.add_argument("--port", type=int, default=settings.web_port)
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the interface in the default browser.",
    )
    args = parser.parse_args()

    url = _browser_url(args.host, args.port)
    if not args.no_browser:
        Timer(1.0, webbrowser.open, args=(url,)).start()

    uvicorn.run(
        create_app(settings),
        host=args.host,
        port=args.port,
        log_level="info",
    )


def _browser_url(host: str, port: int) -> str:
    browser_host = f"[{host}]" if ":" in host else host
    return f"http://{browser_host}:{port}"


if __name__ == "__main__":
    main()
