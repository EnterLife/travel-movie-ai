"""TravelMovieAI package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("travelmovieai")
except PackageNotFoundError:  # pragma: no cover - only an unpackaged source tree
    __version__ = "0.1.0"
