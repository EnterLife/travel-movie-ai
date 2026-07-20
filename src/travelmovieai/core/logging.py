"""Local rotating application logs for console-less desktop and web launches."""

import logging
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from logging.handlers import RotatingFileHandler
from pathlib import Path

from travelmovieai.core.security import redact_sensitive_text

_HANDLER_MARKER = "travelmovieai-local-file"
_CORRELATION_ID: ContextVar[str] = ContextVar("travelmovieai_correlation_id", default="-")


class _LocalFileHandler(RotatingFileHandler):
    private_paths: tuple[Path, ...] = ()


class _CorrelationFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        explicit = getattr(record, "request_id", None) or getattr(record, "job_id", None)
        record.correlation_id = str(explicit or _CORRELATION_ID.get())
        return True


class _PrivacySafeFormatter(logging.Formatter):
    def __init__(self, *, private_paths: Iterable[Path]) -> None:
        super().__init__("%(asctime)s %(levelname)s %(name)s [%(correlation_id)s] %(message)s")
        self._private_paths = tuple(private_paths)

    def format(self, record: logging.LogRecord) -> str:
        rendered = super().format(record)
        return redact_sensitive_text(
            rendered,
            private_paths=self._private_paths,
            max_characters=16_000,
        )


def configure_local_logging(
    log_path: Path,
    *,
    private_paths: Iterable[Path] = (),
) -> Path:
    """Attach one bounded UTF-8 file handler and return its resolved path."""

    resolved = log_path.expanduser().resolve()
    normalized_private_paths = _normalize_private_paths(private_paths)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    for handler in root.handlers:
        if getattr(handler, "name", None) == _HANDLER_MARKER:
            if isinstance(handler, _LocalFileHandler):
                _extend_private_paths(handler, normalized_private_paths)
            existing = getattr(handler, "baseFilename", None)
            return Path(existing).resolve() if isinstance(existing, str) else resolved
    handler = _LocalFileHandler(
        resolved,
        maxBytes=2 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
        delay=True,
    )
    handler.name = _HANDLER_MARKER
    handler.private_paths = normalized_private_paths
    handler.addFilter(_CorrelationFilter())
    handler.setFormatter(_PrivacySafeFormatter(private_paths=normalized_private_paths))
    root.addHandler(handler)
    if root.level > logging.INFO:
        root.setLevel(logging.INFO)
    return resolved


def register_private_log_paths(private_paths: Iterable[Path]) -> None:
    """Extend redaction roots for already configured local logs."""

    normalized = _normalize_private_paths(private_paths)
    if not normalized:
        return
    for handler in logging.getLogger().handlers:
        if isinstance(handler, _LocalFileHandler) and handler.name == _HANDLER_MARKER:
            _extend_private_paths(handler, normalized)


def configured_log_path() -> Path | None:
    """Return the active local log path without creating a handler."""

    for handler in logging.getLogger().handlers:
        if getattr(handler, "name", None) != _HANDLER_MARKER:
            continue
        filename = getattr(handler, "baseFilename", None)
        if isinstance(filename, str):
            return Path(filename).resolve()
    return None


def _extend_private_paths(
    handler: _LocalFileHandler,
    private_paths: tuple[Path, ...],
) -> None:
    handler.acquire()
    try:
        combined_paths = tuple(dict.fromkeys((*handler.private_paths, *private_paths)))
        handler.private_paths = combined_paths
        handler.setFormatter(_PrivacySafeFormatter(private_paths=combined_paths))
    finally:
        handler.release()


def _normalize_private_paths(private_paths: Iterable[Path]) -> tuple[Path, ...]:
    return tuple(dict.fromkeys(path.expanduser().resolve() for path in private_paths))


@contextmanager
def correlation_context(correlation_id: str) -> Iterator[None]:
    """Attach a request or job identifier to every log record in this context."""

    token = _CORRELATION_ID.set(correlation_id)
    try:
        yield
    finally:
        _CORRELATION_ID.reset(token)
