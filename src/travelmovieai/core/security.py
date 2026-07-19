"""Privacy-safe formatting for errors that may be persisted or shown in the UI."""

import re
from collections.abc import Iterable, Sequence
from pathlib import Path

_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(api[_-]?key|authorization|password|secret|token)\b(\s*[:=]\s*)([^\s,;]+)"
)
_BEARER_TOKEN = re.compile(r"(?i)\bbearer\s+[^\s,;]+")


def redact_sensitive_text(
    value: str,
    *,
    private_paths: Iterable[Path] = (),
    max_characters: int = 2000,
) -> str:
    """Remove common credentials and bound text before it reaches persistent state."""

    without_bearer_tokens = _BEARER_TOKEN.sub("Bearer <redacted>", value)
    redacted = _SECRET_ASSIGNMENT.sub(r"\1\2<redacted>", without_bearer_tokens)
    redacted = _redact_paths(redacted, private_paths)
    return _bounded_tail(redacted, max_characters)


def sanitize_process_error(
    stderr: str,
    *,
    private_paths: Iterable[Path] = (),
    fallback: str,
    max_characters: int = 1000,
) -> str:
    """Redact local paths and credentials from bounded subprocess diagnostics."""

    detail = stderr.strip() or fallback
    detail = _redact_paths(detail, private_paths)
    return redact_sensitive_text(detail, max_characters=max_characters)


def _redact_paths(value: str, private_paths: Iterable[Path]) -> str:
    redacted = value
    for path in sorted(private_paths, key=lambda item: len(str(item)), reverse=True):
        raw = str(path)
        variants = {raw, raw.replace("\\", "/"), raw.replace("/", "\\")}
        for variant in variants:
            if variant:
                redacted = re.sub(
                    re.escape(variant),
                    "<local-path>",
                    redacted,
                    flags=re.IGNORECASE,
                )
    return redacted


def absolute_command_paths(command: Sequence[str]) -> list[Path]:
    """Return standalone absolute path arguments without interpreting shell syntax."""

    paths: list[Path] = []
    for argument in command:
        if not argument or len(argument) > 2048 or "\n" in argument:
            continue
        candidate = Path(argument)
        if candidate.is_absolute():
            paths.append(candidate)
    return paths


def _bounded_tail(value: str, max_characters: int) -> str:
    if max_characters < 1:
        raise ValueError("max_characters must be positive")
    if len(value) <= max_characters:
        return value
    marker = "… "
    return marker + value[-(max_characters - len(marker)) :]
