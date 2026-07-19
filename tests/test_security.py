from pathlib import Path

import pytest

from travelmovieai.core.security import (
    absolute_command_paths,
    redact_sensitive_text,
    sanitize_process_error,
)


def test_process_error_redacts_unicode_local_paths_credentials_and_bounds_text() -> None:
    source = Path(r"C:\Users\Private\Видео отпуска\секрет.mp4")
    stderr = f"{'diagnostic ' * 200}{str(source).lower()} token=do-not-persist"

    detail = sanitize_process_error(
        stderr,
        private_paths=[source],
        fallback="unknown FFmpeg error",
        max_characters=180,
    )

    assert len(detail) <= 180
    assert "Users" not in detail
    assert "Видео отпуска" not in detail
    assert "do-not-persist" not in detail
    assert "<local-path>" in detail
    assert "<redacted>" in detail


def test_process_error_uses_fallback_and_extracts_only_absolute_command_paths() -> None:
    source = Path(r"C:\Media Library\clip.mp4")

    assert sanitize_process_error("", fallback="unknown error") == "unknown error"
    assert absolute_command_paths(["ffmpeg", "-i", str(source), "relative.mp4"]) == [source]
    with pytest.raises(ValueError, match="max_characters"):
        redact_sensitive_text("failure", max_characters=0)


def test_persisted_error_redacts_project_and_configured_media_paths() -> None:
    project = Path(r"C:\Users\Private\Секретная поездка")
    music = project / "личная музыка.wav"
    error = f"Failed in {project}; soundtrack={music}; password=hunter2"

    redacted = redact_sensitive_text(error, private_paths=[music, project])

    assert "Private" not in redacted
    assert "Секретная поездка" not in redacted
    assert "личная музыка" not in redacted
    assert "hunter2" not in redacted
    assert redacted.count("<local-path>") >= 1
