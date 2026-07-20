import logging
from pathlib import Path

from travelmovieai.core.logging import (
    configure_local_logging,
    configured_log_path,
    correlation_context,
    register_private_log_paths,
)


def test_local_logging_is_rotating_bounded_and_idempotent(tmp_path: Path) -> None:
    root = logging.getLogger()
    before = list(root.handlers)
    try:
        path = configure_local_logging(tmp_path / "logs" / "app.log")
        first_count = len(root.handlers)
        configure_local_logging(path)
        logging.getLogger("travelmovieai.test").error("diagnostic without private media")
        for handler in root.handlers:
            handler.flush()

        assert path.is_file()
        assert len(root.handlers) == first_count
        assert "diagnostic without private media" in path.read_text(encoding="utf-8")
    finally:
        for handler in list(root.handlers):
            if handler not in before:
                handler.close()
                root.removeHandler(handler)


def test_local_logging_redacts_paths_secrets_and_records_correlation(tmp_path: Path) -> None:
    root = logging.getLogger()
    before = list(root.handlers)
    private_source = tmp_path / "private trip" / "clip.mp4"
    try:
        path = configure_local_logging(
            tmp_path / "logs" / "app.log",
            private_paths=(path for path in [private_source.parent]),
        )
        second_private_source = tmp_path / "later-private" / "clip.mp4"
        configure_local_logging(path, private_paths=[second_private_source.parent])
        registered_source = tmp_path / "registered-private" / "clip.mp4"
        register_private_log_paths((registered_source.parent,))
        with correlation_context("request-123"):
            logging.getLogger("travelmovieai.web").exception(
                "Failed source=%s second=%s registered=%s HF_TOKEN='secret with spaces'",
                private_source,
                second_private_source,
                registered_source,
            )
        for handler in root.handlers:
            handler.flush()

        rendered = path.read_text(encoding="utf-8")
        assert configured_log_path() == path
        assert "request-123" in rendered
        assert "private trip" not in rendered
        assert "later-private" not in rendered
        assert "registered-private" not in rendered
        assert "secret with spaces" not in rendered
        assert "<local-path>" in rendered
        assert "<redacted>" in rendered
    finally:
        for handler in list(root.handlers):
            if handler not in before:
                handler.close()
                root.removeHandler(handler)
