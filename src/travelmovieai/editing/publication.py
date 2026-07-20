"""Fail-closed publication helpers for rendered movies."""

import os
from pathlib import Path
from uuid import uuid4

from travelmovieai.core.exceptions import MontageError


def render_candidate_path(output_path: Path) -> Path:
    """Return a hidden candidate beside the final movie on the same volume."""
    return output_path.with_name(f".{output_path.stem}.{uuid4().hex}.candidate{output_path.suffix}")


def publish_render_candidate(candidate_path: Path, output_path: Path) -> None:
    """Atomically replace the delivery only with a validated sibling candidate."""
    if candidate_path.parent.resolve() != output_path.parent.resolve():
        raise MontageError("Rendered candidate must be on the same volume as the delivery.")
    if not candidate_path.is_file():
        raise MontageError("Validated render candidate is missing.")
    try:
        os.replace(candidate_path, output_path)
    except OSError as error:
        raise MontageError("Could not publish the validated movie atomically.") from error
