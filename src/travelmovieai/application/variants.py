"""Safe, non-overwriting output names for reusable movie analyses."""

import hashlib
import re
import unicodedata
from pathlib import Path
from uuid import UUID

from travelmovieai.core.exceptions import InvalidProjectPathError

_UNSAFE_NAME = re.compile(r"[\\/\x00-\x1f]")
_SLUG_SEPARATOR = re.compile(r"[^a-z0-9]+")


def validate_variant_name(value: str) -> str:
    name = value.strip()
    if not name or len(name) > 80:
        raise ValueError("Movie variant name must contain 1 to 80 characters.")
    if name in {".", ".."} or _UNSAFE_NAME.search(name):
        raise ValueError("Movie variant name cannot contain path separators or control characters.")
    return name


def safe_variant_slug(value: str) -> str:
    name = validate_variant_name(value)
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    slug = _SLUG_SEPARATOR.sub("-", ascii_name.casefold()).strip("-")[:64]
    if slug:
        return slug
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:12]
    return f"variant-{digest}"


def variant_output_path(workspace: Path, variant_name: str, job_id: UUID) -> Path:
    slug = safe_variant_slug(variant_name)
    variants_dir = (workspace / "artifacts" / "variants").resolve()
    output = (variants_dir / f"{slug}-{job_id.hex}.mp4").resolve()
    if not output.is_relative_to(variants_dir):
        raise InvalidProjectPathError("Invalid movie variant output path.")
    return output
