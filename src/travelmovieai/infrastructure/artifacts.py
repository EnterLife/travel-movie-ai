"""Atomic serialization helpers for pipeline artifacts."""

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ValidationError

from travelmovieai.domain.enums import PipelineStage
from travelmovieai.domain.models import StageCacheManifest


def write_json_atomic(path: Path, model: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary_path.write_text(
            model.model_dump_json(indent=2),
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def artifact_fingerprint(*parts: object) -> str:
    serialized = json.dumps(
        _normalize(parts),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def write_stage_cache_manifest(
    path: Path,
    *,
    stage: PipelineStage,
    artifact_schema_version: str,
    input_fingerprint: str,
    config_fingerprint: str,
    artifacts: list[Path],
) -> StageCacheManifest:
    manifest = StageCacheManifest(
        stage=stage,
        artifact_schema_version=artifact_schema_version,
        input_fingerprint=input_fingerprint,
        config_fingerprint=config_fingerprint,
        created_at=datetime.now(UTC),
        artifacts=artifacts,
    )
    write_json_atomic(path, manifest)
    return manifest


def stage_cache_manifest_matches(
    path: Path,
    *,
    stage: PipelineStage,
    artifact_schema_version: str,
    input_fingerprint: str,
    config_fingerprint: str,
    artifacts: list[Path],
) -> bool:
    if not path.is_file() or any(not artifact.exists() for artifact in artifacts):
        return False
    try:
        manifest = StageCacheManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError):
        return False
    return (
        manifest.stage == stage
        and manifest.artifact_schema_version == artifact_schema_version
        and manifest.input_fingerprint == input_fingerprint
        and manifest.config_fingerprint == config_fingerprint
    )


def _normalize(value: object) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_normalize(item) for item in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)
