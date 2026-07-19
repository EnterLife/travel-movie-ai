"""Typed FAISS persistence and search for local scene embeddings."""

import importlib
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from pydantic import ValidationError

from travelmovieai.core.exceptions import DependencyUnavailableError, PipelineStageError
from travelmovieai.domain.models import (
    EmbeddingAnalysisReport,
    SemanticIndexManifest,
    SemanticSearchHit,
)
from travelmovieai.infrastructure.artifacts import artifact_fingerprint, write_json_atomic


def build_semantic_index(
    report: EmbeddingAnalysisReport,
    *,
    index_path: Path,
    manifest_path: Path,
) -> SemanticIndexManifest:
    vectors = _report_matrix(report)
    faiss = _load_faiss()
    index = faiss.IndexFlatIP(report.dimensions)
    if len(vectors):
        faiss.normalize_L2(vectors)
        index.add(vectors)

    resolved_index = index_path.expanduser().resolve()
    resolved_index.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved_index.with_name(f".{resolved_index.name}.{uuid4().hex}.tmp")
    try:
        serialized = faiss.serialize_index(index)
        temporary.write_bytes(serialized.tobytes())
        if not temporary.is_file() or temporary.stat().st_size <= 0:
            raise PipelineStageError("FAISS did not produce a valid index file.")
        os.replace(temporary, resolved_index)
    except (OSError, RuntimeError) as error:
        raise PipelineStageError("Could not persist the local FAISS scene index.") from error
    finally:
        temporary.unlink(missing_ok=True)

    manifest = SemanticIndexManifest(
        created_at=datetime.now(UTC),
        backend=report.backend,
        model=report.model,
        dimensions=report.dimensions,
        scene_ids=[embedding.scene_id for embedding in report.embeddings],
        index_path=resolved_index,
        source_fingerprint=artifact_fingerprint(
            report.backend,
            report.model,
            report.dimensions,
            report.embeddings,
        ),
    )
    write_json_atomic(manifest_path, manifest)
    return manifest


def search_semantic_index(
    query_vector: list[float],
    *,
    index_path: Path,
    manifest_path: Path,
    expected_report: EmbeddingAnalysisReport,
    limit: int = 10,
) -> list[SemanticSearchHit]:
    if limit < 1 or limit > 1000:
        raise ValueError("Semantic search limit must be between 1 and 1000.")
    manifest = _read_manifest(manifest_path)
    _validate_manifest_source(manifest, expected_report)
    resolved_index = index_path.expanduser().resolve()
    if manifest.index_path.expanduser().resolve() != resolved_index:
        raise PipelineStageError("Semantic index manifest references an unexpected index path.")
    vector = _query_matrix(query_vector, manifest.dimensions)
    faiss = _load_faiss()
    try:
        serialized = np.frombuffer(resolved_index.read_bytes(), dtype=np.uint8)
        index = faiss.deserialize_index(serialized)
    except (OSError, RuntimeError) as error:
        raise PipelineStageError("Could not read the local FAISS scene index.") from error
    if index.d != manifest.dimensions or index.ntotal != len(manifest.scene_ids):
        raise PipelineStageError("FAISS index does not match its typed manifest.")
    if not manifest.scene_ids:
        return []
    faiss.normalize_L2(vector)
    requested = min(limit, len(manifest.scene_ids))
    scores, positions = index.search(vector, requested)
    hits: list[SemanticSearchHit] = []
    for rank, (position, score) in enumerate(
        zip(positions[0].tolist(), scores[0].tolist(), strict=True),
        start=1,
    ):
        if position < 0 or position >= len(manifest.scene_ids):
            continue
        hits.append(
            SemanticSearchHit(
                scene_id=manifest.scene_ids[position],
                score=max(-1.0, min(1.0, float(score))),
                rank=rank,
            )
        )
    return hits


def _report_matrix(report: EmbeddingAnalysisReport) -> np.ndarray[Any, np.dtype[np.float32]]:
    if any(len(item.vector) != report.dimensions for item in report.embeddings):
        raise PipelineStageError("Embedding report contains inconsistent vector dimensions.")
    scene_ids = [item.scene_id for item in report.embeddings]
    if len(scene_ids) != len(set(scene_ids)):
        raise PipelineStageError("Embedding report contains duplicate scene identifiers.")
    if not report.embeddings:
        return np.empty((0, report.dimensions), dtype=np.float32)
    matrix = np.asarray([item.vector for item in report.embeddings], dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise PipelineStageError("Embedding report contains non-finite values.")
    return matrix


def _query_matrix(
    vector: list[float],
    dimensions: int,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    if len(vector) != dimensions:
        raise PipelineStageError("Semantic query dimensions do not match the FAISS index.")
    if not vector or any(not math.isfinite(float(value)) for value in vector):
        raise PipelineStageError("Semantic query vector is invalid.")
    matrix = np.asarray([vector], dtype=np.float32)
    if float(np.linalg.norm(matrix[0])) == 0:
        raise PipelineStageError("Semantic query vector must not be empty or zero.")
    return matrix


def _read_manifest(path: Path) -> SemanticIndexManifest:
    try:
        return SemanticIndexManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as error:
        raise PipelineStageError("Could not read the semantic index manifest.") from error


def _validate_manifest_source(
    manifest: SemanticIndexManifest,
    report: EmbeddingAnalysisReport,
) -> None:
    expected_fingerprint = artifact_fingerprint(
        report.backend,
        report.model,
        report.dimensions,
        report.embeddings,
    )
    expected_scene_ids = [embedding.scene_id for embedding in report.embeddings]
    if (
        manifest.backend != report.backend
        or manifest.model != report.model
        or manifest.dimensions != report.dimensions
        or manifest.scene_ids != expected_scene_ids
        or manifest.source_fingerprint != expected_fingerprint
    ):
        raise PipelineStageError(
            "FAISS index manifest does not match the current embeddings artifact."
        )


def _load_faiss() -> Any:
    try:
        return importlib.import_module("faiss")
    except ImportError as error:
        raise DependencyUnavailableError(
            'FAISS indexing requires the optional "embeddings" dependency group.'
        ) from error
