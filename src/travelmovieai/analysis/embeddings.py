"""Deterministic local semantic features for scenes and archive search."""

import hashlib
import math
import re
from datetime import UTC, datetime

from travelmovieai.core.exceptions import PipelineStageError
from travelmovieai.domain.models import (
    EmbeddingAnalysisReport,
    Scene,
    SceneEmbedding,
)
from travelmovieai.infrastructure.embeddings import TextEmbeddingProvider

EMBEDDING_BACKEND = "feature-hash-v1"
EMBEDDING_DIMENSIONS = 64


def embed_scenes(
    scenes: list[Scene],
    provider: TextEmbeddingProvider | None = None,
) -> tuple[EmbeddingAnalysisReport, list[Scene]]:
    texts = [_scene_text(scene) for scene in scenes]
    if provider is None:
        backend = EMBEDDING_BACKEND
        model: str | None = None
        dimensions = EMBEDDING_DIMENSIONS
        vectors = [feature_hash_embedding(text) for text in texts]
    else:
        backend = provider.backend
        model = provider.model
        vectors = provider.encode(texts)
        dimensions = provider.dimensions or 0
        if len(vectors) != len(scenes) or dimensions <= 0:
            raise PipelineStageError(
                "Embedding provider returned an invalid scene vector contract."
            )
    embeddings: list[SceneEmbedding] = []
    updated: list[Scene] = []
    for scene, vector in zip(scenes, vectors, strict=True):
        embeddings.append(SceneEmbedding(scene_id=scene.id, vector=vector))
        updated.append(
            scene.model_copy(
                update={
                    "metadata": {
                        **scene.metadata,
                        "semantic_embedding": vector,
                        "embedding_backend": backend,
                        "embedding_model": model,
                    }
                }
            )
        )
    return (
        EmbeddingAnalysisReport(
            created_at=datetime.now(UTC),
            backend=backend,
            model=model,
            dimensions=dimensions,
            embeddings=embeddings,
        ),
        updated,
    )


def semantic_similarity(first: Scene, second: Scene) -> float | None:
    first_vector = first.metadata.get("semantic_embedding")
    second_vector = second.metadata.get("semantic_embedding")
    if not isinstance(first_vector, list) or not isinstance(second_vector, list):
        return None
    if not first_vector or len(first_vector) != len(second_vector):
        return None
    try:
        first_values = [float(value) for value in first_vector]
        second_values = [float(value) for value in second_vector]
        first_norm = math.sqrt(sum(value * value for value in first_values))
        second_norm = math.sqrt(sum(value * value for value in second_values))
        if first_norm == 0 or second_norm == 0:
            return 0.0
        return max(
            0.0,
            min(
                1.0,
                sum(a * b for a, b in zip(first_values, second_values, strict=True))
                / (first_norm * second_norm),
            ),
        )
    except (TypeError, ValueError):
        return None


def scene_embedding_text(scene: Scene) -> str:
    tags = scene.metadata.get("tags", [])
    resolved_tags = tags if isinstance(tags, list) else []
    values: list[object] = [
        scene.caption or "",
        scene.transcript or "",
        scene.metadata.get("detailed_description", ""),
        scene.metadata.get("location_type", ""),
        scene.metadata.get("activity", ""),
        scene.metadata.get("emotion", ""),
        *resolved_tags,
    ]
    return " ".join(str(value) for value in values if str(value).strip())


def _scene_text(scene: Scene) -> str:
    return scene_embedding_text(scene)


def feature_hash_embedding(text: str) -> list[float]:
    vector = [0.0] * EMBEDDING_DIMENSIONS
    tokens = re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE)
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % EMBEDDING_DIMENSIONS
        vector[index] += 1.0 if digest[4] & 1 else -1.0
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [round(value / norm, 8) for value in vector]
