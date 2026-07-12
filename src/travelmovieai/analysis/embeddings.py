"""Deterministic local semantic features for scenes and archive search."""

import hashlib
import math
import re
from datetime import UTC, datetime

from travelmovieai.domain.models import (
    EmbeddingAnalysisReport,
    Scene,
    SceneEmbedding,
)

EMBEDDING_BACKEND = "feature-hash-v1"
EMBEDDING_DIMENSIONS = 64


def embed_scenes(
    scenes: list[Scene],
) -> tuple[EmbeddingAnalysisReport, list[Scene]]:
    embeddings: list[SceneEmbedding] = []
    updated: list[Scene] = []
    for scene in scenes:
        vector = _feature_hash(_scene_text(scene))
        embeddings.append(SceneEmbedding(scene_id=scene.id, vector=vector))
        updated.append(
            scene.model_copy(
                update={
                    "metadata": {
                        **scene.metadata,
                        "semantic_embedding": vector,
                        "embedding_backend": EMBEDDING_BACKEND,
                    }
                }
            )
        )
    return (
        EmbeddingAnalysisReport(
            created_at=datetime.now(UTC),
            backend=EMBEDDING_BACKEND,
            dimensions=EMBEDDING_DIMENSIONS,
            embeddings=embeddings,
        ),
        updated,
    )


def semantic_similarity(first: Scene, second: Scene) -> float | None:
    first_vector = first.metadata.get("semantic_embedding")
    second_vector = second.metadata.get("semantic_embedding")
    if not isinstance(first_vector, list) or not isinstance(second_vector, list):
        return None
    if len(first_vector) != EMBEDDING_DIMENSIONS or len(second_vector) != EMBEDDING_DIMENSIONS:
        return None
    try:
        return max(
            0.0,
            min(
                1.0,
                sum(float(a) * float(b) for a, b in zip(first_vector, second_vector, strict=True)),
            ),
        )
    except (TypeError, ValueError):
        return None


def _scene_text(scene: Scene) -> str:
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


def _feature_hash(text: str) -> list[float]:
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
