"""Lazy local sentence-transformer embedding provider."""

import importlib
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, cast

from travelmovieai.core.exceptions import DependencyUnavailableError, PipelineStageError


class TextEmbeddingProvider(Protocol):
    backend: str
    model: str

    @property
    def dimensions(self) -> int | None: ...

    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...

    def release(self) -> None: ...


class SentenceTransformerEmbeddingProvider:
    """Load sentence-transformers only when embeddings are actually requested."""

    backend = "sentence-transformers"

    def __init__(
        self,
        *,
        model: str,
        cache_dir: Path,
        device: str = "auto",
        allow_download: bool = True,
        batch_size: int = 32,
    ) -> None:
        self.model = model
        self.cache_dir = cache_dir.expanduser().resolve()
        self.device = device
        self.allow_download = allow_download
        self.batch_size = max(1, batch_size)
        self._runtime: Any | None = None
        self._dimensions: int | None = None

    @property
    def dimensions(self) -> int | None:
        return self._dimensions

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        normalized = [" ".join(text.split()) for text in texts]
        if not normalized:
            return []
        runtime = self._load()
        try:
            encoded = runtime.encode(
                normalized,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise PipelineStageError("Local sentence embedding inference failed.") from error

        vectors = _validated_vectors(encoded, expected_count=len(normalized))
        dimensions = len(vectors[0])
        if self._dimensions is not None and dimensions != self._dimensions:
            raise PipelineStageError(
                "Sentence embedding dimensions changed during one provider session."
            )
        self._dimensions = dimensions
        return vectors

    def release(self) -> None:
        self._runtime = None
        try:
            import torch
        except ImportError:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load(self) -> Any:
        if self._runtime is not None:
            return self._runtime
        try:
            sentence_transformers = importlib.import_module("sentence_transformers")
            sentence_transformer = sentence_transformers.SentenceTransformer
        except (ImportError, AttributeError) as error:
            raise DependencyUnavailableError(
                'Sentence embeddings require the optional "embeddings" dependency group.'
            ) from error

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            runtime = sentence_transformer(
                self.model,
                device=_resolve_device(self.device),
                cache_folder=str(self.cache_dir),
                local_files_only=not self.allow_download,
                trust_remote_code=False,
            )
            dimensions = runtime.get_sentence_embedding_dimension()
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            mode = "offline cache" if not self.allow_download else "local model"
            raise DependencyUnavailableError(
                f"Could not load sentence embedding model from the {mode}."
            ) from error
        if not isinstance(dimensions, int) or dimensions <= 0:
            raise PipelineStageError("Sentence embedding model reported invalid dimensions.")
        self._runtime = runtime
        self._dimensions = dimensions
        return runtime


def _resolve_device(configured: str) -> str:
    if configured == "cpu" or configured == "directml":
        return "cpu"
    if configured == "cuda":
        return "cuda"
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _validated_vectors(value: object, *, expected_count: int) -> list[list[float]]:
    try:
        rows = cast(Sequence[Sequence[object]], value)
        vectors = [[round(float(cast(Any, component)), 8) for component in row] for row in rows]
    except (TypeError, ValueError) as error:
        raise PipelineStageError("Sentence embedding provider returned invalid vectors.") from error
    if len(vectors) != expected_count or not vectors or not vectors[0]:
        raise PipelineStageError("Sentence embedding provider returned an invalid vector count.")
    dimensions = len(vectors[0])
    if any(len(vector) != dimensions for vector in vectors):
        raise PipelineStageError("Sentence embedding vectors have inconsistent dimensions.")
    if any(not math.isfinite(component) for vector in vectors for component in vector):
        raise PipelineStageError("Sentence embedding vectors contain non-finite values.")
    return vectors
