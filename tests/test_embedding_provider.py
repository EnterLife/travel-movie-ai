import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from travelmovieai.core.exceptions import DependencyUnavailableError, PipelineStageError
from travelmovieai.infrastructure.embeddings import SentenceTransformerEmbeddingProvider


class _FakeSentenceTransformer:
    init_kwargs: dict[str, object] = {}
    encode_kwargs: dict[str, object] = {}

    def __init__(self, model: str, **kwargs: object) -> None:
        self.model = model
        type(self).init_kwargs = kwargs

    def get_sentence_embedding_dimension(self) -> int:
        return 3

    def encode(self, texts: list[str], **kwargs: object) -> list[list[float]]:
        type(self).encode_kwargs = kwargs
        return [[1.0, 0.0, 0.0] for _ in texts]


def test_sentence_provider_is_lazy_and_honors_offline_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=_FakeSentenceTransformer),
    )
    provider = SentenceTransformerEmbeddingProvider(
        model="local/model",
        cache_dir=tmp_path / "models",
        device="cpu",
        allow_download=False,
        batch_size=7,
    )

    assert provider.encode([]) == []
    assert provider.dimensions is None
    vectors = provider.encode(["  mountain   view ", "sea"])

    assert vectors == [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    assert provider.dimensions == 3
    assert _FakeSentenceTransformer.init_kwargs["local_files_only"] is True
    assert _FakeSentenceTransformer.init_kwargs["trust_remote_code"] is False
    assert _FakeSentenceTransformer.encode_kwargs["batch_size"] == 7
    assert _FakeSentenceTransformer.encode_kwargs["normalize_embeddings"] is True


def test_sentence_provider_reports_missing_offline_model_without_path_leak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingSentenceTransformer:
        def __init__(self, *_: object, **__: object) -> None:
            raise OSError(f"missing {tmp_path / 'private-model'}")

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FailingSentenceTransformer),
    )
    provider = SentenceTransformerEmbeddingProvider(
        model="private-model",
        cache_dir=tmp_path,
        allow_download=False,
    )

    with pytest.raises(DependencyUnavailableError, match="offline cache") as raised:
        provider.encode(["scene"])

    assert str(tmp_path) not in str(raised.value)


def test_sentence_provider_rejects_inconsistent_vectors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InvalidSentenceTransformer(_FakeSentenceTransformer):
        def encode(self, texts: list[str], **kwargs: object) -> list[list[float]]:
            return [[1.0, 0.0], [1.0]]

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=InvalidSentenceTransformer),
    )
    provider = SentenceTransformerEmbeddingProvider(
        model="local/model",
        cache_dir=tmp_path,
    )

    with pytest.raises(PipelineStageError, match="inconsistent dimensions"):
        provider.encode(["one", "two"])
