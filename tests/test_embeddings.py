from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest

from travelmovieai.analysis.embeddings import embed_scenes, semantic_similarity
from travelmovieai.application.context import ProjectContext
from travelmovieai.core.config import Settings
from travelmovieai.domain.enums import MediaType, StageStatus
from travelmovieai.domain.models import EmbeddingAnalysisReport, MediaAsset, Scene
from travelmovieai.infrastructure.database import MediaAssetRepository
from travelmovieai.pipeline.stages.embeddings import EmbeddingsStage


def test_embeddings_are_deterministic_and_semantically_comparable() -> None:
    first = Scene(
        asset_id=_uuid(1),
        start_seconds=0,
        end_seconds=2,
        caption="Family walking through a sunny city",
        metadata={"tags": ["family", "city", "walking"]},
    )
    second = Scene(
        asset_id=_uuid(2),
        start_seconds=0,
        end_seconds=2,
        caption="Family walking in the city",
        metadata={"tags": ["family", "city", "walking"]},
    )

    first_report, first_updated = embed_scenes([first, second])
    second_report, _ = embed_scenes([first, second])

    assert first_report.embeddings[0].vector == second_report.embeddings[0].vector
    similarity = semantic_similarity(first_updated[0], first_updated[1])
    assert similarity is not None and similarity > 0.5


def test_embeddings_stage_reuses_valid_cache(tmp_path: Path) -> None:
    context = ProjectContext(
        input_path=tmp_path,
        workspace=tmp_path / "workspace",
        settings=Settings(embedding_index="disabled"),
    )
    context.prepare()
    repository = MediaAssetRepository(context.database_path)
    repository.initialize()
    asset = MediaAsset(
        id=_uuid(3),
        path=tmp_path / "mountain.mp4",
        relative_path=Path("mountain.mp4"),
        media_type=MediaType.VIDEO,
        extension=".mp4",
        size_bytes=1,
        modified_at=datetime.now(UTC),
        modified_ns=1,
        duration_seconds=2,
    )
    repository.synchronize([asset], datetime.now(UTC))
    repository.synchronize_scenes(
        [
            Scene(
                asset_id=asset.id,
                start_seconds=0,
                end_seconds=2,
                caption="Mountain view",
                metadata={"tags": ["mountains"]},
            )
        ]
    )

    first = EmbeddingsStage().run(context)
    second = EmbeddingsStage().run(context)

    assert first.skipped is False
    assert second.skipped is True
    assert first.execution.provider == "feature-hash-v1"
    assert second.execution.provider == "feature-hash-v1"
    assert (context.artifacts_dir / "embeddings.json").is_file()


def test_embeddings_stage_uses_configured_sentence_provider_and_releases_it(
    tmp_path: Path,
) -> None:
    context = ProjectContext(
        input_path=tmp_path,
        workspace=tmp_path / "workspace",
        settings=Settings(
            embedding_backend="sentence-transformers",
            embedding_model="local-test-model",
            embedding_index="disabled",
            allow_model_download=False,
        ),
    )
    context.prepare()
    repository = MediaAssetRepository(context.database_path)
    repository.initialize()
    asset = MediaAsset(
        id=_uuid(4),
        path=tmp_path / "sea.mp4",
        relative_path=Path("sea.mp4"),
        media_type=MediaType.VIDEO,
        extension=".mp4",
        size_bytes=1,
        modified_at=datetime.now(UTC),
        modified_ns=1,
        duration_seconds=2,
    )
    repository.synchronize([asset], datetime.now(UTC))
    repository.synchronize_scenes(
        [
            Scene(
                id=_uuid(5),
                asset_id=asset.id,
                start_seconds=0,
                end_seconds=2,
                caption="Sea view",
            )
        ]
    )

    class FakeProvider:
        backend = "sentence-transformers"
        model = "local-test-model"
        dimensions = 3
        released = False

        def encode(self, texts: list[str]) -> list[list[float]]:
            assert texts == ["Sea view"]
            return [[1, 0, 0]]

        def release(self) -> None:
            self.released = True

    provider = FakeProvider()
    result = EmbeddingsStage(provider_factory=lambda _: provider).run(context)
    report = EmbeddingAnalysisReport.model_validate_json(
        (context.artifacts_dir / "embeddings.json").read_text(encoding="utf-8")
    )

    assert result.status is StageStatus.COMPLETED
    assert result.execution.provider == "sentence-transformers"
    assert result.execution.model == "local-test-model"
    assert report.backend == "sentence-transformers"
    assert report.model == "local-test-model"
    assert report.dimensions == 3
    assert provider.released is True
    assert repository.list_scenes()[0].metadata["embedding_model"] == "local-test-model"
    assert "semantic_embedding" not in repository.list_scenes()[0].metadata


def test_embeddings_stage_removes_stale_artifacts_without_scenes(tmp_path: Path) -> None:
    context = ProjectContext(
        input_path=tmp_path,
        workspace=tmp_path / "workspace",
        settings=Settings(),
    )
    context.prepare()
    MediaAssetRepository(context.database_path).initialize()
    stale = [
        context.artifacts_dir / "embeddings.json",
        context.artifacts_dir / "embeddings.cache.json",
        context.artifacts_dir / "embeddings.faiss",
        context.artifacts_dir / "embeddings.index.json",
    ]
    for path in stale:
        path.write_text("stale", encoding="utf-8")

    result = EmbeddingsStage().run(context)

    assert result.status is StageStatus.NO_INPUT
    assert not any(path.exists() for path in stale)


def test_semantic_similarity_supports_model_dimensions() -> None:
    first = Scene(
        asset_id=_uuid(6),
        start_seconds=0,
        end_seconds=1,
        metadata={"semantic_embedding": [1, 1, 0]},
    )
    second = Scene(
        asset_id=_uuid(7),
        start_seconds=0,
        end_seconds=1,
        metadata={"semantic_embedding": [2, 2, 0]},
    )

    assert semantic_similarity(first, second) == pytest.approx(1)


def _uuid(value: int) -> UUID:
    return UUID(int=value)
