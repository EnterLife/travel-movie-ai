from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest

from travelmovieai.analysis.embeddings import EMBEDDING_BACKEND, feature_hash_embedding
from travelmovieai.application.context import ProjectContext
from travelmovieai.application.semantic_search import search_project_scenes
from travelmovieai.core.config import Settings
from travelmovieai.core.exceptions import PipelineStageError
from travelmovieai.domain.models import EmbeddingAnalysisReport, SceneEmbedding
from travelmovieai.infrastructure.artifacts import write_json_atomic
from travelmovieai.infrastructure.semantic_index import (
    build_semantic_index,
    search_semantic_index,
)

pytest.importorskip("faiss")


def test_faiss_index_round_trip_returns_ranked_scene_ids(tmp_path: Path) -> None:
    report = EmbeddingAnalysisReport(
        created_at=datetime.now(UTC),
        backend="test",
        model="tiny",
        dimensions=3,
        embeddings=[
            SceneEmbedding(scene_id=_uuid(1), vector=[1, 0, 0]),
            SceneEmbedding(scene_id=_uuid(2), vector=[0.8, 0.2, 0]),
            SceneEmbedding(scene_id=_uuid(3), vector=[0, 1, 0]),
        ],
    )
    index_path = tmp_path / "индекс с пробелом.faiss"
    manifest_path = tmp_path / "index.json"

    manifest = build_semantic_index(
        report,
        index_path=index_path,
        manifest_path=manifest_path,
    )
    hits = search_semantic_index(
        [1, 0, 0],
        index_path=index_path,
        manifest_path=manifest_path,
        expected_report=report,
        limit=2,
    )

    assert manifest.scene_ids == [_uuid(1), _uuid(2), _uuid(3)]
    assert index_path.is_file()
    assert manifest_path.is_file()
    assert [hit.scene_id for hit in hits] == [_uuid(1), _uuid(2)]
    assert hits[0].score == pytest.approx(1)


def test_faiss_index_rejects_duplicate_scene_identifiers(tmp_path: Path) -> None:
    report = EmbeddingAnalysisReport(
        created_at=datetime.now(UTC),
        backend="test",
        dimensions=2,
        embeddings=[
            SceneEmbedding(scene_id=_uuid(1), vector=[1, 0]),
            SceneEmbedding(scene_id=_uuid(1), vector=[0, 1]),
        ],
    )

    with pytest.raises(PipelineStageError, match="duplicate scene"):
        build_semantic_index(
            report,
            index_path=tmp_path / "index.faiss",
            manifest_path=tmp_path / "index.json",
        )


def test_faiss_search_rejects_manifest_path_substitution(tmp_path: Path) -> None:
    report = EmbeddingAnalysisReport(
        created_at=datetime.now(UTC),
        backend="test",
        dimensions=2,
        embeddings=[SceneEmbedding(scene_id=_uuid(1), vector=[1, 0])],
    )
    index_path = tmp_path / "index.faiss"
    manifest_path = tmp_path / "index.json"
    build_semantic_index(report, index_path=index_path, manifest_path=manifest_path)

    with pytest.raises(PipelineStageError, match="unexpected index path"):
        search_semantic_index(
            [1, 0],
            index_path=tmp_path / "other.faiss",
            manifest_path=manifest_path,
            expected_report=report,
        )


def test_faiss_search_rejects_zero_query(tmp_path: Path) -> None:
    report = EmbeddingAnalysisReport(
        created_at=datetime.now(UTC),
        backend="test",
        dimensions=2,
        embeddings=[SceneEmbedding(scene_id=_uuid(1), vector=[1, 0])],
    )
    index_path = tmp_path / "index.faiss"
    manifest_path = tmp_path / "index.json"
    build_semantic_index(report, index_path=index_path, manifest_path=manifest_path)

    with pytest.raises(PipelineStageError, match="must not be empty or zero"):
        search_semantic_index(
            [0, 0],
            index_path=index_path,
            manifest_path=manifest_path,
            expected_report=report,
        )


def test_project_semantic_search_uses_persisted_feature_hash_index(tmp_path: Path) -> None:
    context = ProjectContext(
        input_path=tmp_path,
        workspace=tmp_path / "workspace",
        settings=Settings(),
    )
    context.prepare()
    report = EmbeddingAnalysisReport(
        created_at=datetime.now(UTC),
        backend=EMBEDDING_BACKEND,
        dimensions=64,
        embeddings=[
            SceneEmbedding(
                scene_id=_uuid(1),
                vector=feature_hash_embedding("mountain sunrise"),
            ),
            SceneEmbedding(
                scene_id=_uuid(2),
                vector=feature_hash_embedding("city night"),
            ),
        ],
    )
    write_json_atomic(context.artifacts_dir / "embeddings.json", report)
    build_semantic_index(
        report,
        index_path=context.artifacts_dir / "embeddings.faiss",
        manifest_path=context.artifacts_dir / "embeddings.index.json",
    )

    result = search_project_scenes(context, "  mountain   sunrise ", limit=1)

    assert result.query == "mountain sunrise"
    assert result.hits[0].scene_id == _uuid(1)
    assert result.hits[0].score == pytest.approx(1)


def test_project_semantic_search_requires_index_artifacts(tmp_path: Path) -> None:
    context = ProjectContext(
        input_path=tmp_path,
        workspace=tmp_path / "workspace",
        settings=Settings(),
    )
    context.prepare()
    write_json_atomic(
        context.artifacts_dir / "embeddings.json",
        EmbeddingAnalysisReport(
            created_at=datetime.now(UTC),
            backend=EMBEDDING_BACKEND,
            dimensions=64,
        ),
    )

    with pytest.raises(PipelineStageError, match="completed FAISS"):
        search_project_scenes(context, "mountain")


def test_project_semantic_search_rejects_stale_index_manifest(tmp_path: Path) -> None:
    context = ProjectContext(
        input_path=tmp_path,
        workspace=tmp_path / "workspace",
        settings=Settings(),
    )
    context.prepare()
    indexed = EmbeddingAnalysisReport(
        created_at=datetime.now(UTC),
        backend=EMBEDDING_BACKEND,
        dimensions=64,
        embeddings=[
            SceneEmbedding(
                scene_id=_uuid(1),
                vector=feature_hash_embedding("mountain sunrise"),
            )
        ],
    )
    write_json_atomic(context.artifacts_dir / "embeddings.json", indexed)
    build_semantic_index(
        indexed,
        index_path=context.artifacts_dir / "embeddings.faiss",
        manifest_path=context.artifacts_dir / "embeddings.index.json",
    )
    stale_report = indexed.model_copy(
        update={
            "embeddings": [
                SceneEmbedding(
                    scene_id=_uuid(2),
                    vector=feature_hash_embedding("city night"),
                )
            ]
        }
    )
    write_json_atomic(context.artifacts_dir / "embeddings.json", stale_report)

    with pytest.raises(PipelineStageError, match="does not match"):
        search_project_scenes(context, "mountain")


def _uuid(value: int) -> UUID:
    return UUID(int=value)
