from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from travelmovieai.analysis.embeddings import embed_scenes, semantic_similarity
from travelmovieai.application.context import ProjectContext
from travelmovieai.core.config import Settings
from travelmovieai.domain.enums import MediaType
from travelmovieai.domain.models import MediaAsset, Scene
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
        settings=Settings(),
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
    assert (context.artifacts_dir / "embeddings.json").is_file()


def _uuid(value: int) -> UUID:
    return UUID(int=value)
