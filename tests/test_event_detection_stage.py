from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

from travelmovieai.application.context import ProjectContext
from travelmovieai.core.config import Settings
from travelmovieai.domain.enums import MediaType, StageStatus
from travelmovieai.domain.models import (
    EmbeddingAnalysisReport,
    EventDetectionReport,
    MediaAsset,
    Scene,
    SceneEmbedding,
)
from travelmovieai.infrastructure.artifacts import write_json_atomic
from travelmovieai.infrastructure.database import MediaAssetRepository
from travelmovieai.pipeline.stages.event_detection import EventDetectionStage


def test_event_detection_cache_invalidates_when_gps_changes(tmp_path: Path) -> None:
    context, repository, assets, scenes = _seed_project(tmp_path)
    nearby = [
        assets[0].model_copy(update={"latitude": 43.5855, "longitude": 39.7231}),
        assets[1].model_copy(update={"latitude": 43.5880, "longitude": 39.7200}),
    ]
    repository.synchronize(nearby, datetime.now(UTC))

    first = EventDetectionStage().run(context)
    cached = EventDetectionStage().run(context)
    distant = [
        nearby[0],
        nearby[1].model_copy(update={"latitude": 44.6167, "longitude": 33.5254}),
    ]
    repository.synchronize(distant, datetime.now(UTC))
    changed = EventDetectionStage().run(context)
    report = _event_report(context)

    assert first.status is StageStatus.COMPLETED
    assert cached.status is StageStatus.CACHED
    assert changed.status is StageStatus.COMPLETED
    assert len(scenes) == 2
    assert len(report.events) == 2


def test_event_detection_cache_invalidates_when_embedding_changes(tmp_path: Path) -> None:
    context, repository, _, scenes = _seed_project(tmp_path)
    orthogonal = [
        scenes[0].model_copy(
            update={"metadata": {**scenes[0].metadata, "semantic_embedding": [1.0, 0.0]}}
        ),
        scenes[1].model_copy(
            update={"metadata": {**scenes[1].metadata, "semantic_embedding": [0.0, 1.0]}}
        ),
    ]
    repository.synchronize_scenes(orthogonal)

    first = EventDetectionStage().run(context)
    matching = [
        orthogonal[0],
        orthogonal[1].model_copy(
            update={"metadata": {**orthogonal[1].metadata, "semantic_embedding": [1.0, 0.0]}}
        ),
    ]
    repository.synchronize_scenes(matching)
    changed = EventDetectionStage().run(context)
    cached = EventDetectionStage().run(context)
    report = _event_report(context)

    assert first.status is StageStatus.COMPLETED
    assert changed.status is StageStatus.COMPLETED
    assert cached.status is StageStatus.CACHED
    assert len(report.events) == 1


def test_event_detection_reads_artifact_vectors_without_persisting_them(
    tmp_path: Path,
) -> None:
    context, repository, _, scenes = _seed_project(tmp_path)
    embedding_path = context.artifacts_dir / "embeddings.json"

    def write_embeddings(vectors: list[list[float]]) -> None:
        write_json_atomic(
            embedding_path,
            EmbeddingAnalysisReport(
                created_at=datetime.now(UTC),
                backend="test",
                dimensions=2,
                embeddings=[
                    SceneEmbedding(scene_id=scene.id, vector=vector)
                    for scene, vector in zip(scenes, vectors, strict=True)
                ],
            ),
        )

    write_embeddings([[1.0, 0.0], [0.0, 1.0]])
    first = EventDetectionStage().run(context)
    first_report = _event_report(context)
    write_embeddings([[1.0, 0.0], [1.0, 0.0]])
    changed = EventDetectionStage().run(context)
    cached = EventDetectionStage().run(context)
    changed_report = _event_report(context)

    assert first.status is StageStatus.COMPLETED
    assert changed.status is StageStatus.COMPLETED
    assert cached.status is StageStatus.CACHED
    assert len(first_report.events) == 2
    assert len(changed_report.events) == 1
    assert all("semantic_embedding" not in scene.metadata for scene in repository.list_scenes())


def test_event_detection_reuses_valid_empty_result(tmp_path: Path) -> None:
    context = ProjectContext(
        input_path=tmp_path / "media",
        workspace=tmp_path / "workspace",
        settings=Settings(),
    )
    context.prepare()

    first = EventDetectionStage().run(context)
    second = EventDetectionStage().run(context)

    assert first.status is StageStatus.NO_INPUT
    assert second.status is StageStatus.CACHED


def _seed_project(
    tmp_path: Path,
) -> tuple[ProjectContext, MediaAssetRepository, list[MediaAsset], list[Scene]]:
    context = ProjectContext(
        input_path=tmp_path / "media",
        workspace=tmp_path / "workspace",
        settings=Settings(),
    )
    context.prepare()
    created_at = datetime(2026, 1, 1, 10, tzinfo=UTC)
    assets = [
        MediaAsset(
            id=UUID(f"00000000-0000-0000-0000-{index:012d}"),
            path=tmp_path / "media" / f"scene-{index}.mp4",
            relative_path=Path(f"scene-{index}.mp4"),
            media_type=MediaType.VIDEO,
            extension=".mp4",
            size_bytes=1,
            modified_at=created_at + timedelta(minutes=index * 20),
            modified_ns=1,
            created_at=created_at + timedelta(minutes=index * 20),
            duration_seconds=5,
        )
        for index in (1, 2)
    ]
    scenes = [
        Scene(
            id=UUID(f"00000000-0000-0000-0001-{index:012d}"),
            asset_id=asset.id,
            start_seconds=0,
            end_seconds=5,
            caption=f"Scene {index}",
            importance_score=70,
            metadata={
                "location_type": "beach" if index == 1 else "museum",
                "activity": "walking" if index == 1 else "sightseeing",
                "landmarks": [],
            },
        )
        for index, asset in enumerate(assets, start=1)
    ]
    repository = MediaAssetRepository(context.database_path)
    repository.initialize()
    repository.synchronize(assets, datetime.now(UTC))
    repository.synchronize_scenes(scenes)
    return context, repository, assets, scenes


def _event_report(context: ProjectContext) -> EventDetectionReport:
    return EventDetectionReport.model_validate_json(
        (context.artifacts_dir / "events.json").read_text(encoding="utf-8")
    )
