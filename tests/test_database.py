import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy.pool import NullPool

from travelmovieai.core.exceptions import PipelineStageError
from travelmovieai.domain.enums import ActivityType, LocationType, MediaType
from travelmovieai.domain.models import Event, MediaAsset, Scene
from travelmovieai.infrastructure import database
from travelmovieai.infrastructure.database import MediaAssetRepository


def _asset(path: Path, relative_path: str) -> MediaAsset:
    return MediaAsset(
        path=path.resolve(),
        relative_path=Path(relative_path),
        media_type=MediaType.VIDEO,
        extension=".mp4",
        size_bytes=5,
        modified_at=datetime(2026, 1, 1, tzinfo=UTC),
        modified_ns=123,
        duration_seconds=2.5,
    )


def test_repository_synchronizes_updates_and_deletions(tmp_path: Path) -> None:
    repository = MediaAssetRepository(tmp_path / "project.db")
    repository.initialize()
    first = _asset(tmp_path / "first.mp4", "first.mp4")
    second = _asset(tmp_path / "second.mp4", "second.mp4")
    scanned_at = datetime.now(UTC)

    repository.synchronize([first, second], scanned_at)
    updated = first.model_copy(update={"duration_seconds": 9.0})
    repository.synchronize([updated], scanned_at)

    assets = repository.list_assets()
    assert len(assets) == 1
    assert assets[0].id == first.id
    assert assets[0].duration_seconds == 9.0


def test_repository_does_not_retain_sqlite_connections_between_operations(
    tmp_path: Path,
) -> None:
    repository = MediaAssetRepository(tmp_path / "project.db")

    assert isinstance(repository._engine.pool, NullPool)

    repository.initialize()
    repository.list_assets()
    assert "NullPool" in repository._engine.pool.status()


def test_repository_persists_scenes_and_cascades_deleted_assets(tmp_path: Path) -> None:
    repository = MediaAssetRepository(tmp_path / "project.db")
    repository.initialize()
    asset = _asset(tmp_path / "first.mp4", "first.mp4")
    repository.synchronize([asset], datetime.now(UTC))
    scene = Scene(
        asset_id=asset.id,
        start_seconds=1,
        end_seconds=3,
        caption="City walk",
        importance_score=82,
        metadata={"cache_key": "test"},
    )

    repository.synchronize_scenes([scene])
    stored = repository.list_scenes()
    repository.synchronize([], datetime.now(UTC))

    assert stored == [scene]
    assert repository.list_scenes() == []


def test_repository_persists_detected_events(tmp_path: Path) -> None:
    repository = MediaAssetRepository(tmp_path / "project.db")
    repository.initialize()
    asset = _asset(tmp_path / "first.mp4", "first.mp4")
    repository.synchronize([asset], datetime.now(UTC))
    scene = Scene(asset_id=asset.id, start_seconds=0, end_seconds=2)
    repository.synchronize_scenes([scene])
    event = Event(
        title="City Exploration",
        scene_ids=[scene.id],
        summary="Walking through the city.",
        importance_score=84,
        start_at=datetime(2026, 1, 1, tzinfo=UTC),
        end_at=datetime(2026, 1, 1, 0, 0, 2, tzinfo=UTC),
        location_type=LocationType.CITY,
        activity=ActivityType.WALKING,
        landmarks=["Old Town"],
        confidence=0.9,
    )

    repository.synchronize_events([event])

    assert repository.list_events() == [event]


def test_repository_applies_and_records_managed_schema_version(tmp_path: Path) -> None:
    database_path = tmp_path / "project.db"
    repository = MediaAssetRepository(database_path)

    repository.initialize()
    repository.initialize()

    with sqlite3.connect(database_path) as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
    assert version == 2
    assert {"media_assets", "scenes", "events", "timeline_versions"} <= tables


def test_repository_rejects_database_from_newer_application_version(tmp_path: Path) -> None:
    database_path = tmp_path / "project.db"
    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA user_version = 999")

    with pytest.raises(PipelineStageError, match="schema is newer"):
        MediaAssetRepository(database_path).initialize()


def test_migrations_use_frozen_ddl_and_upgrade_historical_v1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "project.db"
    with sqlite3.connect(database_path) as connection:
        for statement in database._SCHEMA_V1_DDL:
            connection.execute(statement)
        connection.execute("PRAGMA user_version = 1")

    monkeypatch.setattr(
        database.Base.metadata,
        "create_all",
        lambda *_args, **_kwargs: pytest.fail("live ORM metadata must not mutate old migrations"),
    )
    MediaAssetRepository(database_path).initialize()

    with sqlite3.connect(database_path) as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        scene_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(scenes)").fetchall()
        }
        timeline_exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='timeline_versions'"
        ).fetchone()
    assert version == 2
    assert {"manual_caption", "manual_order", "edit_version"} <= scene_columns
    assert timeline_exists == (1,)


def test_failed_migration_does_not_advance_version_and_can_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "project.db"
    with sqlite3.connect(database_path) as connection:
        for statement in database._SCHEMA_V1_DDL:
            connection.execute(statement)
        connection.execute("PRAGMA user_version = 1")
    valid_v2 = database._SCHEMA_V2_DDL
    monkeypatch.setattr(database, "_SCHEMA_V2_DDL", ("INVALID MIGRATION SQL",))

    with pytest.raises(PipelineStageError, match="Could not migrate"):
        MediaAssetRepository(database_path).initialize()
    with sqlite3.connect(database_path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1

    monkeypatch.setattr(database, "_SCHEMA_V2_DDL", valid_v2)
    MediaAssetRepository(database_path).initialize()
    with sqlite3.connect(database_path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
