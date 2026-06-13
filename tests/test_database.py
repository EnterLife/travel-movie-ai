from datetime import UTC, datetime
from pathlib import Path

from travelmovieai.domain.enums import MediaType
from travelmovieai.domain.models import MediaAsset, Scene
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
