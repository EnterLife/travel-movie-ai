"""SQLAlchemy persistence for project media metadata."""

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy import JSON, DateTime, Float, Integer, String, Text, create_engine, delete
from sqlalchemy import event as sqlalchemy_event
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from travelmovieai.domain.enums import MediaType
from travelmovieai.domain.models import MediaAsset


class Base(DeclarativeBase):
    pass


class MediaAssetRecord(Base):
    __tablename__ = "media_assets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    source_path: Mapped[str] = mapped_column(Text, unique=True, index=True)
    relative_path: Mapped[str] = mapped_column(Text)
    media_type: Mapped[str] = mapped_column(String(16), index=True)
    extension: Mapped[str] = mapped_column(String(16))
    size_bytes: Mapped[int] = mapped_column(Integer)
    modified_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    modified_ns: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    duration_seconds: Mapped[float | None] = mapped_column(Float)
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    fps: Mapped[float | None] = mapped_column(Float)
    latitude: Mapped[float | None] = mapped_column(Float)
    longitude: Mapped[float | None] = mapped_column(Float)
    probe_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    scan_error: Mapped[str | None] = mapped_column(Text)
    last_scanned_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class MediaAssetRepository:
    def __init__(self, database_path: Path) -> None:
        database_path.parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(
            f"sqlite+pysqlite:///{database_path.resolve().as_posix()}",
            connect_args={"check_same_thread": False},
        )

        @sqlalchemy_event.listens_for(engine, "connect")
        def configure_sqlite(connection: Any, _: Any) -> None:
            cursor = connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()

        self._engine = engine
        self._session_factory = sessionmaker(self._engine, expire_on_commit=False)

    def initialize(self) -> None:
        Base.metadata.create_all(self._engine)

    def list_assets(self) -> list[MediaAsset]:
        with self._session_factory() as session:
            records = session.query(MediaAssetRecord).order_by(MediaAssetRecord.relative_path)
            return [_record_to_asset(record) for record in records]

    def synchronize(self, assets: Sequence[MediaAsset], scanned_at: datetime) -> None:
        scanned_at = _ensure_aware(scanned_at)
        with self._session_factory.begin() as session:
            records = session.query(MediaAssetRecord).all()
            existing = {record.source_path: record for record in records}
            existing_by_id = {record.id: record for record in records}
            seen_paths: set[str] = set()

            for asset in assets:
                source_path = str(asset.path)
                seen_paths.add(source_path)
                record = existing.get(source_path) or existing_by_id.get(str(asset.id))
                if record is None:
                    record = MediaAssetRecord(id=str(asset.id), source_path=source_path)
                    session.add(record)
                else:
                    record.source_path = source_path
                _update_record(record, asset, scanned_at)

            _delete_missing_records(session, seen_paths)


def _delete_missing_records(session: Session, seen_paths: set[str]) -> None:
    statement = delete(MediaAssetRecord)
    if seen_paths:
        statement = statement.where(MediaAssetRecord.source_path.not_in(seen_paths))
    session.execute(statement)


def _update_record(record: MediaAssetRecord, asset: MediaAsset, scanned_at: datetime) -> None:
    record.relative_path = asset.relative_path.as_posix()
    record.media_type = asset.media_type.value
    record.extension = asset.extension
    record.size_bytes = asset.size_bytes
    record.modified_at = _ensure_aware(asset.modified_at)
    record.modified_ns = asset.modified_ns
    record.created_at = _ensure_aware(asset.created_at) if asset.created_at else None
    record.duration_seconds = asset.duration_seconds
    record.width = asset.width
    record.height = asset.height
    record.fps = asset.fps
    record.latitude = asset.latitude
    record.longitude = asset.longitude
    record.probe_metadata = asset.probe_metadata
    record.scan_error = asset.scan_error
    record.last_scanned_at = scanned_at


def _record_to_asset(record: MediaAssetRecord) -> MediaAsset:
    return MediaAsset(
        id=UUID(record.id),
        path=Path(record.source_path),
        relative_path=Path(record.relative_path),
        media_type=MediaType(record.media_type),
        extension=record.extension,
        size_bytes=record.size_bytes,
        modified_at=_ensure_aware(record.modified_at),
        modified_ns=record.modified_ns,
        created_at=_ensure_aware(record.created_at) if record.created_at else None,
        duration_seconds=record.duration_seconds,
        width=record.width,
        height=record.height,
        fps=record.fps,
        latitude=record.latitude,
        longitude=record.longitude,
        probe_metadata=record.probe_metadata or {},
        scan_error=record.scan_error,
    )


def _ensure_aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
