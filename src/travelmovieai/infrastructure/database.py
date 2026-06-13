"""SQLAlchemy persistence for project media metadata."""

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    delete,
)
from sqlalchemy import event as sqlalchemy_event
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from travelmovieai.domain.enums import ActivityType, LocationType, MediaType
from travelmovieai.domain.models import Event, MediaAsset, Scene


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


class SceneRecord(Base):
    __tablename__ = "scenes"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    asset_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("media_assets.id", ondelete="CASCADE"),
        index=True,
    )
    start_seconds: Mapped[float] = mapped_column(Float)
    end_seconds: Mapped[float] = mapped_column(Float)
    keyframe_path: Mapped[str | None] = mapped_column(Text)
    caption: Mapped[str | None] = mapped_column(Text)
    transcript: Mapped[str | None] = mapped_column(Text)
    quality_score: Mapped[float | None] = mapped_column(Float)
    importance_score: Mapped[float | None] = mapped_column(Float)
    scene_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class EventRecord(Base):
    __tablename__ = "events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    title: Mapped[str] = mapped_column(Text)
    scene_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    summary: Mapped[str] = mapped_column(Text, default="")
    importance_score: Mapped[float] = mapped_column(Float)
    start_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    end_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    location_type: Mapped[str] = mapped_column(String(32), index=True)
    activity: Mapped[str] = mapped_column(String(32), index=True)
    landmarks: Mapped[list[str]] = mapped_column(JSON, default=list)
    confidence: Mapped[float] = mapped_column(Float)


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

    def list_scenes(self) -> list[Scene]:
        with self._session_factory() as session:
            records = session.query(SceneRecord).order_by(
                SceneRecord.asset_id,
                SceneRecord.start_seconds,
            )
            return [_record_to_scene(record) for record in records]

    def synchronize_scenes(self, scenes: Sequence[Scene]) -> None:
        with self._session_factory.begin() as session:
            session.execute(delete(SceneRecord))
            session.add_all(
                SceneRecord(
                    id=str(scene.id),
                    asset_id=str(scene.asset_id),
                    start_seconds=scene.start_seconds,
                    end_seconds=scene.end_seconds,
                    keyframe_path=str(scene.keyframe_path) if scene.keyframe_path else None,
                    caption=scene.caption,
                    transcript=scene.transcript,
                    quality_score=scene.quality_score,
                    importance_score=scene.importance_score,
                    scene_metadata=scene.metadata,
                )
                for scene in scenes
            )

    def set_scene_selection_override(
        self,
        scene_id: UUID,
        decision: str,
    ) -> Scene | None:
        with self._session_factory.begin() as session:
            record = session.get(SceneRecord, str(scene_id))
            if record is None:
                return None
            metadata = dict(record.scene_metadata or {})
            if decision == "auto":
                metadata.pop("selection_override", None)
            else:
                metadata["selection_override"] = decision
            record.scene_metadata = metadata
            session.flush()
            return _record_to_scene(record)

    def list_events(self) -> list[Event]:
        with self._session_factory() as session:
            records = session.query(EventRecord).order_by(
                EventRecord.start_at,
                EventRecord.title,
            )
            return [_record_to_event(record) for record in records]

    def synchronize_events(self, events: Sequence[Event]) -> None:
        with self._session_factory.begin() as session:
            session.execute(delete(EventRecord))
            session.add_all(
                EventRecord(
                    id=str(event.id),
                    title=event.title,
                    scene_ids=[str(scene_id) for scene_id in event.scene_ids],
                    summary=event.summary,
                    importance_score=event.importance_score,
                    start_at=event.start_at,
                    end_at=event.end_at,
                    location_type=event.location_type.value,
                    activity=event.activity.value,
                    landmarks=event.landmarks,
                    confidence=event.confidence,
                )
                for event in events
            )


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


def _record_to_scene(record: SceneRecord) -> Scene:
    return Scene(
        id=UUID(record.id),
        asset_id=UUID(record.asset_id),
        start_seconds=record.start_seconds,
        end_seconds=record.end_seconds,
        keyframe_path=Path(record.keyframe_path) if record.keyframe_path else None,
        caption=record.caption,
        transcript=record.transcript,
        quality_score=record.quality_score,
        importance_score=record.importance_score,
        metadata=record.scene_metadata or {},
    )


def _record_to_event(record: EventRecord) -> Event:
    return Event(
        id=UUID(record.id),
        title=record.title,
        scene_ids=[UUID(scene_id) for scene_id in record.scene_ids],
        summary=record.summary,
        importance_score=record.importance_score,
        start_at=_ensure_aware(record.start_at) if record.start_at else None,
        end_at=_ensure_aware(record.end_at) if record.end_at else None,
        location_type=LocationType(record.location_type),
        activity=ActivityType(record.activity),
        landmarks=record.landmarks or [],
        confidence=record.confidence,
    )


def _ensure_aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
