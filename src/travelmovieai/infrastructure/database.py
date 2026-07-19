"""SQLAlchemy persistence for project media metadata."""

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast
from uuid import UUID, uuid4

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
    update,
)
from sqlalchemy import event as sqlalchemy_event
from sqlalchemy.engine import Connection, CursorResult
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from travelmovieai.core.exceptions import PipelineStageError
from travelmovieai.domain.enums import ActivityType, LocationType, MediaType
from travelmovieai.domain.manual_editing import (
    EditableEvent,
    EditableScene,
    TimelineVersionSnapshot,
)
from travelmovieai.domain.models import Event, MediaAsset, QuickMontagePlan, Scene


class EditConflictError(PipelineStageError):
    """Raised when an optimistic manual edit revision is stale."""


class EditValidationError(PipelineStageError):
    """Raised when a manual edit does not match current project state."""


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
    manual_caption: Mapped[str | None] = mapped_column(Text, nullable=True)
    manual_transcript: Mapped[str | None] = mapped_column(Text, nullable=True)
    manual_landmarks: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    manual_order: Mapped[int | None] = mapped_column(Integer, nullable=True)
    edit_version: Mapped[int] = mapped_column(Integer, default=1)


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
    manual_title: Mapped[str | None] = mapped_column(Text, nullable=True)
    manual_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    manual_landmarks: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    manual_order: Mapped[int | None] = mapped_column(Integer, nullable=True)
    edit_version: Mapped[int] = mapped_column(Integer, default=1)


class TimelineVersionRecord(Base):
    __tablename__ = "timeline_versions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    phase: Mapped[Literal["built", "rendered"]] = mapped_column(String(16), index=True)
    variant_name: Mapped[str] = mapped_column(Text)
    variant_slug: Mapped[str] = mapped_column(String(100), index=True)
    output_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    plan_payload: Mapped[dict[str, Any]] = mapped_column(JSON)


CURRENT_SCHEMA_VERSION = 2


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

    def close(self) -> None:
        """Release pooled SQLite handles, primarily for bounded benchmark lifetimes."""

        self._engine.dispose()

    def initialize(self) -> None:
        try:
            with self._engine.begin() as connection:
                current = _schema_version(connection)
                if current > CURRENT_SCHEMA_VERSION:
                    raise PipelineStageError(
                        "Project database schema is newer than this TravelMovieAI version."
                    )
                for version in range(current + 1, CURRENT_SCHEMA_VERSION + 1):
                    migration = _MIGRATIONS.get(version)
                    if migration is None:
                        raise PipelineStageError(
                            f"Project database migration {version} is not registered."
                        )
                    migration(connection)
                    connection.exec_driver_sql(f"PRAGMA user_version = {version}")
        except SQLAlchemyError as error:
            raise PipelineStageError(
                "Could not migrate the project database. Keep the database file and retry "
                "after checking free disk space and file permissions."
            ) from error

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
                SceneRecord.manual_order.is_(None),
                SceneRecord.manual_order,
                SceneRecord.asset_id,
                SceneRecord.start_seconds,
            )
            return [_record_to_scene(record) for record in records]

    def get_scene(self, scene_id: UUID) -> Scene | None:
        """Return one scene without materializing the project scene library."""

        with self._session_factory() as session:
            record = session.get(SceneRecord, str(scene_id))
            return _record_to_scene(record) if record is not None else None

    def get_editable_scene(self, scene_id: UUID) -> EditableScene | None:
        """Return one scene together with its current manual-edit revision."""

        with self._session_factory() as session:
            record = session.get(SceneRecord, str(scene_id))
            return _record_to_editable_scene(record) if record is not None else None

    def list_editable_scenes(self) -> list[EditableScene]:
        with self._session_factory() as session:
            records = session.query(SceneRecord).order_by(
                SceneRecord.manual_order.is_(None),
                SceneRecord.manual_order,
                SceneRecord.asset_id,
                SceneRecord.start_seconds,
            )
            return [_record_to_editable_scene(record) for record in records]

    def list_editable_scenes_page(
        self,
        *,
        offset: int,
        limit: int,
        event_id: UUID | None = None,
    ) -> tuple[list[EditableScene], int] | None:
        """Return a bounded scene page and total, optionally for one event.

        ``None`` distinguishes a missing event from an event with no scenes so the
        HTTP API can provide an actionable 404 response.
        """

        if offset < 0:
            raise ValueError("offset must be non-negative")
        if limit < 1:
            raise ValueError("limit must be positive")
        with self._session_factory() as session:
            query = session.query(SceneRecord)
            if event_id is not None:
                event = session.get(EventRecord, str(event_id))
                if event is None:
                    return None
                scene_ids = list(event.scene_ids or [])
                if not scene_ids:
                    return [], 0
                query = query.filter(SceneRecord.id.in_(scene_ids))
            total = query.count()
            records = (
                query.order_by(
                    SceneRecord.manual_order.is_(None),
                    SceneRecord.manual_order,
                    SceneRecord.asset_id,
                    SceneRecord.start_seconds,
                )
                .offset(offset)
                .limit(limit)
            )
            return [_record_to_editable_scene(record) for record in records], total

    def synchronize_scenes(self, scenes: Sequence[Scene]) -> None:
        with self._session_factory.begin() as session:
            manual_state = {
                record.id: (
                    record.manual_caption,
                    record.manual_transcript,
                    record.manual_landmarks,
                    record.manual_order,
                    record.edit_version,
                    (record.scene_metadata or {}).get("event_id"),
                )
                for record in session.query(SceneRecord).all()
            }
            session.execute(delete(SceneRecord))
            records: list[SceneRecord] = []
            for scene in scenes:
                state = manual_state.get(str(scene.id), (None, None, None, None, 1, None))
                manual_order = state[3]
                if state[5] != scene.metadata.get("event_id"):
                    manual_order = None
                records.append(
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
                        manual_caption=state[0],
                        manual_transcript=state[1],
                        manual_landmarks=state[2],
                        manual_order=manual_order,
                        edit_version=state[4],
                    )
                )
            session.add_all(records)

    def update_scene(
        self,
        scene_id: UUID,
        *,
        expected_version: int,
        caption: str | None = None,
        transcript: str | None = None,
        landmarks: list[str] | None = None,
        update_caption: bool = False,
        update_transcript: bool = False,
        update_landmarks: bool = False,
    ) -> EditableScene | None:
        with self._session_factory.begin() as session:
            record = session.get(SceneRecord, str(scene_id))
            if record is None:
                return None
            values: dict[str, Any] = {"edit_version": SceneRecord.edit_version + 1}
            if update_caption:
                values["manual_caption"] = caption or ""
            if update_transcript:
                values["manual_transcript"] = transcript or ""
            if update_landmarks:
                values["manual_landmarks"] = list(landmarks or [])
            result = cast(
                CursorResult[Any],
                session.execute(
                    update(SceneRecord)
                    .where(
                        SceneRecord.id == str(scene_id),
                        SceneRecord.edit_version == expected_version,
                    )
                    .values(**values)
                ),
            )
            if result.rowcount != 1:
                _check_edit_version(record.edit_version, expected_version)
                raise EditConflictError("The scene could not be updated atomically.")
            session.expire(record)
            return _record_to_editable_scene(record)

    def reorder_scenes(
        self,
        event_id: UUID,
        ordered_scene_ids: Sequence[UUID],
        expected_versions: dict[UUID, int],
    ) -> list[EditableScene] | None:
        with self._session_factory.begin() as session:
            event = session.get(EventRecord, str(event_id))
            if event is None:
                return None
            current_ids = [UUID(scene_id) for scene_id in event.scene_ids]
            _validate_complete_order(current_ids, ordered_scene_ids, "scene")
            records = {
                UUID(record.id): record
                for record in session.query(SceneRecord).filter(
                    SceneRecord.id.in_([str(scene_id) for scene_id in current_ids])
                )
            }
            _validate_expected_versions(records, current_ids, expected_versions)
            event_result = cast(
                CursorResult[Any],
                session.execute(
                    update(EventRecord)
                    .where(
                        EventRecord.id == str(event_id),
                        EventRecord.edit_version == event.edit_version,
                    )
                    .values(
                        scene_ids=[str(scene_id) for scene_id in ordered_scene_ids],
                        edit_version=EventRecord.edit_version + 1,
                    )
                ),
            )
            if event_result.rowcount != 1:
                raise EditConflictError("The event order changed concurrently.")
            for order, scene_id in enumerate(ordered_scene_ids):
                record = records[scene_id]
                result = cast(
                    CursorResult[Any],
                    session.execute(
                        update(SceneRecord)
                        .where(
                            SceneRecord.id == str(scene_id),
                            SceneRecord.edit_version == expected_versions[scene_id],
                        )
                        .values(
                            manual_order=order,
                            edit_version=SceneRecord.edit_version + 1,
                        )
                    ),
                )
                if result.rowcount != 1:
                    raise EditConflictError("The scene order changed concurrently.")
                session.expire(record)
            return [_record_to_editable_scene(records[scene_id]) for scene_id in ordered_scene_ids]

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
                EventRecord.manual_order.is_(None),
                EventRecord.manual_order,
                EventRecord.start_at,
                EventRecord.title,
            )
            return [_record_to_event(record) for record in records]

    def list_editable_events(self) -> list[EditableEvent]:
        with self._session_factory() as session:
            records = session.query(EventRecord).order_by(
                EventRecord.manual_order.is_(None),
                EventRecord.manual_order,
                EventRecord.start_at,
                EventRecord.title,
            )
            return [_record_to_editable_event(record) for record in records]

    def synchronize_events(self, events: Sequence[Event]) -> None:
        with self._session_factory.begin() as session:
            manual_state = {
                record.id: (
                    record.manual_title,
                    record.manual_summary,
                    record.manual_landmarks,
                    record.manual_order,
                    record.edit_version,
                )
                for record in session.query(EventRecord).all()
            }
            session.execute(delete(EventRecord))
            records: list[EventRecord] = []
            for event in events:
                state = manual_state.get(str(event.id), (None, None, None, None, 1))
                records.append(
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
                        manual_title=state[0],
                        manual_summary=state[1],
                        manual_landmarks=state[2],
                        manual_order=state[3],
                        edit_version=state[4],
                    )
                )
                _apply_event_manual_metadata(
                    session,
                    event.scene_ids,
                    title=state[0],
                    summary=state[1],
                    landmarks=state[2],
                    order=state[3],
                )
            session.add_all(records)

    def update_event(
        self,
        event_id: UUID,
        *,
        expected_version: int,
        title: str | None = None,
        summary: str | None = None,
        landmarks: list[str] | None = None,
        update_title: bool = False,
        update_summary: bool = False,
        update_landmarks: bool = False,
    ) -> EditableEvent | None:
        with self._session_factory.begin() as session:
            record = session.get(EventRecord, str(event_id))
            if record is None:
                return None
            values: dict[str, Any] = {"edit_version": EventRecord.edit_version + 1}
            if update_title:
                values["manual_title"] = title
            if update_summary:
                values["manual_summary"] = summary
            if update_landmarks:
                values["manual_landmarks"] = list(landmarks or [])
            result = cast(
                CursorResult[Any],
                session.execute(
                    update(EventRecord)
                    .where(
                        EventRecord.id == str(event_id),
                        EventRecord.edit_version == expected_version,
                    )
                    .values(**values)
                ),
            )
            if result.rowcount != 1:
                _check_edit_version(record.edit_version, expected_version)
                raise EditConflictError("The event could not be updated atomically.")
            _apply_event_manual_metadata(
                session,
                [UUID(scene_id) for scene_id in record.scene_ids],
                title=(title if title is not None else record.title)
                if update_title
                else record.manual_title,
                summary=(summary if summary is not None else record.summary)
                if update_summary
                else record.manual_summary,
                landmarks=landmarks if update_landmarks else record.manual_landmarks,
                order=record.manual_order,
            )
            session.expire(record)
            return _record_to_editable_event(record)

    def reorder_events(
        self,
        ordered_event_ids: Sequence[UUID],
        expected_versions: dict[UUID, int],
    ) -> list[EditableEvent]:
        with self._session_factory.begin() as session:
            records = {UUID(record.id): record for record in session.query(EventRecord).all()}
            current_ids = list(records)
            _validate_complete_order(current_ids, ordered_event_ids, "event")
            _validate_expected_versions(records, current_ids, expected_versions)
            scene_records = {UUID(record.id): record for record in session.query(SceneRecord).all()}
            for order, event_id in enumerate(ordered_event_ids):
                record = records[event_id]
                result = cast(
                    CursorResult[Any],
                    session.execute(
                        update(EventRecord)
                        .where(
                            EventRecord.id == str(event_id),
                            EventRecord.edit_version == expected_versions[event_id],
                        )
                        .values(
                            manual_order=order,
                            edit_version=EventRecord.edit_version + 1,
                        )
                    ),
                )
                if result.rowcount != 1:
                    raise EditConflictError("The event order changed concurrently.")
                session.expire(record)
                for scene_id_text in record.scene_ids:
                    scene = scene_records.get(UUID(scene_id_text))
                    if scene is None:
                        continue
                    metadata = dict(scene.scene_metadata or {})
                    metadata["manual_event_order"] = order
                    scene.scene_metadata = metadata
            session.flush()
            return [_record_to_editable_event(records[event_id]) for event_id in ordered_event_ids]

    def record_timeline_version(
        self,
        plan: QuickMontagePlan,
        *,
        phase: Literal["built", "rendered"],
        variant_name: str = "Default",
        variant_slug: str = "default",
        output_path: Path | None = None,
    ) -> TimelineVersionSnapshot:
        snapshot = TimelineVersionSnapshot(
            id=uuid4(),
            created_at=datetime.now(UTC),
            phase=phase,
            variant_name=variant_name,
            variant_slug=variant_slug,
            plan=plan,
            output_path=output_path,
        )
        with self._session_factory.begin() as session:
            session.add(
                TimelineVersionRecord(
                    id=str(snapshot.id),
                    created_at=snapshot.created_at,
                    phase=snapshot.phase,
                    variant_name=snapshot.variant_name,
                    variant_slug=snapshot.variant_slug,
                    output_path=str(snapshot.output_path) if snapshot.output_path else None,
                    plan_payload=snapshot.plan.model_dump(mode="json"),
                )
            )
        return snapshot

    def list_timeline_versions(self, limit: int = 50) -> list[TimelineVersionSnapshot]:
        with self._session_factory() as session:
            records = (
                session.query(TimelineVersionRecord)
                .order_by(TimelineVersionRecord.created_at.desc())
                .limit(limit)
            )
            return [_record_to_timeline_version(record) for record in records]

    def get_timeline_version(self, version_id: UUID) -> TimelineVersionSnapshot | None:
        with self._session_factory() as session:
            record = session.get(TimelineVersionRecord, str(version_id))
            return _record_to_timeline_version(record) if record else None


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


def _apply_event_manual_metadata(
    session: Session,
    scene_ids: Sequence[UUID],
    *,
    title: str | None,
    summary: str | None,
    landmarks: list[str] | None,
    order: int | None,
) -> None:
    if title is None and summary is None and landmarks is None and order is None:
        return
    records = session.query(SceneRecord).filter(
        SceneRecord.id.in_([str(scene_id) for scene_id in scene_ids])
    )
    for record in records:
        metadata = dict(record.scene_metadata or {})
        if title is not None:
            metadata["event_title"] = title
        if summary is not None:
            metadata["event_summary"] = summary
        if landmarks is not None:
            metadata["event_landmarks"] = list(landmarks)
        if order is not None:
            metadata["manual_event_order"] = order
        record.scene_metadata = metadata


def _record_to_scene(record: SceneRecord) -> Scene:
    caption = record.caption if record.manual_caption is None else record.manual_caption or None
    transcript = (
        record.transcript if record.manual_transcript is None else record.manual_transcript or None
    )
    metadata = dict(record.scene_metadata or {})
    if record.manual_order is not None:
        metadata["manual_scene_order"] = record.manual_order
    if record.manual_landmarks is not None:
        metadata["landmarks"] = [
            {"name": name, "confidence": 1.0, "evidence": "manual edit"}
            for name in record.manual_landmarks
        ]
    return Scene(
        id=UUID(record.id),
        asset_id=UUID(record.asset_id),
        start_seconds=record.start_seconds,
        end_seconds=record.end_seconds,
        keyframe_path=Path(record.keyframe_path) if record.keyframe_path else None,
        caption=caption,
        transcript=transcript,
        quality_score=record.quality_score,
        importance_score=record.importance_score,
        metadata=metadata,
    )


def _record_to_editable_scene(record: SceneRecord) -> EditableScene:
    scene = _record_to_scene(record)
    return EditableScene(
        **scene.model_dump(),
        landmarks=(
            record.manual_landmarks
            if record.manual_landmarks is not None
            else _detected_landmark_names(record.scene_metadata or {})
        ),
        edit_version=record.edit_version,
    )


def _record_to_event(record: EventRecord) -> Event:
    return Event(
        id=UUID(record.id),
        title=record.manual_title or record.title,
        scene_ids=[UUID(scene_id) for scene_id in record.scene_ids],
        summary=record.manual_summary if record.manual_summary is not None else record.summary,
        importance_score=record.importance_score,
        start_at=_ensure_aware(record.start_at) if record.start_at else None,
        end_at=_ensure_aware(record.end_at) if record.end_at else None,
        location_type=LocationType(record.location_type),
        activity=ActivityType(record.activity),
        landmarks=(
            record.manual_landmarks
            if record.manual_landmarks is not None
            else record.landmarks or []
        ),
        confidence=record.confidence,
    )


def _record_to_editable_event(record: EventRecord) -> EditableEvent:
    event = _record_to_event(record)
    return EditableEvent(**event.model_dump(), edit_version=record.edit_version)


def _record_to_timeline_version(record: TimelineVersionRecord) -> TimelineVersionSnapshot:
    return TimelineVersionSnapshot(
        id=UUID(record.id),
        created_at=_ensure_aware(record.created_at),
        phase=record.phase,
        variant_name=record.variant_name,
        variant_slug=record.variant_slug,
        output_path=Path(record.output_path) if record.output_path else None,
        plan=QuickMontagePlan.model_validate(record.plan_payload),
    )


def _detected_landmark_names(metadata: dict[str, Any]) -> list[str]:
    names: list[str] = []
    raw_landmarks = metadata.get("landmarks", [])
    if not isinstance(raw_landmarks, list):
        return names
    for landmark in raw_landmarks:
        if isinstance(landmark, str) and landmark.strip():
            names.append(landmark.strip())
        elif isinstance(landmark, dict) and isinstance(landmark.get("name"), str):
            name = landmark["name"].strip()
            if name:
                names.append(name)
    return names


def _check_edit_version(current: int, expected: int) -> None:
    if current != expected:
        raise EditConflictError(
            f"The edit is stale: expected revision {expected}, current revision is {current}."
        )


def _validate_complete_order(
    current_ids: Sequence[UUID],
    ordered_ids: Sequence[UUID],
    label: str,
) -> None:
    if len(ordered_ids) != len(set(ordered_ids)):
        raise EditValidationError(f"The {label} order contains duplicate identifiers.")
    if set(current_ids) != set(ordered_ids):
        raise EditValidationError(
            f"The {label} order must contain every current {label} exactly once."
        )


def _validate_expected_versions(
    records: dict[UUID, Any],
    record_ids: Sequence[UUID],
    expected_versions: dict[UUID, int],
) -> None:
    if set(records) != set(record_ids):
        raise EditValidationError("Some reordered items no longer exist.")
    if set(expected_versions) != set(record_ids):
        raise EditValidationError("Expected revisions must cover every reordered item.")
    for record_id in record_ids:
        _check_edit_version(records[record_id].edit_version, expected_versions[record_id])


def _ensure_aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _schema_version(connection: Connection) -> int:
    value = connection.exec_driver_sql("PRAGMA user_version").scalar_one()
    if not isinstance(value, int) or value < 0:
        raise PipelineStageError("Project database reported an invalid schema version.")
    return value


def _create_initial_schema(connection: Connection) -> None:
    for statement in _SCHEMA_V1_DDL:
        connection.exec_driver_sql(statement)


def _add_manual_editing_schema(connection: Connection) -> None:
    _add_column_if_missing(connection, "scenes", "manual_caption", "TEXT")
    _add_column_if_missing(connection, "scenes", "manual_transcript", "TEXT")
    _add_column_if_missing(connection, "scenes", "manual_landmarks", "JSON")
    _add_column_if_missing(connection, "scenes", "manual_order", "INTEGER")
    _add_column_if_missing(
        connection,
        "scenes",
        "edit_version",
        "INTEGER NOT NULL DEFAULT 1",
    )
    _add_column_if_missing(connection, "events", "manual_title", "TEXT")
    _add_column_if_missing(connection, "events", "manual_summary", "TEXT")
    _add_column_if_missing(connection, "events", "manual_landmarks", "JSON")
    _add_column_if_missing(connection, "events", "manual_order", "INTEGER")
    _add_column_if_missing(
        connection,
        "events",
        "edit_version",
        "INTEGER NOT NULL DEFAULT 1",
    )
    for statement in _SCHEMA_V2_DDL:
        connection.exec_driver_sql(statement)


def _add_column_if_missing(
    connection: Connection,
    table: str,
    column: str,
    declaration: str,
) -> None:
    columns = {row[1] for row in connection.exec_driver_sql(f'PRAGMA table_info("{table}")').all()}
    if column not in columns:
        connection.exec_driver_sql(f'ALTER TABLE "{table}" ADD COLUMN "{column}" {declaration}')


_SCHEMA_V1_DDL = (
    """
    CREATE TABLE IF NOT EXISTS media_assets (
        id VARCHAR(36) NOT NULL PRIMARY KEY,
        source_path TEXT NOT NULL,
        relative_path TEXT NOT NULL,
        media_type VARCHAR(16) NOT NULL,
        extension VARCHAR(16) NOT NULL,
        size_bytes INTEGER NOT NULL,
        modified_at DATETIME NOT NULL,
        modified_ns INTEGER NOT NULL,
        created_at DATETIME,
        duration_seconds FLOAT,
        width INTEGER,
        height INTEGER,
        fps FLOAT,
        latitude FLOAT,
        longitude FLOAT,
        probe_metadata JSON NOT NULL,
        scan_error TEXT,
        last_scanned_at DATETIME NOT NULL
    )
    """,
    "CREATE UNIQUE INDEX IF NOT EXISTS ix_media_assets_source_path ON media_assets (source_path)",
    "CREATE INDEX IF NOT EXISTS ix_media_assets_media_type ON media_assets (media_type)",
    """
    CREATE TABLE IF NOT EXISTS scenes (
        id VARCHAR(36) NOT NULL PRIMARY KEY,
        asset_id VARCHAR(36) NOT NULL,
        start_seconds FLOAT NOT NULL,
        end_seconds FLOAT NOT NULL,
        keyframe_path TEXT,
        caption TEXT,
        transcript TEXT,
        quality_score FLOAT,
        importance_score FLOAT,
        scene_metadata JSON NOT NULL,
        FOREIGN KEY(asset_id) REFERENCES media_assets (id) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_scenes_asset_id ON scenes (asset_id)",
    """
    CREATE TABLE IF NOT EXISTS events (
        id VARCHAR(36) NOT NULL PRIMARY KEY,
        title TEXT NOT NULL,
        scene_ids JSON NOT NULL,
        summary TEXT NOT NULL,
        importance_score FLOAT NOT NULL,
        start_at DATETIME,
        end_at DATETIME,
        location_type VARCHAR(32) NOT NULL,
        activity VARCHAR(32) NOT NULL,
        landmarks JSON NOT NULL,
        confidence FLOAT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_events_location_type ON events (location_type)",
    "CREATE INDEX IF NOT EXISTS ix_events_activity ON events (activity)",
)

_SCHEMA_V2_DDL = (
    """
    CREATE TABLE IF NOT EXISTS timeline_versions (
        id VARCHAR(36) NOT NULL PRIMARY KEY,
        created_at DATETIME NOT NULL,
        phase VARCHAR(16) NOT NULL,
        variant_name TEXT NOT NULL,
        variant_slug VARCHAR(100) NOT NULL,
        output_path TEXT,
        plan_payload JSON NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_timeline_versions_created_at ON timeline_versions (created_at)",
    "CREATE INDEX IF NOT EXISTS ix_timeline_versions_phase ON timeline_versions (phase)",
    "CREATE INDEX IF NOT EXISTS ix_timeline_versions_variant_slug "
    "ON timeline_versions (variant_slug)",
)

_MIGRATIONS = {1: _create_initial_schema, 2: _add_manual_editing_schema}
