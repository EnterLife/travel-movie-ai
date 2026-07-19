import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from threading import Event as ThreadEvent
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from travelmovieai.application.validation import ProjectPaths, validate_project_paths
from travelmovieai.application.variants import safe_variant_slug, variant_output_path
from travelmovieai.application.workspace_identity import ensure_workspace_identity
from travelmovieai.core.config import Settings
from travelmovieai.domain.enums import MediaType
from travelmovieai.domain.manual_editing import compare_timeline_versions
from travelmovieai.domain.models import (
    Event,
    MediaAsset,
    MontageClip,
    MusicPlan,
    QuickMontagePlan,
    QuickMontageResult,
    QuickMontageSettings,
    Scene,
)
from travelmovieai.editing.timeline import build_semantic_montage_plan
from travelmovieai.infrastructure.database import (
    EditConflictError,
    EditValidationError,
    MediaAssetRepository,
)
from travelmovieai.story.events import detect_events
from travelmovieai.web.app import create_app
from travelmovieai.web.movie_jobs import MovieJobManager


def _asset(path: Path) -> MediaAsset:
    return MediaAsset(
        path=path.resolve(),
        relative_path=Path(path.name),
        media_type=MediaType.VIDEO,
        extension=".mp4",
        size_bytes=10,
        modified_at=datetime(2026, 1, 1, tzinfo=UTC),
        modified_ns=1,
        duration_seconds=10,
    )


def _project_repository(
    tmp_path: Path,
) -> tuple[MediaAssetRepository, MediaAsset, list[Scene], Event]:
    repository = MediaAssetRepository(tmp_path / "project.db")
    repository.initialize()
    asset = _asset(tmp_path / "source.mp4")
    repository.synchronize([asset], datetime.now(UTC))
    scenes = [
        Scene(
            asset_id=asset.id,
            start_seconds=index * 2,
            end_seconds=index * 2 + 2,
            caption=f"Automatic {index}",
            metadata={"landmarks": [{"name": "Detected"}]},
        )
        for index in range(2)
    ]
    repository.synchronize_scenes(scenes)
    event = Event(title="Automatic event", scene_ids=[scene.id for scene in scenes])
    repository.synchronize_events([event])
    return repository, asset, scenes, event


def _plan(
    asset: MediaAsset,
    scenes: list[Scene],
    *,
    duration: float = 20,
    reverse: bool = False,
) -> QuickMontagePlan:
    ordered = list(reversed(scenes)) if reverse else scenes
    settings = QuickMontageSettings(target_duration_seconds=duration)
    clips = [
        MontageClip(
            asset_id=asset.id,
            source_path=asset.path,
            relative_path=asset.relative_path,
            media_type=asset.media_type,
            source_start_seconds=scene.start_seconds,
            duration_seconds=2,
            scene_id=scene.id,
        )
        for scene in ordered
    ]
    return QuickMontagePlan(
        created_at=datetime.now(UTC),
        settings=settings,
        clips=clips,
        total_duration_seconds=4,
        selection_mode="semantic",
    )


def test_v1_database_is_migrated_in_place_without_losing_tables(tmp_path: Path) -> None:
    database_path = tmp_path / "legacy.db"
    with sqlite3.connect(database_path) as connection:
        connection.executescript(
            """
            CREATE TABLE scenes (id TEXT PRIMARY KEY);
            CREATE TABLE events (id TEXT PRIMARY KEY);
            PRAGMA user_version = 1;
            """
        )

    MediaAssetRepository(database_path).initialize()

    with sqlite3.connect(database_path) as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        scene_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(scenes)").fetchall()
        }
        event_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(events)").fetchall()
        }
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }

    assert version == 2
    assert {"manual_caption", "manual_landmarks", "manual_order", "edit_version"} <= (scene_columns)
    assert {"manual_title", "manual_landmarks", "manual_order", "edit_version"} <= (event_columns)
    assert "timeline_versions" in tables


def test_manual_edits_survive_pipeline_synchronization_and_reject_stale_revision(
    tmp_path: Path,
) -> None:
    repository, asset, scenes, event = _project_repository(tmp_path)

    edited_scene = repository.update_scene(
        scenes[0].id,
        expected_version=1,
        caption="My opening",
        transcript="Welcome",
        landmarks=["Harbour"],
        update_caption=True,
        update_transcript=True,
        update_landmarks=True,
    )
    edited_event = repository.update_event(
        event.id,
        expected_version=1,
        title="Arrival",
        summary="First evening",
        landmarks=["Harbour"],
        update_title=True,
        update_summary=True,
        update_landmarks=True,
    )

    assert edited_scene is not None and edited_scene.edit_version == 2
    assert edited_scene.caption == "My opening"
    assert edited_scene.landmarks == ["Harbour"]
    assert edited_event is not None and edited_event.title == "Arrival"
    with pytest.raises(EditConflictError, match="stale|atomically"):
        repository.update_scene(
            scenes[0].id,
            expected_version=1,
            caption="Lost concurrent edit",
            update_caption=True,
        )

    repository.synchronize_scenes(
        [scenes[0].model_copy(update={"caption": "New automatic caption"}), scenes[1]]
    )
    repository.synchronize_events([event.model_copy(update={"title": "New automatic event"})])

    stored_scene = repository.list_editable_scenes()[0]
    stored_event = repository.list_editable_events()[0]
    assert stored_scene.caption == "My opening"
    assert stored_scene.metadata["landmarks"] == [
        {"name": "Harbour", "confidence": 1.0, "evidence": "manual edit"}
    ]
    assert stored_scene.metadata["event_title"] == "Arrival"
    assert stored_scene.metadata["event_summary"] == "First evening"
    assert stored_scene.metadata["event_landmarks"] == ["Harbour"]
    assert stored_event.title == "Arrival"
    assert stored_event.summary == "First evening"
    detected, _ = detect_events([stored_scene], [asset])
    assert detected.events[0].landmarks == ["Harbour"]


def test_manual_order_is_complete_persistent_and_versioned(tmp_path: Path) -> None:
    repository, _, scenes, event = _project_repository(tmp_path)

    reordered = repository.reorder_scenes(
        event.id,
        [scenes[1].id, scenes[0].id],
        {scenes[0].id: 1, scenes[1].id: 1},
    )

    assert reordered is not None
    assert [scene.id for scene in reordered] == [scenes[1].id, scenes[0].id]
    assert [scene.id for scene in repository.list_editable_scenes()] == [
        scenes[1].id,
        scenes[0].id,
    ]
    assert [scene.edit_version for scene in reordered] == [2, 2]
    with pytest.raises(EditConflictError, match="stale|concurrently"):
        repository.reorder_scenes(
            event.id,
            [scenes[0].id, scenes[1].id],
            {scenes[0].id: 1, scenes[1].id: 1},
        )
    with pytest.raises(EditValidationError, match="duplicate"):
        repository.reorder_scenes(
            event.id,
            [scenes[0].id, scenes[0].id],
            {scenes[0].id: 2, scenes[1].id: 2},
        )


def test_scene_regrouping_clears_manual_order_from_previous_event(tmp_path: Path) -> None:
    repository, _, scenes, event = _project_repository(tmp_path)
    repository.reorder_scenes(
        event.id,
        [scenes[1].id, scenes[0].id],
        {scenes[0].id: 1, scenes[1].id: 1},
    )
    new_event_id = uuid4()
    regrouped = [
        scene.model_copy(update={"metadata": {**scene.metadata, "event_id": str(new_event_id)}})
        for scene in scenes
    ]

    repository.synchronize_scenes(regrouped)
    stored = repository.list_editable_scenes()

    assert [scene.id for scene in stored] == [scenes[0].id, scenes[1].id]
    assert all("manual_scene_order" not in scene.metadata for scene in stored)


def test_semantic_planner_applies_manual_event_and_scene_order(tmp_path: Path) -> None:
    asset = _asset(tmp_path / "source.mp4")
    first_event = uuid4()
    second_event = uuid4()
    scenes = [
        Scene(
            asset_id=asset.id,
            start_seconds=0,
            end_seconds=2,
            importance_score=90,
            metadata={
                "selection_override": "include",
                "event_id": str(first_event),
                "manual_event_order": 1,
                "manual_scene_order": 1,
            },
        ),
        Scene(
            asset_id=asset.id,
            start_seconds=2,
            end_seconds=4,
            importance_score=90,
            metadata={
                "selection_override": "include",
                "event_id": str(first_event),
                "manual_event_order": 1,
                "manual_scene_order": 0,
            },
        ),
        Scene(
            asset_id=asset.id,
            start_seconds=4,
            end_seconds=6,
            importance_score=90,
            metadata={
                "selection_override": "include",
                "event_id": str(second_event),
                "manual_event_order": 0,
                "manual_scene_order": 0,
            },
        ),
    ]

    plan = build_semantic_montage_plan(
        [asset],
        scenes,
        QuickMontageSettings(
            target_duration_seconds=6,
            max_video_clip_seconds=2,
            max_scenes_per_source=10,
            preserve_chronology=True,
            min_semantic_score=0,
        ),
    )

    assert [clip.scene_id for clip in plan.clips] == [
        scenes[2].id,
        scenes[1].id,
        scenes[0].id,
    ]


def test_timeline_versions_compare_selection_order_and_settings(tmp_path: Path) -> None:
    repository, asset, scenes, _ = _project_repository(tmp_path)
    third = Scene(asset_id=asset.id, start_seconds=5, end_seconds=7)
    before = repository.record_timeline_version(
        _plan(asset, scenes),
        phase="built",
    )
    after = repository.record_timeline_version(
        _plan(asset, [scenes[0], third], duration=30, reverse=True),
        phase="rendered",
        variant_name="Short social cut",
        variant_slug="short-social-cut",
        output_path=tmp_path / "short-social-cut.mp4",
    )

    comparison = compare_timeline_versions(before, after)
    restored = repository.get_timeline_version(after.id)

    assert restored == after
    assert comparison.selected_scene_ids_added == [third.id]
    assert comparison.selected_scene_ids_removed == [scenes[1].id]
    assert comparison.order_changes[0].scene_id == scenes[0].id
    assert comparison.settings_changes["target_duration_seconds"].before == 20
    assert comparison.settings_changes["target_duration_seconds"].after == 30


def test_timeline_versions_compare_chronological_clip_and_audio_changes(
    tmp_path: Path,
) -> None:
    repository, first_asset, _, _ = _project_repository(tmp_path)
    second_asset = _asset(tmp_path / "second.mp4")
    repository.synchronize([first_asset, second_asset], datetime.now(UTC))
    settings = QuickMontageSettings(target_duration_seconds=20)
    first_clip = MontageClip(
        asset_id=first_asset.id,
        source_path=first_asset.path,
        relative_path=first_asset.relative_path,
        media_type=first_asset.media_type,
        duration_seconds=4,
        caption="Before",
    )
    second_clip = MontageClip(
        asset_id=second_asset.id,
        source_path=second_asset.path,
        relative_path=second_asset.relative_path,
        media_type=second_asset.media_type,
        duration_seconds=4,
    )
    before = repository.record_timeline_version(
        QuickMontagePlan(
            created_at=datetime.now(UTC),
            settings=settings,
            clips=[first_clip, second_clip],
            total_duration_seconds=8,
        ),
        phase="built",
    )
    after = repository.record_timeline_version(
        QuickMontagePlan(
            created_at=datetime.now(UTC),
            settings=settings,
            clips=[
                second_clip,
                first_clip.model_copy(update={"duration_seconds": 6, "caption": "After"}),
            ],
            total_duration_seconds=10,
            music_path=tmp_path / "theme.wav",
            music_plan=MusicPlan(mode="manual", duration_seconds=10),
            narration_path=tmp_path / "narration.wav",
        ),
        phase="rendered",
    )

    comparison = compare_timeline_versions(before, after)

    assert len(comparison.clip_order_changes) == 2
    first_change = next(
        change
        for change in comparison.clip_changes
        if change.clip_key.startswith(f"asset:{first_asset.id}")
    )
    assert first_change.changed_fields["duration_seconds"].after == 6
    assert first_change.changed_fields["caption"].after == "After"
    assert comparison.plan_changes["total_duration_seconds"].after == 10
    assert comparison.plan_changes["music_file"].after == "theme.wav"
    assert comparison.plan_changes["narration_file"].after == "narration.wav"


def test_variant_paths_are_safe_unique_and_do_not_replace_final(tmp_path: Path) -> None:
    first_id = uuid4()
    second_id = uuid4()

    first = variant_output_path(tmp_path, "Family Highlights", first_id)
    second = variant_output_path(tmp_path, "Family Highlights", second_id)

    assert safe_variant_slug("Family Highlights") == "family-highlights"
    assert first != second
    assert first.parent == (tmp_path / "artifacts" / "variants").resolve()
    assert first.name != "final.mp4"
    with pytest.raises(ValueError, match="path separators"):
        safe_variant_slug("../outside")


class _VariantMovieService:
    def __init__(self) -> None:
        self.seen_variants: list[str] = []

    def resolve_project_paths(self, input_path: Path, workspace: Path | None) -> ProjectPaths:
        assert workspace is not None
        return validate_project_paths(input_path, workspace)

    def create_quick_montage(
        self,
        *,
        input_path: Path,
        workspace: Path | None,
        settings: QuickMontageSettings,
        variant_name: str = "Default",
        output_path: Path | None = None,
        progress: object | None = None,
    ) -> QuickMontageResult:
        assert workspace is not None and output_path is not None
        self.seen_variants.append(variant_name)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"movie")
        timeline_path = workspace / "artifacts" / f"{output_path.stem}.json"
        plan = QuickMontagePlan(
            created_at=datetime.now(UTC),
            settings=settings,
            total_duration_seconds=5,
            selection_mode="semantic" if settings.semantic_analysis else "chronological",
        )
        timeline_path.write_text(plan.model_dump_json(), encoding="utf-8")
        repository = MediaAssetRepository(workspace / "project.db")
        repository.initialize()
        repository.record_timeline_version(
            plan,
            phase="rendered",
            variant_name=variant_name,
            variant_slug=safe_variant_slug(variant_name),
            output_path=output_path,
        )
        return QuickMontageResult(
            output_path=output_path,
            timeline_path=timeline_path,
            clip_count=0,
            duration_seconds=5,
            selection_mode=plan.selection_mode,
        )


def test_movie_jobs_keep_multiple_named_variant_outputs_and_history(tmp_path: Path) -> None:
    input_path = tmp_path / "input"
    workspace = tmp_path / "workspace"
    input_path.mkdir()
    service = _VariantMovieService()
    manager = MovieJobManager(
        service,
        render_disk_reserve_mb=0,
        render_disk_safety_factor=1,
    )
    try:
        first = manager.submit(
            input_path,
            workspace,
            QuickMontageSettings(target_duration_seconds=5),
            "Director cut",
        )
        first_done = _wait_for_movie(manager, first.id)
        second = manager.submit(
            input_path,
            workspace,
            QuickMontageSettings(target_duration_seconds=5),
            "Social cut",
        )
        second_done = _wait_for_movie(manager, second.id)
    finally:
        manager.shutdown()

    assert first_done.output_path != second_done.output_path
    assert first_done.variant_name == "Director cut"
    assert second_done.variant_slug == "social-cut"
    assert first_done.output_path is not None and first_done.output_path.is_file()
    assert second_done.output_path is not None and second_done.output_path.is_file()
    assert not (workspace / "artifacts" / "final.mp4").exists()
    assert {job.variant_name for job in manager.list()} == {"Director cut", "Social cut"}
    assert service.seen_variants == ["Director cut", "Social cut"]
    versions = MediaAssetRepository(workspace / "project.db").list_timeline_versions()
    assert [version.variant_name for version in versions] == ["Social cut", "Director cut"]


def test_semantic_movie_job_does_not_duplicate_stage_version_snapshot(tmp_path: Path) -> None:
    input_path = tmp_path / "input"
    workspace = tmp_path / "workspace"
    input_path.mkdir()
    manager = MovieJobManager(
        _VariantMovieService(),
        render_disk_reserve_mb=0,
        render_disk_safety_factor=1,
    )
    try:
        submitted = manager.submit(
            input_path,
            workspace,
            QuickMontageSettings(target_duration_seconds=5, semantic_analysis=True),
            "Exact semantic name",
        )
        _wait_for_movie(manager, submitted.id)
    finally:
        manager.shutdown()

    repository = MediaAssetRepository(workspace / "project.db")
    repository.initialize()
    versions = repository.list_timeline_versions()
    assert len(versions) == 1
    assert versions[0].variant_name == "Exact semantic name"


def _wait_for_movie(manager: MovieJobManager, job_id: UUID):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        job = manager.get(job_id)
        assert job is not None
        if job.status in {"completed", "failed"}:
            assert job.status == "completed", job.error
            return job
        time.sleep(0.01)
    raise AssertionError("movie job did not finish")


def test_manual_edit_api_returns_conflict_and_version_comparison(tmp_path: Path) -> None:
    input_path = tmp_path / "input"
    workspace = tmp_path / "workspace"
    input_path.mkdir()
    ensure_workspace_identity(input_path, workspace)
    repository, asset, scenes, event = _project_repository(workspace)
    before = repository.record_timeline_version(_plan(asset, scenes), phase="built")
    after = repository.record_timeline_version(
        _plan(asset, scenes, reverse=True),
        phase="rendered",
    )
    app = create_app(settings=Settings(workspace=tmp_path / "root-workspace"))
    query = {"input_path": str(input_path), "workspace": str(workspace)}
    with TestClient(app) as client:
        updated = client.patch(
            f"/api/scenes/{scenes[0].id}",
            json={**query, "expected_version": 1, "caption": "Manual caption"},
        )
        stale = client.patch(
            f"/api/scenes/{scenes[0].id}",
            json={**query, "expected_version": 1, "caption": "Stale caption"},
        )
        updated_event = client.patch(
            f"/api/events/{event.id}",
            json={
                **query,
                "expected_version": 1,
                "title": "Edited event",
                "summary": "Edited summary",
                "landmarks": ["Promenade"],
            },
        )
        reordered_event = client.put(
            "/api/events/order",
            json={
                **query,
                "ordered_ids": [str(event.id)],
                "expected_versions": {str(event.id): 2},
            },
        )
        invalid_order = client.put(
            "/api/events/order",
            json={
                **query,
                "ordered_ids": [str(event.id), str(event.id)],
                "expected_versions": {str(event.id): 1},
            },
        )
        comparison = client.get(
            "/api/timeline-versions/compare",
            params={
                **query,
                "before_id": str(before.id),
                "after_id": str(after.id),
            },
        )
        version_detail = client.get(
            f"/api/timeline-versions/{after.id}",
            params=query,
        )

    assert updated.status_code == 200
    assert updated.json()["scenes"][0]["caption"] == "Manual caption"
    assert stale.status_code == 409
    assert updated_event.status_code == 200
    assert updated_event.json()["events"][0]["title"] == "Edited event"
    assert reordered_event.status_code == 200
    assert invalid_order.status_code == 422
    assert comparison.status_code == 200
    assert comparison.json()["comparison"]["order_changes"]
    assert version_detail.status_code == 200
    assert version_detail.json()["version"]["id"] == str(after.id)


def test_manual_edit_holds_workspace_reservation_until_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "input"
    workspace = tmp_path / "workspace"
    input_path.mkdir()
    ensure_workspace_identity(input_path, workspace)
    _, _, _, event = _project_repository(workspace)
    edit_started = ThreadEvent()
    release_edit = ThreadEvent()
    original_update = MediaAssetRepository.update_event

    def blocking_update(self: MediaAssetRepository, *args: object, **kwargs: object):
        edit_started.set()
        assert release_edit.wait(timeout=2)
        return original_update(self, *args, **kwargs)

    monkeypatch.setattr(MediaAssetRepository, "update_event", blocking_update)
    movie_manager = MovieJobManager(
        _VariantMovieService(),
        render_disk_reserve_mb=0,
        render_disk_safety_factor=1,
    )
    app = create_app(
        settings=Settings(workspace=tmp_path / "root-workspace"),
        movie_job_manager=movie_manager,
    )
    query = {"input_path": str(input_path), "workspace": str(workspace)}

    with TestClient(app) as client, ThreadPoolExecutor(max_workers=2) as executor:
        edit_future = executor.submit(
            client.patch,
            f"/api/events/{event.id}",
            json={**query, "expected_version": 1, "title": "Committed title"},
        )
        assert edit_started.wait(timeout=1)
        movie_future = executor.submit(
            client.post,
            "/api/movies",
            json={
                **query,
                "settings": {
                    "target_duration_seconds": 5,
                    "music_enabled": False,
                },
            },
        )
        time.sleep(0.1)
        assert not movie_future.done()
        release_edit.set()
        edit_response = edit_future.result(timeout=2)
        movie_response = movie_future.result(timeout=2)
        completed = _wait_for_movie(movie_manager, UUID(movie_response.json()["id"]))

    assert edit_response.status_code == 200
    assert movie_response.status_code == 202
    assert completed.status == "completed"


def test_manual_edit_conflict_releases_workspace_reservation(tmp_path: Path) -> None:
    input_path = tmp_path / "input"
    workspace = tmp_path / "workspace"
    input_path.mkdir()
    ensure_workspace_identity(input_path, workspace)
    _, _, scenes, _ = _project_repository(workspace)
    movie_manager = MovieJobManager(
        _VariantMovieService(),
        render_disk_reserve_mb=0,
        render_disk_safety_factor=1,
    )
    app = create_app(
        settings=Settings(workspace=tmp_path / "root-workspace"),
        movie_job_manager=movie_manager,
    )
    query = {"input_path": str(input_path), "workspace": str(workspace)}

    with TestClient(app) as client:
        conflict = client.patch(
            f"/api/scenes/{scenes[0].id}",
            json={**query, "expected_version": 99, "caption": "Stale edit"},
        )
        movie = client.post(
            "/api/movies",
            json={
                **query,
                "settings": {
                    "target_duration_seconds": 5,
                    "music_enabled": False,
                },
            },
        )
        completed = _wait_for_movie(movie_manager, UUID(movie.json()["id"]))

    assert conflict.status_code == 409
    assert movie.status_code == 202
    assert completed.status == "completed"
