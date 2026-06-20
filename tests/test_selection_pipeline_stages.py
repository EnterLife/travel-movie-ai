from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from travelmovieai.application.context import ProjectContext
from travelmovieai.core.config import Settings
from travelmovieai.domain.enums import ActivityType, LocationType, MediaType, PipelineStage
from travelmovieai.domain.models import (
    Event,
    MediaAsset,
    QuickMontagePlan,
    Scene,
    SceneDetectionReport,
    SceneSelectionReport,
)
from travelmovieai.infrastructure.database import MediaAssetRepository
from travelmovieai.pipeline.stages.scene_ranking import SceneRankingStage
from travelmovieai.pipeline.stages.storyboard import StoryBuilderStage
from travelmovieai.pipeline.stages.timeline_builder import TimelineBuilderStage


def test_scene_ranking_stage_persists_ranked_scenes(tmp_path: Path) -> None:
    context = _context(tmp_path)
    asset = _asset(tmp_path / "clip.mp4")
    weaker = _scene(asset, score=45, start=0)
    stronger = _scene(asset, score=90, start=4)
    _seed_project(context, [asset], [weaker, stronger])

    result = SceneRankingStage().run(context)
    report = SceneDetectionReport.model_validate_json(
        (context.artifacts_dir / "ranked_scenes.json").read_text(encoding="utf-8")
    )
    stored = {
        scene.id: scene for scene in MediaAssetRepository(context.database_path).list_scenes()
    }

    assert result.stage is PipelineStage.SCENE_RANKING
    assert result.skipped is False
    assert [scene.id for scene in report.scenes] == [stronger.id, weaker.id]
    assert (
        stored[stronger.id].metadata["ranking_score"] > stored[weaker.id].metadata["ranking_score"]
    )
    assert "ranking_factors" in stored[stronger.id].metadata


def test_timeline_builder_stage_writes_plan_and_selection_report(tmp_path: Path) -> None:
    context = _context(tmp_path)
    assets = [
        _asset(tmp_path / "opening.mp4"),
        _asset(tmp_path / "highlight.mp4"),
    ]
    scenes = [
        _scene(
            assets[0],
            score=85,
            start=0,
            story_section_index=0,
            story_section_role="opening",
            story_role_order=0,
        ),
        _scene(
            assets[1],
            score=92,
            start=0,
            story_section_index=2,
            story_section_role="highlight",
            story_role_order=2,
        ),
    ]
    _seed_project(context, assets, scenes)
    SceneRankingStage().run(context)

    result = TimelineBuilderStage().run(context)
    plan = QuickMontagePlan.model_validate_json(
        (context.artifacts_dir / "quick_timeline.json").read_text(encoding="utf-8")
    )
    selection = SceneSelectionReport.model_validate_json(
        (context.artifacts_dir / "selection_decisions.json").read_text(encoding="utf-8")
    )

    assert result.stage is PipelineStage.TIMELINE_BUILDER
    assert result.skipped is False
    assert plan.selection_mode == "semantic"
    assert [clip.scene_id for clip in plan.clips] == [scenes[0].id, scenes[1].id]
    assert {decision.scene_id for decision in selection.decisions} == {scene.id for scene in scenes}


def test_story_builder_stage_applies_story_metadata_to_scenes(tmp_path: Path) -> None:
    context = _context(tmp_path)
    asset = _asset(tmp_path / "arrival.mp4")
    event_id = uuid4()
    scene = _scene(asset, score=88, start=0, event_id=str(event_id))
    event = Event(
        id=event_id,
        title="Arrival",
        scene_ids=[scene.id],
        summary="Arrival at the destination.",
        importance_score=88,
        start_at=datetime(2026, 1, 1, tzinfo=UTC),
        end_at=datetime(2026, 1, 1, 0, 0, 6, tzinfo=UTC),
        location_type=LocationType.AIRPORT,
        activity=ActivityType.ARRIVING,
        confidence=0.9,
    )
    _seed_project(context, [asset], [scene], [event])

    result = StoryBuilderStage().run(context)
    stored = MediaAssetRepository(context.database_path).list_scenes()[0]

    assert result.stage is PipelineStage.STORY_BUILDER
    assert result.skipped is False
    assert stored.metadata["story_section_role"] == "opening"
    assert stored.metadata["story_role_order"] == 0


def test_timeline_builder_stage_skips_without_assets_or_scenes(tmp_path: Path) -> None:
    context = _context(tmp_path)
    context.prepare()

    result = TimelineBuilderStage().run(context)

    assert result.stage is PipelineStage.TIMELINE_BUILDER
    assert result.skipped is True
    assert not (context.artifacts_dir / "quick_timeline.json").exists()


def _context(tmp_path: Path) -> ProjectContext:
    context = ProjectContext(
        input_path=tmp_path / "media",
        workspace=tmp_path / "workspace",
        settings=Settings(),
    )
    context.prepare()
    return context


def _seed_project(
    context: ProjectContext,
    assets: list[MediaAsset],
    scenes: list[Scene],
    events: list[Event] | None = None,
) -> None:
    repository = MediaAssetRepository(context.database_path)
    repository.initialize()
    repository.synchronize(assets, datetime.now(UTC))
    repository.synchronize_scenes(scenes)
    if events is not None:
        repository.synchronize_events(events)


def _asset(path: Path) -> MediaAsset:
    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    return MediaAsset(
        path=path,
        relative_path=Path(path.name),
        media_type=MediaType.VIDEO,
        extension=".mp4",
        size_bytes=1,
        modified_at=created_at,
        modified_ns=1,
        created_at=created_at,
        duration_seconds=8,
    )


def _scene(asset: MediaAsset, *, score: float, start: float, **metadata: object) -> Scene:
    event_id = uuid4()
    return Scene(
        asset_id=asset.id,
        start_seconds=start,
        end_seconds=start + 6,
        quality_score=80,
        importance_score=score,
        caption=asset.relative_path.stem,
        metadata={
            "event_id": str(event_id),
            "event_importance": score,
            "location_type": "city",
            "activity": "walking",
            "emotion": "joyful",
            **metadata,
        },
    )
