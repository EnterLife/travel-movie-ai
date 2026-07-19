from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest

from travelmovieai.application.context import ProjectContext
from travelmovieai.application.reporting import generate_project_report
from travelmovieai.core.config import Settings
from travelmovieai.core.exceptions import PipelineStageError
from travelmovieai.domain.enums import MediaType, StoryStyle
from travelmovieai.domain.models import (
    Event,
    MediaAsset,
    MontageClip,
    QuickMontagePlan,
    QuickMontageSettings,
    Scene,
    SceneSelectionDecision,
    SceneSelectionReport,
    Storyboard,
)
from travelmovieai.infrastructure.artifacts import write_json_atomic
from travelmovieai.infrastructure.database import MediaAssetRepository


def test_project_report_is_local_self_contained_and_escapes_private_text(tmp_path: Path) -> None:
    context, asset, scene = _project(tmp_path)
    repository = MediaAssetRepository(context.database_path)
    repository.synchronize_events(
        [
            Event(
                id=_uuid(3),
                title="Cliff <script>alert(1)</script>",
                scene_ids=[scene.id],
                summary="Aerial & sea",
                importance_score=88,
            )
        ]
    )
    write_json_atomic(
        context.artifacts_dir / "storyboard.json",
        Storyboard(
            title="Trip <unsafe>",
            style=StoryStyle.CINEMATIC,
            event_ids=[_uuid(3)],
        ),
    )
    write_json_atomic(
        context.artifacts_dir / "quick_timeline.json",
        QuickMontagePlan(
            created_at=datetime.now(UTC),
            settings=QuickMontageSettings(),
            clips=[
                MontageClip(
                    asset_id=asset.id,
                    source_path=asset.path,
                    relative_path=asset.relative_path,
                    media_type=MediaType.VIDEO,
                    duration_seconds=2,
                    scene_id=scene.id,
                )
            ],
            total_duration_seconds=2,
        ),
    )
    write_json_atomic(
        context.artifacts_dir / "selection_decisions.json",
        SceneSelectionReport(
            created_at=datetime.now(UTC),
            decisions=[
                SceneSelectionDecision(
                    scene_id=scene.id,
                    selected=True,
                    reason="Best <opening> & stable shot",
                    score=91,
                )
            ],
        ),
    )

    result = generate_project_report(context)
    rendered = result.path.read_text(encoding="utf-8")

    assert result.asset_count == 1
    assert result.scene_count == 1
    assert result.event_count == 1
    assert result.selected_clip_count == 1
    assert "default-src 'none'" in rendered
    assert "Trip &lt;unsafe&gt;" in rendered
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in rendered
    assert "Sea &amp; sky" in rendered
    assert "Best &lt;opening&gt; &amp; stable shot" in rendered
    assert "<script>alert(1)</script>" not in rendered
    assert "https://" not in rendered


def test_project_report_rejects_corrupt_typed_artifact(tmp_path: Path) -> None:
    context, _, _ = _project(tmp_path)
    (context.artifacts_dir / "storyboard.json").write_text("not-json", encoding="utf-8")

    with pytest.raises(PipelineStageError, match="storyboard.json"):
        generate_project_report(context)


def _project(tmp_path: Path) -> tuple[ProjectContext, MediaAsset, Scene]:
    input_path = tmp_path / "input"
    input_path.mkdir()
    context = ProjectContext(
        input_path=input_path,
        workspace=tmp_path / "workspace",
        settings=Settings(),
    )
    context.prepare()
    asset = MediaAsset(
        id=_uuid(1),
        path=input_path / "clip.mp4",
        relative_path=Path("clip.mp4"),
        media_type=MediaType.VIDEO,
        extension=".mp4",
        size_bytes=1,
        modified_at=datetime.now(UTC),
        modified_ns=1,
        duration_seconds=2,
    )
    scene = Scene(
        id=_uuid(2),
        asset_id=asset.id,
        start_seconds=0,
        end_seconds=2,
        caption="Sea & sky",
        quality_score=80,
        importance_score=90,
    )
    repository = MediaAssetRepository(context.database_path)
    repository.initialize()
    repository.synchronize([asset], datetime.now(UTC))
    repository.synchronize_scenes([scene])
    return context, asset, scene


def _uuid(value: int) -> UUID:
    return UUID(int=value)
