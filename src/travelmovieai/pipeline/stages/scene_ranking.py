"""Pipeline stage for explainable semantic scene ranking."""

from datetime import UTC, datetime

from travelmovieai.application.context import ProjectContext
from travelmovieai.domain.enums import PipelineStage
from travelmovieai.domain.models import SceneDetectionReport, StageResult
from travelmovieai.infrastructure.artifacts import write_json_atomic
from travelmovieai.infrastructure.database import MediaAssetRepository
from travelmovieai.pipeline.base import Stage
from travelmovieai.story.ranking import rank_scenes


class SceneRankingStage(Stage):
    name = PipelineStage.SCENE_RANKING

    def run(self, context: ProjectContext) -> StageResult:
        repository = MediaAssetRepository(context.database_path)
        repository.initialize()
        ranked_scenes = rank_scenes(repository.list_scenes())
        repository.synchronize_scenes(ranked_scenes)
        report = SceneDetectionReport(
            created_at=datetime.now(UTC),
            scenes=ranked_scenes,
            detected_count=len(ranked_scenes),
        )
        artifact = context.artifacts_dir / "ranked_scenes.json"
        write_json_atomic(artifact, report)
        return StageResult(
            stage=self.name,
            skipped=not ranked_scenes,
            artifacts=[context.database_path, artifact],
            message=f"Scene ranking scored {len(ranked_scenes)} scene(s).",
        )
