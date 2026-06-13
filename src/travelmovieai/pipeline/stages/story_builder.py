"""Pipeline stage that assembles multimodal scene descriptions."""

from travelmovieai.application.context import ProjectContext
from travelmovieai.domain.enums import PipelineStage
from travelmovieai.domain.models import StageResult
from travelmovieai.infrastructure.artifacts import write_json_atomic
from travelmovieai.infrastructure.database import MediaAssetRepository
from travelmovieai.pipeline.base import Stage
from travelmovieai.story.builder import build_multimodal_descriptions


class SceneCaptioningStage(Stage):
    name = PipelineStage.SCENE_CAPTIONING

    def run(self, context: ProjectContext) -> StageResult:
        repository = MediaAssetRepository(context.database_path)
        repository.initialize()
        report = build_multimodal_descriptions(repository.list_scenes())
        artifact = context.artifacts_dir / "scene_descriptions.json"
        write_json_atomic(artifact, report)
        return StageResult(
            stage=self.name,
            skipped=not report.descriptions,
            artifacts=[artifact],
            message=(
                f"Story builder prepared {len(report.descriptions)} "
                "multimodal scene description(s)."
            ),
        )
