"""Pipeline stage for OpenCV visual quality analysis."""

from travelmovieai.analysis.quality import analyze_scene_quality
from travelmovieai.application.context import ProjectContext
from travelmovieai.domain.enums import PipelineStage
from travelmovieai.domain.models import StageResult
from travelmovieai.infrastructure.artifacts import write_json_atomic
from travelmovieai.infrastructure.database import MediaAssetRepository
from travelmovieai.pipeline.base import Stage


class QualityAnalysisStage(Stage):
    name = PipelineStage.QUALITY_ANALYSIS

    def run(self, context: ProjectContext) -> StageResult:
        repository = MediaAssetRepository(context.database_path)
        repository.initialize()
        report = analyze_scene_quality(repository.list_scenes())
        repository.synchronize_scenes(report.scenes)
        artifact = context.artifacts_dir / "quality_analysis.json"
        write_json_atomic(artifact, report)
        return StageResult(
            stage=self.name,
            skipped=not report.scenes,
            artifacts=[context.database_path, artifact],
            message=f"Visual quality analyzed for {len(report.scenes)} scene(s).",
        )
