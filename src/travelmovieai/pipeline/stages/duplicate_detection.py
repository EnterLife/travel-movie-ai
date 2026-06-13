"""Pipeline stage for perceptual duplicate scene detection."""

from travelmovieai.analysis.duplicates import detect_duplicate_scenes
from travelmovieai.application.context import ProjectContext
from travelmovieai.domain.enums import PipelineStage
from travelmovieai.domain.models import StageResult
from travelmovieai.infrastructure.artifacts import write_json_atomic
from travelmovieai.infrastructure.database import MediaAssetRepository
from travelmovieai.pipeline.base import Stage


class DuplicateDetectionStage(Stage):
    name = PipelineStage.DUPLICATE_DETECTION

    def run(self, context: ProjectContext) -> StageResult:
        repository = MediaAssetRepository(context.database_path)
        repository.initialize()
        report, scenes = detect_duplicate_scenes(repository.list_scenes())
        repository.synchronize_scenes(scenes)
        artifact = context.artifacts_dir / "duplicates.json"
        write_json_atomic(artifact, report)
        return StageResult(
            stage=self.name,
            skipped=not report.groups,
            artifacts=[context.database_path, artifact],
            message=(
                f"Duplicate detection found {report.duplicate_count} duplicate "
                f"scene(s) in {len(report.groups)} group(s)."
            ),
        )
