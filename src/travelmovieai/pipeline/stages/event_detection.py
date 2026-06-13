"""Pipeline stage for semantic event clustering."""

from travelmovieai.application.context import ProjectContext
from travelmovieai.domain.enums import PipelineStage
from travelmovieai.domain.models import StageResult
from travelmovieai.infrastructure.artifacts import write_json_atomic
from travelmovieai.infrastructure.database import MediaAssetRepository
from travelmovieai.pipeline.base import Stage
from travelmovieai.story.events import detect_events


class EventDetectionStage(Stage):
    name = PipelineStage.EVENT_DETECTION

    def run(self, context: ProjectContext) -> StageResult:
        repository = MediaAssetRepository(context.database_path)
        repository.initialize()
        report, scenes = detect_events(
            repository.list_scenes(),
            repository.list_assets(),
        )
        repository.synchronize_scenes(scenes)
        repository.synchronize_events(report.events)
        artifact = context.artifacts_dir / "events.json"
        write_json_atomic(artifact, report)
        return StageResult(
            stage=self.name,
            skipped=not report.events,
            artifacts=[context.database_path, artifact],
            message=f"Event detection produced {len(report.events)} event(s).",
        )
