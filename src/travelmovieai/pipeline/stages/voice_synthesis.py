"""Explicit optional boundary for future local voice synthesis."""

from travelmovieai.application.context import ProjectContext
from travelmovieai.domain.enums import PipelineStage
from travelmovieai.domain.models import StageResult
from travelmovieai.pipeline.base import Stage


class VoiceSynthesisStage(Stage):
    name = PipelineStage.VOICE_SYNTHESIS

    def run(self, context: ProjectContext) -> StageResult:
        narration = context.artifacts_dir / "narration.json"
        return StageResult(
            stage=self.name,
            skipped=True,
            artifacts=[narration] if narration.is_file() else [],
            message=(
                "Voice synthesis is disabled because no local voice provider is configured; "
                "text narration remains available."
            ),
        )
