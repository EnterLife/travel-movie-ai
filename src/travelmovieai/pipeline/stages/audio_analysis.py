"""Pipeline stage for scene-level audio context classification."""

from travelmovieai.analysis.audio import analyze_audio
from travelmovieai.application.context import ProjectContext
from travelmovieai.domain.enums import PipelineStage
from travelmovieai.domain.models import StageResult
from travelmovieai.infrastructure.artifacts import write_json_atomic
from travelmovieai.infrastructure.database import MediaAssetRepository
from travelmovieai.pipeline.base import Stage


class AudioAnalysisStage(Stage):
    name = PipelineStage.AUDIO_ANALYSIS

    def run(self, context: ProjectContext) -> StageResult:
        repository = MediaAssetRepository(context.database_path)
        repository.initialize()
        report = analyze_audio(
            repository.list_scenes(),
            repository.list_assets(),
            context.settings.ffmpeg_binary,
        )
        repository.synchronize_scenes(report.scenes)
        artifact = context.artifacts_dir / "audio_analysis.json"
        write_json_atomic(artifact, report)
        return StageResult(
            stage=self.name,
            skipped=report.analyzed_count == 0,
            artifacts=[context.database_path, artifact],
            message=(
                f"Audio analysis classified {report.analyzed_count} scene(s), "
                f"{report.skipped_count} skipped."
            ),
        )
