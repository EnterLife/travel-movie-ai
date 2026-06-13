"""Pipeline stage for optional scene-level Faster Whisper transcription."""

from travelmovieai.analysis.speech import analyze_speech
from travelmovieai.application.context import ProjectContext
from travelmovieai.domain.enums import PipelineStage
from travelmovieai.domain.models import StageResult
from travelmovieai.infrastructure.artifacts import write_json_atomic
from travelmovieai.infrastructure.database import MediaAssetRepository
from travelmovieai.infrastructure.whisper import FasterWhisperProvider
from travelmovieai.pipeline.base import Stage


class SpeechAnalysisStage(Stage):
    name = PipelineStage.SPEECH_ANALYSIS

    def run(self, context: ProjectContext) -> StageResult:
        repository = MediaAssetRepository(context.database_path)
        repository.initialize()
        report = analyze_speech(
            repository.list_scenes(),
            repository.list_assets(),
            FasterWhisperProvider(
                context.settings.whisper_model,
                context.settings.device,
            ),
            context.settings.ffmpeg_binary,
            context.cache_dir / "speech",
        )
        repository.synchronize_scenes(report.scenes)
        artifact = context.artifacts_dir / "speech_analysis.json"
        write_json_atomic(artifact, report)
        return StageResult(
            stage=self.name,
            skipped=report.transcribed_count == 0,
            artifacts=[context.database_path, artifact],
            message=(
                f"Speech analysis transcribed {report.transcribed_count} scene(s), "
                f"{report.cached_count} cached."
            ),
        )
