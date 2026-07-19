"""Pipeline stage for media discovery and SQLite persistence."""

from travelmovieai.application.context import ProjectContext
from travelmovieai.domain.enums import PipelineStage, StageStatus
from travelmovieai.domain.models import StageResult
from travelmovieai.infrastructure.artifacts import write_json_atomic
from travelmovieai.infrastructure.database import MediaAssetRepository
from travelmovieai.infrastructure.ffmpeg import FFprobeClient
from travelmovieai.media.scanner import MediaProbe, MediaScanner
from travelmovieai.pipeline.base import Stage


class MediaScanStage(Stage):
    name = PipelineStage.MEDIA_SCAN

    def __init__(self, probe: MediaProbe | None = None) -> None:
        self._probe = probe

    def run(self, context: ProjectContext) -> StageResult:
        repository = MediaAssetRepository(context.database_path)
        repository.initialize()
        cached_assets = repository.list_assets()
        probe = self._probe or FFprobeClient(context.settings.ffprobe_binary)
        report = MediaScanner(probe).scan(
            context.input_path,
            cached_assets=cached_assets,
            excluded_roots=(context.workspace,),
            progress=context.progress,
        )
        repository.synchronize(report.assets, report.scanned_at)

        analysis_path = context.artifacts_dir / "analysis.json"
        write_json_atomic(analysis_path, report)

        return StageResult(
            stage=self.name,
            status=(
                StageStatus.CACHED
                if report.probed_count == 0 and report.cached_count > 0
                else StageStatus.NO_INPUT
                if report.discovered_count == 0
                else StageStatus.COMPLETED
            ),
            artifacts=[context.database_path, analysis_path],
            message=(
                f"Media scan found {report.discovered_count} file(s): "
                f"{report.probed_count} inspected, {report.cached_count} cached, "
                f"{report.error_count} with errors."
            ),
        )
