import json
from pathlib import Path

from travelmovieai.application.context import ProjectContext
from travelmovieai.core.config import Settings
from travelmovieai.infrastructure.database import MediaAssetRepository
from travelmovieai.infrastructure.ffmpeg import ProbeResult
from travelmovieai.pipeline.stages.media_scan import MediaScanStage


class FakeProbe:
    def probe(self, path: Path) -> ProbeResult:
        return ProbeResult(duration_seconds=1.25, metadata={"format_name": "fake"})


def test_media_scan_stage_writes_database_and_analysis_artifact(tmp_path: Path) -> None:
    input_path = tmp_path / "media"
    input_path.mkdir()
    (input_path / "clip.mp4").write_bytes(b"video")
    context = ProjectContext(
        input_path=input_path,
        workspace=tmp_path / "workspace",
        settings=Settings(),
    )
    context.prepare()

    result = MediaScanStage(FakeProbe()).run(context)

    assert result.skipped is False
    assert context.database_path in result.artifacts
    analysis_path = context.artifacts_dir / "analysis.json"
    payload = json.loads(analysis_path.read_text(encoding="utf-8"))
    assert payload["discovered_count"] == 1
    assert payload["assets"][0]["relative_path"] == "clip.mp4"
    assert len(MediaAssetRepository(context.database_path).list_assets()) == 1
