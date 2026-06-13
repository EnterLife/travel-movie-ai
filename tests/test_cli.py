from pathlib import Path

from typer.testing import CliRunner

from travelmovieai.cli import app


def test_analyze_reports_media_scan_summary(tmp_path: Path) -> None:
    media = tmp_path / "media"
    media.mkdir()
    workspace = tmp_path / "workspace"

    result = CliRunner().invoke(
        app,
        ["analyze", "--input", str(media), "--workspace", str(workspace)],
    )

    assert result.exit_code == 0
    assert "Media scan found 0 file(s)" in result.stdout
