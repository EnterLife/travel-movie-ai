from pathlib import Path

import pytest
from typer.testing import CliRunner

from travelmovieai import cli
from travelmovieai.application.diagnostics import DiagnosticCheck, SystemDiagnosticReport
from travelmovieai.cli import app
from travelmovieai.domain.enums import PipelineStage
from travelmovieai.domain.models import QuickMontageSettings, StageResult


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
    assert "Starting media scan" not in result.stdout
    assert "Starting media scan" in result.stderr
    assert "[100%]" in result.stderr


def test_analyze_reports_unsafe_workspace_without_traceback(tmp_path: Path) -> None:
    media = tmp_path / "media"
    media.mkdir()

    result = CliRunner().invoke(
        app,
        ["analyze", "--input", str(media), "--workspace", str(tmp_path)],
    )

    assert result.exit_code == 1
    assert "Workspace" in result.stderr
    assert "nested" in result.stderr
    assert "Traceback" not in result.stderr


def test_analyze_rejects_workspace_inside_source_without_traceback(tmp_path: Path) -> None:
    media = tmp_path / "media"
    media.mkdir()

    result = CliRunner().invoke(
        app,
        ["analyze", "--input", str(media), "--workspace", str(media / "workspace")],
    )

    assert result.exit_code == 1
    assert "Workspace" in result.stderr
    assert "nested" in result.stderr
    assert "Traceback" not in result.stderr


def test_estimate_command_scans_metadata_and_prints_typed_json(tmp_path: Path) -> None:
    media = tmp_path / "media"
    media.mkdir()
    workspace = tmp_path / "workspace"

    result = CliRunner().invoke(
        app,
        [
            "estimate",
            "--input",
            str(media),
            "--workspace",
            str(workspace),
            "--semantic",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert '"estimated_peak_workspace_bytes"' in result.stdout
    assert '"runtime"' in result.stdout


def test_report_command_creates_offline_html(tmp_path: Path) -> None:
    media = tmp_path / "media"
    media.mkdir()
    workspace = tmp_path / "workspace"

    result = CliRunner().invoke(
        app,
        ["report", "--input", str(media), "--workspace", str(workspace)],
    )

    assert result.exit_code == 0
    assert "HTML report generated" in result.stdout
    assert (workspace / "artifacts" / "report.html").is_file()


def test_search_command_reports_missing_index_without_traceback(tmp_path: Path) -> None:
    media = tmp_path / "media"
    media.mkdir()
    workspace = tmp_path / "workspace"

    result = CliRunner().invoke(
        app,
        [
            "search",
            "--input",
            str(media),
            "mountain sunrise",
            "--workspace",
            str(workspace),
        ],
    )

    assert result.exit_code == 1
    assert "embeddings.json" in result.stderr
    assert "Traceback" not in result.stderr


def test_export_and_restore_commands_round_trip_project_metadata(tmp_path: Path) -> None:
    media = tmp_path / "media"
    media.mkdir()
    workspace = tmp_path / "workspace"
    initialized = CliRunner().invoke(
        app,
        ["analyze", "--input", str(media), "--workspace", str(workspace)],
    )
    artifacts = workspace / "artifacts"
    (artifacts / "note.json").write_text('{"local": true}', encoding="utf-8")
    archive = tmp_path / "project-backup.zip"

    exported = CliRunner().invoke(
        app,
        [
            "export",
            "--input",
            str(media),
            "--workspace",
            str(workspace),
            "--output",
            str(archive),
        ],
    )
    restored_workspace = tmp_path / "restored"
    restored = CliRunner().invoke(
        app,
        [
            "restore",
            "--archive",
            str(archive),
            "--workspace",
            str(restored_workspace),
        ],
    )

    assert initialized.exit_code == 0
    assert exported.exit_code == 0
    assert restored.exit_code == 0
    assert (restored_workspace / "artifacts" / "note.json").is_file()


def test_doctor_returns_failure_for_required_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeService:
        def diagnostics(self) -> SystemDiagnosticReport:
            return SystemDiagnosticReport(
                ready=False,
                checks=(DiagnosticCheck("FFmpeg", "error", "not found"),),
            )

    monkeypatch.setattr(cli, "_service", lambda: FakeService())

    result = CliRunner().invoke(app, ["doctor"], color=False)

    assert result.exit_code == 1
    assert "[FAIL] FFmpeg: not found" in result.stdout


def test_create_command_plumbs_advanced_render_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media = tmp_path / "media"
    media.mkdir()
    captured: dict[str, object] = {}

    class FakeService:
        def create(self, **kwargs: object) -> StageResult:
            captured.update(kwargs)
            return StageResult(stage=PipelineStage.RENDERING, message="created")

    monkeypatch.setattr(cli, "_service", lambda: FakeService())
    result = CliRunner().invoke(
        app,
        [
            "create",
            "--input",
            str(media),
            "--output",
            str(tmp_path / "movie.mp4"),
            "--semantic",
            "--analysis-quality",
            "deep",
            "--width",
            "1920",
            "--height",
            "1080",
            "--fps",
            "24",
            "--validate-full-render-decode",
            "--variant",
            "vertical cut",
            "--framing",
            "smart",
            "--vertical-layout",
            "blur",
            "--photo-motion",
            "ken_burns",
            "--color-normalization",
            "--hdr-to-sdr",
            "--text-overlays",
            "--event-titles",
            "--subtitles",
            "--credits",
            "Local trip",
            "--bpm-analysis",
            "--music-envelope",
            "--narration",
        ],
    )

    settings = captured["montage_settings"]
    assert result.exit_code == 0
    assert isinstance(settings, QuickMontageSettings)
    assert settings.width == 1920
    assert settings.height == 1080
    assert settings.fps == 24
    assert settings.validate_full_render_decode is True
    assert settings.framing_mode == "smart"
    assert settings.analysis_quality_mode == "deep"
    assert settings.vertical_video_layout == "blur"
    assert settings.photo_motion == "ken_burns"
    assert settings.color_normalization is True
    assert settings.hdr_to_sdr is True
    assert settings.text_overlays_enabled is True
    assert settings.event_titles_enabled is True
    assert settings.scene_subtitles_enabled is True
    assert settings.music_bpm_analysis is True
    assert settings.music_volume_envelope is True
    assert settings.narration_enabled is True
    assert captured["variant_name"] == "vertical cut"


def test_create_command_reports_invalid_advanced_setting_without_traceback(
    tmp_path: Path,
) -> None:
    media = tmp_path / "media"
    media.mkdir()

    result = CliRunner().invoke(
        app,
        [
            "create",
            "--input",
            str(media),
            "--output",
            str(tmp_path / "movie.mp4"),
            "--framing",
            "telepathic",
        ],
    )

    assert result.exit_code == 1
    assert "framing_mode" in result.stderr
    assert "Traceback" not in result.stderr


def test_create_command_rejects_invalid_analysis_quality_without_traceback(
    tmp_path: Path,
) -> None:
    media = tmp_path / "media"
    media.mkdir()

    result = CliRunner().invoke(
        app,
        [
            "create",
            "--input",
            str(media),
            "--output",
            str(tmp_path / "movie.mp4"),
            "--semantic",
            "--analysis-quality",
            "extreme",
        ],
    )

    assert result.exit_code == 1
    assert "analysis_quality_mode" in result.stderr
    assert "Traceback" not in result.stderr


def test_create_command_rejects_odd_output_width_without_traceback(
    tmp_path: Path,
) -> None:
    media = tmp_path / "media"
    media.mkdir()

    result = CliRunner().invoke(
        app,
        [
            "create",
            "--input",
            str(media),
            "--output",
            str(tmp_path / "movie.mp4"),
            "--width",
            "1279",
        ],
    )

    assert result.exit_code == 1
    assert "width" in result.stderr
    assert "multiple of 2" in result.stderr
    assert "Traceback" not in result.stderr
