from pathlib import Path

from travelmovieai.core.exceptions import MediaProbeError
from travelmovieai.domain.enums import MediaType
from travelmovieai.infrastructure.ffmpeg import ProbeResult
from travelmovieai.media.scanner import MediaScanner


class FakeProbe:
    def __init__(self) -> None:
        self.paths: list[Path] = []

    def probe(self, path: Path) -> ProbeResult:
        self.paths.append(path)
        return ProbeResult(
            duration_seconds=3.5,
            width=1280,
            height=720,
            fps=25,
            metadata={"format_name": "fake"},
        )


class FailingProbe:
    def probe(self, path: Path) -> ProbeResult:
        raise MediaProbeError(f"Could not inspect {path.name}: corrupt media")


def test_scan_discovers_supported_files_and_reuses_cache(tmp_path: Path) -> None:
    media = tmp_path / "Моя поездка"
    media.mkdir()
    video = media / "clip one.MP4"
    video.write_bytes(b"video")
    (media / "notes.txt").write_text("ignore", encoding="utf-8")
    probe = FakeProbe()
    scanner = MediaScanner(probe)

    first = scanner.scan(media)
    second = scanner.scan(media, cached_assets=first.assets)

    assert first.discovered_count == 1
    assert first.probed_count == 1
    assert first.assets[0].media_type is MediaType.VIDEO
    assert first.assets[0].relative_path == Path("clip one.MP4")
    assert second.cached_count == 1
    assert second.probed_count == 0
    assert probe.paths == [video.resolve()]


def test_scan_excludes_workspace_inside_media_folder(tmp_path: Path) -> None:
    media = tmp_path / "media"
    workspace = media / "workspace"
    workspace.mkdir(parents=True)
    (media / "source.mp4").write_bytes(b"source")
    (workspace / "generated.mp4").write_bytes(b"generated")

    report = MediaScanner(FakeProbe()).scan(media, excluded_roots=(workspace,))

    assert [asset.relative_path for asset in report.assets] == [Path("source.mp4")]


def test_scan_records_probe_error_without_stopping_project(tmp_path: Path) -> None:
    media = tmp_path / "media"
    media.mkdir()
    (media / "broken.mp4").write_bytes(b"not a video")

    report = MediaScanner(FailingProbe()).scan(media)
    cached_report = MediaScanner(FailingProbe()).scan(media, cached_assets=report.assets)

    assert report.discovered_count == 1
    assert report.error_count == 1
    assert report.assets[0].scan_error == "Could not inspect broken.mp4: corrupt media"
    assert cached_report.probed_count == 0
    assert cached_report.cached_count == 1
    assert cached_report.error_count == 1
