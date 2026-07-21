from pathlib import Path

from PIL import Image

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


class InvalidLocationProbe:
    def probe(self, path: Path) -> ProbeResult:
        del path
        return ProbeResult(
            duration_seconds=1,
            latitude=91,
            longitude=37,
        )


class MismatchedVideoDurationProbe:
    def probe(self, path: Path) -> ProbeResult:
        del path
        return ProbeResult(
            duration_seconds=7,
            video_duration_seconds=6.189517,
            width=1280,
            height=720,
            fps=59.94,
            metadata={"format_name": "mov,mp4"},
        )


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


def test_scan_reports_each_discovered_asset_progress(tmp_path: Path) -> None:
    media = tmp_path / "media"
    media.mkdir()
    (media / "a.mp4").write_bytes(b"a")
    (media / "b.mp4").write_bytes(b"b")
    events: list[tuple[int, int, str]] = []

    report = MediaScanner(FakeProbe()).scan(
        media,
        progress=lambda current, total, message: events.append((current, total, message)),
    )

    assert report.discovered_count == 2
    assert events == [
        (1, 2, "Media scan: 1/2"),
        (2, 2, "Media scan: 2/2"),
    ]


def test_scan_uses_primary_video_duration_instead_of_longer_container_duration(
    tmp_path: Path,
) -> None:
    media = tmp_path / "media"
    media.mkdir()
    (media / "clip.mp4").write_bytes(b"video")

    report = MediaScanner(MismatchedVideoDurationProbe()).scan(media)

    assert report.assets[0].duration_seconds == 6.189517


def test_scan_reprobes_cached_assets_from_older_duration_contract(tmp_path: Path) -> None:
    media = tmp_path / "media"
    media.mkdir()
    (media / "clip.mp4").write_bytes(b"video")
    legacy = (
        MediaScanner(FakeProbe())
        .scan(media)
        .assets[0]
        .model_copy(update={"probe_metadata": {"format_name": "fake"}})
    )

    report = MediaScanner(MismatchedVideoDurationProbe()).scan(media, cached_assets=[legacy])

    assert report.probed_count == 1
    assert report.cached_count == 0
    assert report.assets[0].duration_seconds == 6.189517


def test_scan_preserves_asset_identity_when_existing_file_changes(tmp_path: Path) -> None:
    media = tmp_path / "media"
    media.mkdir()
    video = media / "clip.mp4"
    video.write_bytes(b"first version")
    probe = FakeProbe()
    scanner = MediaScanner(probe)

    first = scanner.scan(media)
    video.write_bytes(b"second version with a different size")
    second = scanner.scan(media, cached_assets=first.assets)

    assert second.probed_count == 1
    assert second.cached_count == 0
    assert second.assets[0].id == first.assets[0].id


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
    assert cached_report.probed_count == 1
    assert cached_report.cached_count == 0
    assert cached_report.error_count == 1


def test_scan_retries_unchanged_probe_error_and_recovers_identity(tmp_path: Path) -> None:
    media = tmp_path / "Поездка"
    media.mkdir()
    video = media / "неисправный клип.mp4"
    video.write_bytes(b"video")

    failed = MediaScanner(FailingProbe()).scan(media)
    recovered = MediaScanner(FakeProbe()).scan(media, cached_assets=failed.assets)

    assert recovered.probed_count == 1
    assert recovered.cached_count == 0
    assert recovered.error_count == 0
    assert recovered.assets[0].id == failed.assets[0].id
    assert recovered.assets[0].scan_error is None
    assert recovered.assets[0].duration_seconds == 3.5


def test_scan_accepts_readable_photo_when_ffprobe_cannot_inspect_it(tmp_path: Path) -> None:
    media = tmp_path / "Фото"
    media.mkdir()
    photo = media / "вид на море.jpg"
    Image.new("RGB", (16, 9), color=(20, 80, 140)).save(photo)

    report = MediaScanner(FailingProbe()).scan(media)

    assert report.error_count == 0
    assert report.assets[0].scan_error is None
    assert report.assets[0].width == 16
    assert report.assets[0].height == 9


def test_scan_ignores_invalid_probe_coordinates_without_stopping(tmp_path: Path) -> None:
    media = tmp_path / "media"
    media.mkdir()
    (media / "bad-gps.mp4").write_bytes(b"video")

    report = MediaScanner(InvalidLocationProbe()).scan(media)

    assert report.discovered_count == 1
    assert report.error_count == 0
    assert report.assets[0].latitude is None
    assert report.assets[0].longitude is None
