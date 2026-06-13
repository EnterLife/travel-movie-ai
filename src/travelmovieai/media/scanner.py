"""Recursive media discovery and metadata extraction."""

import os
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from PIL import ExifTags, Image

from travelmovieai.core.exceptions import MediaProbeError
from travelmovieai.domain.enums import MediaType
from travelmovieai.domain.models import MediaAsset, MediaScanReport
from travelmovieai.infrastructure.ffmpeg import ProbeResult

VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".avi", ".mkv", ".m4v"})
PHOTO_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".heic"})
AUDIO_EXTENSIONS = frozenset({".mp3", ".wav", ".flac", ".m4a"})

MEDIA_TYPES = {
    **dict.fromkeys(VIDEO_EXTENSIONS, MediaType.VIDEO),
    **dict.fromkeys(PHOTO_EXTENSIONS, MediaType.PHOTO),
    **dict.fromkeys(AUDIO_EXTENSIONS, MediaType.AUDIO),
}


class MediaProbe(Protocol):
    def probe(self, path: Path) -> ProbeResult: ...


class MediaScanner:
    def __init__(self, probe: MediaProbe) -> None:
        self.probe = probe

    def scan(
        self,
        input_path: Path,
        *,
        cached_assets: Sequence[MediaAsset] = (),
        excluded_roots: Sequence[Path] = (),
    ) -> MediaScanReport:
        root = input_path.resolve()
        cached_by_path = {_path_key(asset.path): asset for asset in cached_assets}
        excluded = tuple(path.resolve() for path in excluded_roots)
        assets: list[MediaAsset] = []
        probed_count = 0
        cached_count = 0
        error_count = 0

        for path in _discover_media(root, excluded):
            stat = path.stat()
            cached = cached_by_path.get(_path_key(path))
            if (
                cached is not None
                and cached.size_bytes == stat.st_size
                and cached.modified_ns == stat.st_mtime_ns
            ):
                assets.append(
                    cached.model_copy(
                        update={
                            "path": path,
                            "relative_path": path.relative_to(root),
                        }
                    )
                )
                cached_count += 1
                if cached.scan_error:
                    error_count += 1
                continue

            probed_count += 1
            asset = self._inspect_asset(root, path, stat)
            if asset.scan_error:
                error_count += 1
            assets.append(asset)

        return MediaScanReport(
            input_path=root,
            scanned_at=datetime.now(UTC),
            assets=assets,
            discovered_count=len(assets),
            probed_count=probed_count,
            cached_count=cached_count,
            error_count=error_count,
        )

    def _inspect_asset(self, root: Path, path: Path, stat: os.stat_result) -> MediaAsset:
        extension = path.suffix.lower()
        probe_result = ProbeResult()
        scan_error: str | None = None
        try:
            probe_result = self.probe.probe(path)
        except MediaProbeError as error:
            scan_error = str(error)

        image_result = (
            _read_image_metadata(path) if MEDIA_TYPES[extension] is MediaType.PHOTO else None
        )
        created_at = probe_result.created_at or _filesystem_created_at(stat)
        width = probe_result.width
        height = probe_result.height
        latitude = probe_result.latitude
        longitude = probe_result.longitude
        if image_result:
            width = width or image_result.width
            height = height or image_result.height
            latitude = latitude if latitude is not None else image_result.latitude
            longitude = longitude if longitude is not None else image_result.longitude

        return MediaAsset(
            path=path,
            relative_path=path.relative_to(root),
            media_type=MEDIA_TYPES[extension],
            extension=extension,
            size_bytes=stat.st_size,
            modified_at=datetime.fromtimestamp(stat.st_mtime, UTC),
            modified_ns=stat.st_mtime_ns,
            created_at=created_at,
            duration_seconds=probe_result.duration_seconds,
            width=width,
            height=height,
            fps=probe_result.fps,
            latitude=latitude,
            longitude=longitude,
            probe_metadata=probe_result.metadata,
            scan_error=scan_error,
        )


def _discover_media(root: Path, excluded_roots: Sequence[Path]) -> list[Path]:
    discovered: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in MEDIA_TYPES:
            continue
        resolved = path.resolve()
        if any(resolved.is_relative_to(excluded) for excluded in excluded_roots):
            continue
        discovered.append(resolved)
    return sorted(discovered, key=lambda path: path.relative_to(root).as_posix().casefold())


def _filesystem_created_at(stat: os.stat_result) -> datetime:
    return datetime.fromtimestamp(stat.st_ctime, UTC)


def _path_key(path: Path) -> str:
    return os.path.normcase(str(path.resolve()))


class _ImageMetadata:
    def __init__(
        self,
        width: int,
        height: int,
        latitude: float | None,
        longitude: float | None,
    ) -> None:
        self.width = width
        self.height = height
        self.latitude = latitude
        self.longitude = longitude


def _read_image_metadata(path: Path) -> _ImageMetadata | None:
    try:
        with Image.open(path) as image:
            latitude, longitude = _read_exif_gps(image)
            return _ImageMetadata(image.width, image.height, latitude, longitude)
    except (OSError, SyntaxError, ValueError):
        return None


def _read_exif_gps(image: Image.Image) -> tuple[float | None, float | None]:
    exif = image.getexif()
    if not exif:
        return None, None
    gps = exif.get_ifd(ExifTags.IFD.GPSInfo)
    if not gps:
        return None, None

    latitude = _gps_coordinate(
        gps.get(ExifTags.GPS.GPSLatitude),
        gps.get(ExifTags.GPS.GPSLatitudeRef),
    )
    longitude = _gps_coordinate(
        gps.get(ExifTags.GPS.GPSLongitude),
        gps.get(ExifTags.GPS.GPSLongitudeRef),
    )
    return latitude, longitude


def _gps_coordinate(value: object, reference: object) -> float | None:
    if not isinstance(value, tuple) or len(value) != 3:
        return None
    try:
        degrees, minutes, seconds = (float(part) for part in value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    coordinate = degrees + minutes / 60 + seconds / 3600
    if reference in {"S", "W", b"S", b"W"}:
        coordinate *= -1
    return coordinate
