"""Project-local video proxies used only by analysis stages."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Literal, Protocol
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, ValidationError

from travelmovieai.core.exceptions import DependencyUnavailableError, MontageError
from travelmovieai.core.security import sanitize_process_error
from travelmovieai.domain.enums import MediaType
from travelmovieai.domain.models import MediaAsset
from travelmovieai.infrastructure.artifacts import artifact_fingerprint, write_json_atomic
from travelmovieai.infrastructure.ffmpeg import FFprobeClient, ProbeResult

PROXY_SCHEMA_VERSION: Literal["analysis-proxy-v1"] = "analysis-proxy-v1"


class MediaProbe(Protocol):
    def probe(self, path: Path) -> ProbeResult: ...


class AnalysisProxyDecision(BaseModel):
    """Typed explanation of whether a probed source needs an analysis proxy."""

    asset_id: UUID
    required: bool
    reason: Literal[
        "disabled",
        "not-video",
        "scan-error",
        "unknown-dimensions",
        "within-limit",
        "forced",
        "oversized",
    ]
    source_width: int | None = None
    source_height: int | None = None
    max_dimension: int = Field(ge=1)


class AnalysisMedia(BaseModel):
    """Transient media path for analysis while retaining the original identity."""

    asset_id: UUID
    source_path: Path
    analysis_path: Path
    width: int | None = None
    height: int | None = None
    duration_seconds: float | None = None
    proxied: bool = False
    cache_hit: bool = False
    cache_key: str | None = Field(default=None, min_length=64, max_length=64)


class AnalysisProxyManifest(BaseModel):
    schema_version: Literal["analysis-proxy-v1"] = PROXY_SCHEMA_VERSION
    asset_id: UUID
    source_fingerprint: str = Field(min_length=64, max_length=64)
    proxy_path: Path
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    duration_seconds: float | None = Field(default=None, gt=0)
    size_bytes: int = Field(gt=0)
    modified_ns: int = Field(ge=0)


class AnalysisProxyManager:
    """Resolve and atomically generate bounded-resolution analysis media."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        ffmpeg_binary: str = "ffmpeg",
        ffprobe_binary: str = "ffprobe",
        mode: Literal["auto", "disabled", "always"] = "auto",
        max_dimension: int = 1920,
        video_bitrate_mbps: float = 6.0,
        timeout_seconds: float = 3600,
        probe: MediaProbe | None = None,
    ) -> None:
        self.cache_dir = cache_dir
        self.ffmpeg_binary = ffmpeg_binary
        self.mode = mode
        self.max_dimension = max_dimension
        self.video_bitrate_mbps = video_bitrate_mbps
        self.timeout_seconds = timeout_seconds
        self._probe = probe or FFprobeClient(
            ffprobe_binary,
            timeout_seconds=min(60, timeout_seconds),
        )

    def decide(self, asset: MediaAsset) -> AnalysisProxyDecision:
        width = asset.width
        height = asset.height
        if (
            self.mode != "disabled"
            and asset.media_type is MediaType.VIDEO
            and not asset.scan_error
            and asset.path.is_file()
            and (width is None or height is None)
        ):
            probe = self._probe.probe(asset.path)
            width = width or probe.width
            height = height or probe.height
        return decide_analysis_proxy(
            asset,
            mode=self.mode,
            max_dimension=self.max_dimension,
            width=width,
            height=height,
        )

    def cache_identity(self) -> str:
        """Identify proxy behavior that can change decoded scene boundaries."""
        return artifact_fingerprint(
            {
                "schema": PROXY_SCHEMA_VERSION,
                "mode": self.mode,
                "max_dimension": self.max_dimension,
                "video_bitrate_mbps": self.video_bitrate_mbps,
                "ffmpeg_binary": self.ffmpeg_binary,
            }
        )

    def resolve(self, asset: MediaAsset) -> AnalysisMedia:
        decision = self.decide(asset)
        if not decision.required:
            return AnalysisMedia(
                asset_id=asset.id,
                source_path=asset.path,
                analysis_path=asset.path,
                width=decision.source_width,
                height=decision.source_height,
                duration_seconds=asset.duration_seconds,
            )

        fingerprint = _proxy_fingerprint(asset, self)
        proxy_path = self.cache_dir / f"{asset.id}-{fingerprint[:16]}.mp4"
        manifest_path = proxy_path.with_suffix(".json")
        cached = _read_valid_manifest(
            manifest_path,
            proxy_path=proxy_path,
            asset_id=asset.id,
            fingerprint=fingerprint,
            max_dimension=self.max_dimension,
        )
        if cached is not None:
            return _analysis_media(asset, cached, cache_hit=True)

        manifest = self._generate(asset, fingerprint, proxy_path, manifest_path)
        return _analysis_media(asset, manifest, cache_hit=False)

    def _generate(
        self,
        asset: MediaAsset,
        fingerprint: str,
        proxy_path: Path,
        manifest_path: Path,
    ) -> AnalysisProxyManifest:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        temporary_path = proxy_path.with_name(
            f".{proxy_path.stem}.{uuid4().hex}.tmp{proxy_path.suffix}"
        )
        command = self._command(asset.path, temporary_path)
        try:
            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    check=False,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.timeout_seconds,
                )
            except FileNotFoundError as error:
                raise DependencyUnavailableError(
                    f"FFmpeg executable was not found: {self.ffmpeg_binary}"
                ) from error
            except subprocess.TimeoutExpired as error:
                raise MontageError(
                    f"FFmpeg timed out after {self.timeout_seconds:g}s while creating an "
                    f"analysis proxy for {asset.relative_path}."
                ) from error
            if completed.returncode != 0 or not temporary_path.is_file():
                detail = sanitize_process_error(
                    completed.stderr,
                    private_paths=[asset.path, temporary_path, proxy_path],
                    fallback="FFmpeg did not create an output file.",
                )
                raise MontageError(
                    f"Could not create an analysis proxy for {asset.relative_path}: {detail}"
                )
            probe = self._probe.probe(temporary_path)
            if (
                probe.width is None
                or probe.height is None
                or max(probe.width, probe.height) > self.max_dimension
            ):
                raise MontageError(
                    f"The analysis proxy for {asset.relative_path} has invalid dimensions."
                )
            size_bytes = temporary_path.stat().st_size
            if size_bytes <= 0:
                raise MontageError(f"The analysis proxy for {asset.relative_path} is empty.")
            try:
                os.replace(temporary_path, proxy_path)
                proxy_stat = proxy_path.stat()
                manifest = AnalysisProxyManifest(
                    asset_id=asset.id,
                    source_fingerprint=fingerprint,
                    proxy_path=proxy_path,
                    width=probe.width,
                    height=probe.height,
                    duration_seconds=probe.duration_seconds,
                    size_bytes=proxy_stat.st_size,
                    modified_ns=proxy_stat.st_mtime_ns,
                )
                write_json_atomic(manifest_path, manifest)
                return manifest
            except OSError as error:
                proxy_path.unlink(missing_ok=True)
                manifest_path.unlink(missing_ok=True)
                raise MontageError(
                    f"Could not finalize the analysis proxy for {asset.relative_path}."
                ) from error
        finally:
            temporary_path.unlink(missing_ok=True)

    def _command(self, source_path: Path, output_path: Path) -> list[str]:
        scale = (
            f"scale={self.max_dimension}:{self.max_dimension}:"
            "force_original_aspect_ratio=decrease,"
            "scale=trunc(iw/2)*2:trunc(ih/2)*2"
        )
        return [
            self.ffmpeg_binary,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source_path),
            "-map",
            "0:v:0",
            "-map_metadata",
            "-1",
            "-an",
            "-sn",
            "-dn",
            "-vf",
            scale,
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-b:v",
            f"{self.video_bitrate_mbps:g}M",
            "-maxrate",
            f"{self.video_bitrate_mbps:g}M",
            "-bufsize",
            f"{self.video_bitrate_mbps * 2:g}M",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ]


def decide_analysis_proxy(
    asset: MediaAsset,
    *,
    mode: Literal["auto", "disabled", "always"],
    max_dimension: int,
    width: int | None = None,
    height: int | None = None,
) -> AnalysisProxyDecision:
    """Decide from FFprobe-derived dimensions without touching the source."""

    source_width = width if width is not None else asset.width
    source_height = height if height is not None else asset.height
    reason: Literal[
        "disabled",
        "not-video",
        "scan-error",
        "unknown-dimensions",
        "within-limit",
        "forced",
        "oversized",
    ]
    if mode == "disabled":
        reason = "disabled"
    elif asset.media_type is not MediaType.VIDEO:
        reason = "not-video"
    elif asset.scan_error:
        reason = "scan-error"
    elif mode == "always":
        reason = "forced"
    elif source_width is None or source_height is None:
        reason = "unknown-dimensions"
    elif max(source_width, source_height) > max_dimension:
        reason = "oversized"
    else:
        reason = "within-limit"
    return AnalysisProxyDecision(
        asset_id=asset.id,
        required=reason in {"forced", "oversized"},
        reason=reason,
        source_width=source_width,
        source_height=source_height,
        max_dimension=max_dimension,
    )


def _proxy_fingerprint(asset: MediaAsset, manager: AnalysisProxyManager) -> str:
    return artifact_fingerprint(
        {
            "schema": PROXY_SCHEMA_VERSION,
            "asset_id": asset.id,
            "relative_path": asset.relative_path,
            "size_bytes": asset.size_bytes,
            "modified_ns": asset.modified_ns,
            "width": asset.width,
            "height": asset.height,
            "duration_seconds": asset.duration_seconds,
            "max_dimension": manager.max_dimension,
            "video_bitrate_mbps": manager.video_bitrate_mbps,
            "ffmpeg_binary": manager.ffmpeg_binary,
        }
    )


def _read_valid_manifest(
    manifest_path: Path,
    *,
    proxy_path: Path,
    asset_id: UUID,
    fingerprint: str,
    max_dimension: int,
) -> AnalysisProxyManifest | None:
    if not manifest_path.is_file() or not proxy_path.is_file():
        return None
    try:
        manifest = AnalysisProxyManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        proxy_stat = proxy_path.stat()
    except (OSError, ValidationError):
        return None
    if (
        manifest.asset_id != asset_id
        or manifest.source_fingerprint != fingerprint
        or manifest.proxy_path.resolve() != proxy_path.resolve()
        or manifest.size_bytes != proxy_stat.st_size
        or manifest.modified_ns != proxy_stat.st_mtime_ns
        or max(manifest.width, manifest.height) > max_dimension
    ):
        return None
    return manifest


def _analysis_media(
    asset: MediaAsset,
    manifest: AnalysisProxyManifest,
    *,
    cache_hit: bool,
) -> AnalysisMedia:
    return AnalysisMedia(
        asset_id=asset.id,
        source_path=asset.path,
        analysis_path=manifest.proxy_path,
        width=manifest.width,
        height=manifest.height,
        duration_seconds=manifest.duration_seconds or asset.duration_seconds,
        proxied=True,
        cache_hit=cache_hit,
        cache_key=manifest.source_fingerprint,
    )
