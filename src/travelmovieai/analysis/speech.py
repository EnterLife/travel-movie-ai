"""Scene-level speech extraction and Faster Whisper orchestration."""

import hashlib
import json
import os
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from travelmovieai.core.exceptions import DependencyUnavailableError, PipelineStageError
from travelmovieai.core.security import sanitize_process_error
from travelmovieai.domain.enums import MediaType
from travelmovieai.domain.models import (
    MediaAsset,
    Scene,
    SpeechAnalysisReport,
    SpeechTranscript,
)


class SpeechProvider(Protocol):
    name: str
    model: str

    def transcribe(self, audio_path: Path) -> SpeechTranscript: ...


def analyze_speech(
    scenes: list[Scene],
    assets: list[MediaAsset],
    provider: SpeechProvider,
    ffmpeg_binary: str,
    audio_dir: Path,
    progress: Callable[[int, int, str], None] | None = None,
    *,
    timeout_seconds: float = 120,
) -> SpeechAnalysisReport:
    assets_by_id = {asset.id: asset for asset in assets}
    updated: list[Scene] = []
    transcribed_count = 0
    cached_count = 0
    total = len(scenes)
    for index, scene in enumerate(scenes, start=1):
        asset = assets_by_id.get(scene.asset_id)
        if asset is None or asset.media_type is not MediaType.VIDEO or not _has_audio(asset):
            updated.append(scene)
            if progress:
                progress(index, total, f"Whisper: scene {index}/{total}, no speech")
            continue
        cache_key = _speech_cache_key(scene, asset, provider.model)
        if scene.metadata.get("speech_cache_key") == cache_key and scene.transcript is not None:
            updated.append(scene)
            cached_count += 1
            if progress:
                progress(index, total, f"Whisper cache: scene {index}/{total}")
            continue
        if progress:
            progress(index - 1, total, f"Whisper: scene {index}/{total}")
        audio_path = audio_dir / f"{scene.id}-{cache_key[:12]}.wav"
        _extract_scene_audio(
            ffmpeg_binary,
            asset.path,
            scene,
            audio_path,
            timeout_seconds,
        )
        transcript = provider.transcribe(audio_path)
        updated.append(
            scene.model_copy(
                update={
                    "transcript": transcript.text,
                    "metadata": {
                        **scene.metadata,
                        "speech_cache_key": cache_key,
                        "speech_provider": provider.name,
                        "speech_model": provider.model,
                        "speech_language": transcript.language,
                        "speech_confidence": transcript.confidence,
                        "speech_segments": [
                            segment.model_dump(mode="json") for segment in transcript.segments
                        ],
                    },
                }
            )
        )
        transcribed_count += 1
        if progress:
            progress(index, total, f"Whisper: complete {index}/{total}")
    return SpeechAnalysisReport(
        created_at=datetime.now(UTC),
        provider=provider.name,
        model=provider.model,
        scenes=updated,
        transcribed_count=transcribed_count,
        cached_count=cached_count,
    )


def _extract_scene_audio(
    ffmpeg_binary: str,
    source_path: Path,
    scene: Scene,
    output_path: Path,
    timeout_seconds: float = 120,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{uuid4().hex}.tmp.wav")
    command = [
        ffmpeg_binary,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{scene.start_seconds:.3f}",
        "-t",
        f"{scene.end_seconds - scene.start_seconds:.3f}",
        "-i",
        str(source_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(temporary),
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            check=False,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
    except FileNotFoundError as error:
        raise DependencyUnavailableError(
            f"FFmpeg executable was not found: {ffmpeg_binary}"
        ) from error
    except subprocess.TimeoutExpired as error:
        temporary.unlink(missing_ok=True)
        raise PipelineStageError(
            f"FFmpeg timed out after {timeout_seconds:g}s while extracting speech "
            f"from {source_path.name}."
        ) from error
    try:
        if completed.returncode != 0:
            detail = sanitize_process_error(
                completed.stderr,
                private_paths=[source_path, temporary, output_path],
                fallback="unknown FFmpeg error",
            )
            raise PipelineStageError(f"Could not extract speech from {source_path.name}: {detail}")
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)


def _speech_cache_key(scene: Scene, asset: MediaAsset, model: str) -> str:
    payload = {
        "asset": str(asset.id),
        "size": asset.size_bytes,
        "modified_ns": asset.modified_ns,
        "start": scene.start_seconds,
        "end": scene.end_seconds,
        "model": model,
        "version": 2,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _has_audio(asset: MediaAsset) -> bool:
    return any(
        stream.get("codec_type") == "audio" for stream in asset.probe_metadata.get("streams", [])
    )
