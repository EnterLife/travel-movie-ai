"""Scene-level audio context and importance analysis."""

import math
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Literal, cast
from uuid import UUID

import numpy as np
from numpy.typing import NDArray

from travelmovieai.core.exceptions import DependencyUnavailableError, PipelineStageError
from travelmovieai.core.security import sanitize_process_error
from travelmovieai.domain.enums import MediaType
from travelmovieai.domain.models import (
    AudioAnalysisReport,
    AudioSceneAnalysis,
    MediaAsset,
    Scene,
)

SAMPLE_RATE = 16000
AudioLabel = Literal[
    "speech",
    "silence",
    "wind",
    "music",
    "crowd",
    "water",
    "transport",
    "ambient",
    "unknown",
]


def analyze_audio(
    scenes: list[Scene],
    assets: list[MediaAsset],
    ffmpeg_binary: str = "ffmpeg",
    progress: Callable[[int, int, str], None] | None = None,
    *,
    timeout_seconds: float = 120,
) -> AudioAnalysisReport:
    assets_by_id = {asset.id: asset for asset in assets}
    analyzed: list[Scene] = []
    analyses: list[AudioSceneAnalysis] = []
    analyzed_count = 0
    skipped_count = 0
    for index, scene in enumerate(scenes, start=1):
        asset = assets_by_id.get(scene.asset_id)
        analysis = _analyze_scene_audio(scene, asset, ffmpeg_binary, timeout_seconds)
        analyses.append(analysis)
        if analysis.has_audio:
            analyzed_count += 1
        else:
            skipped_count += 1
        analyzed.append(_scene_with_audio_analysis(scene, analysis))
        if progress:
            progress(
                index,
                len(scenes),
                f"Audio Analysis: scene {index}/{len(scenes)} · {analysis.primary_label}",
            )
    return AudioAnalysisReport(
        created_at=datetime.now(UTC),
        scenes=analyzed,
        analyses=analyses,
        analyzed_count=analyzed_count,
        skipped_count=skipped_count,
    )


def _analyze_scene_audio(
    scene: Scene,
    asset: MediaAsset | None,
    ffmpeg_binary: str,
    timeout_seconds: float,
) -> AudioSceneAnalysis:
    if asset is None or asset.media_type is not MediaType.VIDEO or not _has_audio(asset):
        return AudioSceneAnalysis(
            scene_id=scene.id,
            has_audio=False,
            primary_label="unknown",
            labels=[],
        )
    samples = _decode_scene_audio(scene, asset, ffmpeg_binary, timeout_seconds)
    if samples.size == 0:
        return AudioSceneAnalysis(
            scene_id=scene.id,
            has_audio=False,
            primary_label="silence",
            labels=["silence"],
        )
    return classify_audio_samples(scene.id, samples, transcript=scene.transcript)


def classify_audio_samples(
    scene_id: UUID,
    samples: NDArray[np.float64],
    *,
    transcript: str | None = None,
) -> AudioSceneAnalysis:
    duration_seconds = max(samples.size / SAMPLE_RATE, 0.001)
    rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    rms_dbfs = _dbfs(rms)
    peak_dbfs = _dbfs(peak)
    windows = _window_features(samples)
    centroid = _spectral_centroid(samples)
    low_ratio, high_ratio = _frequency_ratios(samples)
    zcr = _zero_crossing_rate(samples)
    dynamic_range = _dynamic_range(windows)
    speech_likelihood = _speech_likelihood(
        transcript=transcript,
        rms_dbfs=rms_dbfs,
        centroid=centroid,
        low_ratio=low_ratio,
        high_ratio=high_ratio,
        dynamic_range=dynamic_range,
    )
    noise_score = _noise_score(
        rms_dbfs=rms_dbfs,
        zcr=zcr,
        high_ratio=high_ratio,
        low_ratio=low_ratio,
        dynamic_range=dynamic_range,
    )
    primary, labels = _labels(
        rms_dbfs=rms_dbfs,
        centroid=centroid,
        low_ratio=low_ratio,
        high_ratio=high_ratio,
        dynamic_range=dynamic_range,
        zcr=zcr,
        speech_likelihood=speech_likelihood,
        transcript=transcript,
    )
    ambience_score = _ambience_score(primary, labels, noise_score)
    return AudioSceneAnalysis(
        scene_id=scene_id,
        has_audio=True,
        primary_label=primary,
        labels=labels,
        rms_dbfs=rms_dbfs,
        peak_dbfs=peak_dbfs,
        zero_crossing_rate=zcr,
        spectral_centroid_hz=centroid,
        low_frequency_ratio=low_ratio,
        high_frequency_ratio=high_ratio,
        dynamic_range_db=dynamic_range,
        speech_likelihood=speech_likelihood,
        noise_score=noise_score,
        ambience_score=ambience_score,
        candidate_windows=_audio_candidate_windows(windows, duration_seconds),
    )


def _decode_scene_audio(
    scene: Scene,
    asset: MediaAsset,
    ffmpeg_binary: str,
    timeout_seconds: float,
) -> NDArray[np.float64]:
    duration = max(0.05, scene.end_seconds - scene.start_seconds)
    command = [
        ffmpeg_binary,
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{scene.start_seconds:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(asset.path),
        "-vn",
        "-f",
        "s16le",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "pipe:1",
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as error:
        raise DependencyUnavailableError(
            f"FFmpeg executable was not found: {ffmpeg_binary}"
        ) from error
    except subprocess.TimeoutExpired as error:
        raise PipelineStageError(
            f"FFmpeg timed out after {timeout_seconds:g}s while decoding audio "
            f"from {asset.relative_path}."
        ) from error
    if completed.returncode != 0:
        stderr = (
            completed.stderr.decode("utf-8", errors="replace")
            if isinstance(completed.stderr, bytes)
            else str(completed.stderr or "")
        )
        detail = sanitize_process_error(
            stderr,
            private_paths=[asset.path, asset.relative_path],
            fallback="unknown FFmpeg audio decode error",
        )
        raise PipelineStageError(f"FFmpeg could not decode audio for a scene: {detail}")
    if completed.stdout and len(completed.stdout) % np.dtype("<i2").itemsize:
        raise PipelineStageError(
            "FFmpeg returned an invalid PCM payload while decoding scene audio."
        )
    pcm = np.frombuffer(completed.stdout or b"", dtype="<i2").astype(np.float64)
    return pcm / 32768.0


def _scene_with_audio_analysis(scene: Scene, analysis: AudioSceneAnalysis) -> Scene:
    metadata = {
        **scene.metadata,
        "audio_analysis": analysis.model_dump(mode="json"),
        "audio_context": analysis.labels,
        "audio_features": {
            "primary_label": analysis.primary_label,
            "speech_likelihood": analysis.speech_likelihood,
            "noise_score": analysis.noise_score,
            "ambience_score": analysis.ambience_score,
            "rms_dbfs": analysis.rms_dbfs,
            "dynamic_range_db": analysis.dynamic_range_db,
        },
    }
    if analysis.candidate_windows:
        existing = scene.metadata.get("candidate_windows", [])
        existing_windows = existing if isinstance(existing, list) else []
        metadata["candidate_windows"] = [*existing_windows, *analysis.candidate_windows]
    return scene.model_copy(update={"metadata": metadata})


def _window_features(samples: NDArray[np.float64]) -> list[dict[str, float]]:
    window_size = max(1, SAMPLE_RATE // 2)
    features = []
    total = max(samples.size / SAMPLE_RATE, 0.001)
    for start in range(0, samples.size, window_size):
        chunk = samples[start : start + window_size]
        if chunk.size < window_size // 4:
            continue
        rms = float(np.sqrt(np.mean(np.square(chunk))))
        features.append(
            {
                "relative_position": min(1.0, (start + chunk.size / 2) / samples.size),
                "start_seconds": start / SAMPLE_RATE,
                "duration_seconds": chunk.size / SAMPLE_RATE,
                "score": _window_score(chunk, total),
                "rms_dbfs": _dbfs(rms),
            }
        )
    return features


def _window_score(chunk: NDArray[np.float64], total_duration: float) -> float:
    rms = float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0
    zcr = _zero_crossing_rate(chunk)
    energy = max(0.0, min(1.0, (rms + 0.02) / 0.18))
    clarity = max(0.0, 1.0 - min(1.0, zcr / 0.22))
    duration_bonus = 1.0 if total_duration >= 1 else 0.75
    return max(0.0, min(100.0, (energy * 65 + clarity * 35) * duration_bonus))


def _audio_candidate_windows(
    windows: list[dict[str, float]],
    duration_seconds: float,
) -> list[dict[str, Any]]:
    if not windows:
        return []
    sorted_windows = sorted(windows, key=lambda item: item["score"], reverse=True)[:3]
    return [
        {
            "relative_position": item["relative_position"],
            "score": item["score"],
            "source": "audio_analysis",
            "label": "clean audio moment",
            "duration_seconds": min(duration_seconds, item["duration_seconds"]),
        }
        for item in sorted_windows
        if item["score"] >= 35
    ]


def _labels(
    *,
    rms_dbfs: float,
    centroid: float,
    low_ratio: float,
    high_ratio: float,
    dynamic_range: float,
    zcr: float,
    speech_likelihood: float,
    transcript: str | None,
) -> tuple[AudioLabel, list[str]]:
    if rms_dbfs < -48:
        return "silence", ["silence"]
    labels: list[str] = []
    if speech_likelihood >= 0.55:
        labels.append("speech")
    if high_ratio > 0.42 and zcr > 0.12 and dynamic_range < 15:
        labels.append("wind")
    if low_ratio > 0.52 and centroid < 900 and dynamic_range < 14:
        labels.append("transport")
    if 900 <= centroid <= 2600 and 0.18 <= high_ratio <= 0.48 and dynamic_range >= 8:
        labels.append("music")
    if high_ratio > 0.3 and dynamic_range >= 10 and speech_likelihood >= 0.35:
        labels.append("crowd")
    if high_ratio > 0.35 and dynamic_range < 10 and 1100 <= centroid <= 3600:
        labels.append("water")
    if not labels:
        labels.append("ambient")
    if transcript and "speech" not in labels:
        labels.insert(0, "speech")
    primary = _audio_label(labels[0])
    return primary, labels[:6]


def _speech_likelihood(
    *,
    transcript: str | None,
    rms_dbfs: float,
    centroid: float,
    low_ratio: float,
    high_ratio: float,
    dynamic_range: float,
) -> float:
    if transcript and transcript.strip():
        return 0.95
    if rms_dbfs < -42:
        return 0.0
    score = 0.0
    if 300 <= centroid <= 2800:
        score += 0.35
    if 0.12 <= low_ratio <= 0.55:
        score += 0.2
    if 0.08 <= high_ratio <= 0.5:
        score += 0.15
    if dynamic_range >= 8:
        score += 0.25
    return max(0.0, min(1.0, score))


def _noise_score(
    *,
    rms_dbfs: float,
    zcr: float,
    high_ratio: float,
    low_ratio: float,
    dynamic_range: float,
) -> float:
    if rms_dbfs < -48:
        return 0.0
    broadband = max(high_ratio, low_ratio) * 45
    roughness = min(35.0, zcr * 180)
    steadiness = max(0.0, 14 - dynamic_range) * 1.4
    loudness = max(0.0, rms_dbfs + 36) * 1.1
    return max(0.0, min(100.0, broadband + roughness + steadiness + loudness))


def _ambience_score(primary: str, labels: list[str], noise_score: float) -> float:
    if primary == "silence":
        return 5
    score = 48.0
    if {"water", "crowd", "music"} & set(labels):
        score += 22
    if "speech" in labels:
        score += 12
    if {"wind", "transport"} & set(labels):
        score -= 18
    score -= max(0.0, noise_score - 55) * 0.45
    return max(0.0, min(100.0, score))


def _audio_label(value: str) -> AudioLabel:
    allowed = {
        "speech",
        "silence",
        "wind",
        "music",
        "crowd",
        "water",
        "transport",
        "ambient",
        "unknown",
    }
    return cast(AudioLabel, value if value in allowed else "unknown")


def _spectral_centroid(samples: NDArray[np.float64]) -> float:
    if samples.size < 32:
        return 0.0
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(samples.size)))
    frequencies = np.fft.rfftfreq(samples.size, 1 / SAMPLE_RATE)
    total = float(np.sum(spectrum))
    if total <= 0:
        return 0.0
    return float(np.sum(frequencies * spectrum) / total)


def _frequency_ratios(samples: NDArray[np.float64]) -> tuple[float, float]:
    if samples.size < 32:
        return 0.0, 0.0
    spectrum = np.square(np.abs(np.fft.rfft(samples * np.hanning(samples.size))))
    frequencies = np.fft.rfftfreq(samples.size, 1 / SAMPLE_RATE)
    total = float(np.sum(spectrum))
    if total <= 0:
        return 0.0, 0.0
    low = float(np.sum(spectrum[frequencies < 350]) / total)
    high = float(np.sum(spectrum[frequencies > 2500]) / total)
    return low, high


def _zero_crossing_rate(samples: NDArray[np.float64]) -> float:
    if samples.size < 2:
        return 0.0
    return float(np.mean(np.signbit(samples[1:]) != np.signbit(samples[:-1])))


def _dynamic_range(windows: list[dict[str, float]]) -> float:
    if not windows:
        return 0.0
    values = [item["rms_dbfs"] for item in windows]
    return max(values) - min(values)


def _dbfs(value: float) -> float:
    return 20 * math.log10(max(value, 1e-9))


def _has_audio(asset: MediaAsset) -> bool:
    return any(
        stream.get("codec_type") == "audio" for stream in asset.probe_metadata.get("streams", [])
    )
