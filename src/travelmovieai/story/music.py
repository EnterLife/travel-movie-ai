"""Local soundtrack planning and deterministic melodic generation."""

import hashlib
import json
import math
import os
import random
import subprocess
import wave
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from travelmovieai.core.exceptions import MontageError, MusicGenerationError
from travelmovieai.domain.enums import MediaType, StoryStyle
from travelmovieai.domain.models import (
    MediaAsset,
    MusicAccent,
    MusicBeat,
    MusicCueSection,
    MusicPlan,
    QuickMontagePlan,
    QuickMontageSettings,
    Scene,
)

type MusicProfile = Literal["calm", "lounge", "cinematic", "warm", "energetic"]
type FloatArray = NDArray[np.float64]
type NeuralGeneratorName = Literal["ace-step", "musicgen"]
type MusicGeneratorName = Literal["procedural", "ace-step", "musicgen"]
ARRANGEMENT_VERSION = "adaptive-lounge-v7-content-revision"
MusicProgress = Callable[[int, int, str], None]


@dataclass(slots=True)
class MusicPlanExecution:
    """Ephemeral execution details that must not affect the persisted music plan."""

    cache_hit: bool = False


class NeuralMusicGenerator(Protocol):
    name: NeuralGeneratorName
    model: str

    def generate(
        self,
        output_path: Path,
        *,
        prompt: str,
        cue_sheet: list[MusicCueSection],
        duration_seconds: float,
        bpm: int,
        seed: int,
        progress: MusicProgress | None = None,
    ) -> None: ...


STYLE_PROFILES: dict[StoryStyle, MusicProfile] = {
    StoryStyle.CINEMATIC: "cinematic",
    StoryStyle.DOCUMENTARY: "lounge",
    StoryStyle.FAMILY: "warm",
    StoryStyle.VLOG: "lounge",
    StoryStyle.ADVENTURE: "energetic",
    StoryStyle.ROMANTIC: "warm",
}
PROFILE_BPM = {
    "calm": 60,
    "lounge": 76,
    "cinematic": 68,
    "warm": 72,
    "energetic": 96,
}
PROFILE_CHORDS = {
    "calm": (
        (130.81, 164.81, 196.00, 246.94),
        (110.00, 130.81, 164.81, 196.00),
        (146.83, 174.61, 220.00, 261.63),
        (98.00, 123.47, 146.83, 196.00),
    ),
    "lounge": (
        (130.81, 164.81, 196.00, 246.94),
        (110.00, 130.81, 164.81, 196.00),
        (146.83, 174.61, 220.00, 261.63),
        (98.00, 123.47, 146.83, 174.61),
    ),
    "cinematic": (
        (73.42, 110.00, 146.83, 174.61),
        (65.41, 98.00, 130.81, 164.81),
        (87.31, 130.81, 164.81, 196.00),
        (61.74, 92.50, 123.47, 146.83),
    ),
    "warm": (
        (98.00, 123.47, 146.83, 196.00),
        (87.31, 110.00, 130.81, 164.81),
        (110.00, 130.81, 164.81, 220.00),
        (82.41, 103.83, 123.47, 164.81),
    ),
    "energetic": (
        (110.00, 138.59, 164.81, 220.00),
        (123.47, 155.56, 185.00, 246.94),
        (98.00, 123.47, 164.81, 196.00),
        (130.81, 164.81, 196.00, 261.63),
    ),
}
STYLE_KEYWORDS: dict[StoryStyle, tuple[str, ...]] = {
    StoryStyle.CINEMATIC: ("cinematic", "epic", "score"),
    StoryStyle.DOCUMENTARY: ("documentary", "ambient", "calm"),
    StoryStyle.FAMILY: ("family", "happy", "warm"),
    StoryStyle.VLOG: ("vlog", "travel", "upbeat"),
    StoryStyle.ADVENTURE: ("adventure", "energy", "action"),
    StoryStyle.ROMANTIC: ("romantic", "love", "emotional"),
}
MUSIC_KEYWORDS = ("music", "theme", "soundtrack", "score", "song")


def build_music_plan(
    assets: list[MediaAsset],
    scenes: list[Scene],
    settings: QuickMontageSettings,
    bundled_music_dir: Path,
    generated_path: Path,
    montage_plan: QuickMontagePlan,
    neural_generator: NeuralMusicGenerator | None = None,
    progress: MusicProgress | None = None,
    *,
    ffmpeg_binary: str = "ffmpeg",
    execution: MusicPlanExecution | None = None,
) -> MusicPlan:
    if execution is not None:
        execution.cache_hit = False
    duration_seconds = montage_plan.total_duration_seconds
    accents = (
        build_music_accents(montage_plan)
        if settings.music_sync
        else _edge_accents(duration_seconds)
    )
    mode = "none" if not settings.music_enabled else settings.music_mode
    if mode == "none":
        return MusicPlan(mode="none", reasoning="Music was disabled by the user.")
    profile, reasoning = choose_music_profile(scenes, settings)
    bpm = PROFILE_BPM[profile]
    cue_sections = build_music_cue_sections(montage_plan, accents, bpm)
    beat_grid = build_music_beat_grid(duration_seconds, bpm, accents)
    if mode == "manual":
        path = _manual_music(settings)
        bpm, cue_sections, beat_grid = _track_rhythm(
            path,
            fallback_bpm=bpm,
            montage_plan=montage_plan,
            accents=accents,
            settings=settings,
            ffmpeg_binary=ffmpeg_binary,
        )
        return MusicPlan(
            mode="manual",
            source_path=path,
            profile=profile,
            bpm=bpm,
            duration_seconds=duration_seconds,
            accents=accents,
            cue_sections=cue_sections,
            beat_grid=beat_grid,
            reasoning="Used the music file selected by the user.",
        )

    if mode == "library":
        library_path = _select_library_track(assets, settings, bundled_music_dir)
        if library_path is None:
            raise MontageError("No suitable music file was found in the local library.")
        bpm, cue_sections, beat_grid = _track_rhythm(
            library_path,
            fallback_bpm=bpm,
            montage_plan=montage_plan,
            accents=accents,
            settings=settings,
            ffmpeg_binary=ffmpeg_binary,
        )
        return MusicPlan(
            mode="library",
            source_path=library_path,
            profile=profile,
            bpm=bpm,
            duration_seconds=duration_seconds,
            accents=accents,
            cue_sections=cue_sections,
            beat_grid=beat_grid,
            reasoning=reasoning + " Selected a track from the local library.",
        )

    if mode == "auto" and settings.music_path is not None:
        path = _manual_music(settings)
        bpm, cue_sections, beat_grid = _track_rhythm(
            path,
            fallback_bpm=bpm,
            montage_plan=montage_plan,
            accents=accents,
            settings=settings,
            ffmpeg_binary=ffmpeg_binary,
        )
        return MusicPlan(
            mode="manual",
            source_path=path,
            profile=profile,
            bpm=bpm,
            duration_seconds=duration_seconds,
            accents=accents,
            cue_sections=cue_sections,
            beat_grid=beat_grid,
            reasoning=reasoning + " Used the explicitly selected file.",
        )

    target_generator = (
        neural_generator.name
        if settings.music_engine in {"auto", "ace-step"} and neural_generator is not None
        else "procedural"
    )
    target_model = neural_generator.model if neural_generator is not None else None
    cache_key = _music_cache_key(
        montage_plan,
        profile=profile,
        bpm=bpm,
        accents=accents,
        cue_sections=cue_sections,
        generator=target_generator,
        model=target_model,
    )
    cached = _cached_music_plan(
        generated_path,
        cache_key=cache_key,
        expected_generator=target_generator,
    )
    if cached is not None:
        if execution is not None:
            execution.cache_hit = True
        if progress:
            progress(1, 1, "Music AI: reused a cached composition")
        return cached

    generator_name: MusicGeneratorName = "procedural"
    model_name = None
    fallback_used = False
    generation_reason = ""
    if settings.music_engine in {"auto", "ace-step"}:
        if neural_generator is None:
            if settings.music_engine == "ace-step":
                raise MusicGenerationError(
                    "ACE-Step is unavailable. Run scripts\\setup_windows.bat."
                )
            fallback_used = True
            generation_reason = " ACE-Step is unavailable, so procedural fallback was used."
        else:
            try:
                neural_generator.generate(
                    generated_path,
                    prompt=_music_generation_prompt(profile, bpm, accents, cue_sections),
                    cue_sheet=cue_sections,
                    duration_seconds=duration_seconds,
                    bpm=bpm,
                    seed=_music_seed(montage_plan, profile),
                    progress=progress,
                )
                apply_music_accents(
                    generated_path,
                    duration_seconds=duration_seconds,
                    accents=accents,
                )
                generator_name = neural_generator.name
                model_name = neural_generator.model
                generation_reason = " The composition was created by a dedicated local model."
            except MusicGenerationError as error:
                if settings.music_engine == "ace-step":
                    raise
                fallback_used = True
                generation_reason = (
                    " ACE-Step did not finish generation; procedural "
                    f"fallback ({_short_error(error)})."
                )

    if generator_name == "procedural":
        if progress:
            progress(0, 1, "Procedural adaptive music synthesis")
        generate_ambient_soundtrack(
            generated_path,
            duration_seconds=duration_seconds,
            profile=profile,
            bpm=bpm,
            accents=accents,
            cue_sections=cue_sections,
        )
        if progress:
            progress(1, 1, "Adaptive music created")

    source_content_sha256 = music_source_content_sha256(generated_path)
    if source_content_sha256 is None:
        raise MusicGenerationError("Could not fingerprint the generated soundtrack.")
    return MusicPlan(
        mode="generated",
        source_path=generated_path,
        profile=profile,
        bpm=bpm,
        duration_seconds=duration_seconds,
        accents=accents,
        cue_sections=cue_sections,
        beat_grid=beat_grid,
        arrangement_version=ARRANGEMENT_VERSION,
        generator=generator_name,
        model=model_name,
        fallback_used=fallback_used,
        source_content_sha256=source_content_sha256,
        cache_key=cache_key,
        reasoning=(
            reasoning + f" Created one {duration_seconds:.1f}s composition with "
            f"{len(accents)} synchronized music accent(s)." + generation_reason
        ),
        generated=True,
    )


def _track_rhythm(
    path: Path,
    *,
    fallback_bpm: int,
    montage_plan: QuickMontagePlan,
    accents: list[MusicAccent],
    settings: QuickMontageSettings,
    ffmpeg_binary: str,
) -> tuple[int, list[MusicCueSection], list[MusicBeat]]:
    bpm = fallback_bpm
    if settings.music_bpm_analysis:
        bpm = _analyze_music_bpm(path, ffmpeg_binary=ffmpeg_binary) or fallback_bpm
    return (
        bpm,
        build_music_cue_sections(montage_plan, accents, bpm),
        build_music_beat_grid(montage_plan.total_duration_seconds, bpm, accents),
    )


def _analyze_music_bpm(path: Path, *, ffmpeg_binary: str = "ffmpeg") -> int | None:
    """Estimate soundtrack tempo from a bounded mono PCM decode, with safe fallback."""

    sample_rate = 11_025
    command = [
        ffmpeg_binary,
        "-hide_banner",
        "-loglevel",
        "error",
        "-t",
        "90",
        "-i",
        str(path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "pipe:1",
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            check=False,
            timeout=60,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0 or len(completed.stdout) < sample_rate * 4:
        return None
    samples = np.frombuffer(completed.stdout, dtype="<f4").astype(np.float64, copy=False)
    return _estimate_bpm(samples, sample_rate=sample_rate)


def _estimate_bpm(samples: NDArray[np.float64], *, sample_rate: int) -> int | None:
    if sample_rate <= 0 or samples.size < sample_rate:
        return None
    frame_size = 1024
    hop_size = 256
    frame_count = 1 + (samples.size - frame_size) // hop_size
    if frame_count < 12:
        return None
    energy = np.empty(frame_count, dtype=np.float64)
    for index in range(frame_count):
        start = index * hop_size
        frame = samples[start : start + frame_size]
        energy[index] = math.sqrt(float(np.mean(frame * frame)))
    onset = np.maximum(0.0, np.diff(energy, prepend=energy[0]))
    threshold = float(np.mean(onset) + np.std(onset) * 0.75)
    onset = np.where(onset >= threshold, onset, 0.0)
    if np.count_nonzero(onset) < 8 or float(np.max(onset)) <= 1e-8:
        return None
    onset -= float(np.mean(onset))
    envelope_rate = sample_rate / hop_size
    minimum_lag = max(1, int(envelope_rate * 60 / 180))
    maximum_lag = min(len(onset) - 1, int(envelope_rate * 60 / 45))
    if maximum_lag <= minimum_lag:
        return None
    scores = np.array(
        [float(np.dot(onset[lag:], onset[:-lag])) for lag in range(minimum_lag, maximum_lag + 1)],
        dtype=np.float64,
    )
    best_score = float(np.max(scores))
    if not math.isfinite(best_score) or best_score <= 0:
        return None
    best_lag = minimum_lag + int(np.argmax(scores))
    half_lag = int(round(best_lag / 2))
    if best_lag >= 2 * minimum_lag and half_lag >= minimum_lag:
        half_score = scores[half_lag - minimum_lag]
        if half_score >= best_score * 0.45:
            best_lag = half_lag
    bpm = int(round(60 * envelope_rate / best_lag))
    return min(180, max(45, bpm))


def build_music_cue_sections(
    plan: QuickMontagePlan,
    accents: list[MusicAccent],
    bpm: int,
) -> list[MusicCueSection]:
    """Build arrangement sections from the final movie timeline."""

    duration = plan.total_duration_seconds
    if duration <= 0:
        return []
    if not plan.clips:
        return [
            MusicCueSection(
                role="intro",
                start_seconds=0,
                end_seconds=duration,
                bpm=bpm,
                intensity=0.35,
                accent_count=len(accents),
                description="Single quiet bed for a short movie.",
            )
        ]

    transition = _effective_transition(plan)
    starts = _clip_starts(plan)
    highlight_times = {
        round(accent.time_seconds, 2) for accent in accents if accent.kind == "highlight"
    }
    boundaries = {0.0, duration}
    for start in starts[1:]:
        if 1.0 < start < duration - 1.0:
            boundaries.add(round(start, 3))
    for time_seconds in highlight_times:
        if 1.0 < time_seconds < duration - 1.0:
            boundaries.add(round(max(0.0, time_seconds - transition), 3))

    ordered = sorted(boundaries)
    sections: list[MusicCueSection] = []
    for index, (start, end) in enumerate(zip(ordered, ordered[1:], strict=False)):
        if end - start < 0.35:
            continue
        role = _cue_section_role(index, start, end, duration, accents)
        section_accents = [accent for accent in accents if start <= accent.time_seconds < end]
        intensity = _cue_section_intensity(role, section_accents)
        sections.append(
            MusicCueSection(
                role=role,
                start_seconds=start,
                end_seconds=end,
                bpm=bpm,
                intensity=intensity,
                accent_count=len(section_accents),
                description=_cue_section_description(role, section_accents),
            )
        )

    return sections or [
        MusicCueSection(
            role="journey",
            start_seconds=0,
            end_seconds=duration,
            bpm=bpm,
            intensity=0.45,
            accent_count=len(accents),
            description="Continuous soft travel underscore.",
        )
    ]


def build_music_beat_grid(
    duration_seconds: float,
    bpm: int,
    accents: list[MusicAccent],
) -> list[MusicBeat]:
    """Create a compact beat grid for future beat-aware editing decisions."""

    if duration_seconds <= 0 or bpm <= 0:
        return []
    beat_seconds = 60 / bpm
    beat_count = min(4096, math.ceil(duration_seconds / beat_seconds))
    grid: list[MusicBeat] = []
    for beat_index in range(beat_count):
        time_seconds = min(duration_seconds, beat_index * beat_seconds)
        nearest = _nearest_accent(time_seconds, accents)
        beat_in_bar = beat_index % 4
        base_strength = 0.72 if beat_in_bar == 0 else 0.42 if beat_in_bar == 2 else 0.24
        accent_strength = nearest.strength * 0.45 if nearest is not None else 0.0
        grid.append(
            MusicBeat(
                time_seconds=time_seconds,
                beat_index=beat_index,
                bar_index=beat_index // 4,
                strength=min(1.0, base_strength + accent_strength),
                nearest_accent_kind=nearest.kind if nearest is not None else None,
            )
        )
    for accent in accents:
        if not 0 <= accent.time_seconds <= duration_seconds:
            continue
        if any(abs(beat.time_seconds - accent.time_seconds) <= 0.1 for beat in grid):
            continue
        beat_index = max(0, round(accent.time_seconds / beat_seconds))
        grid.append(
            MusicBeat(
                time_seconds=round(accent.time_seconds, 3),
                beat_index=beat_index,
                bar_index=beat_index // 4,
                strength=min(1.0, max(0.68, accent.strength + 0.52)),
                nearest_accent_kind=accent.kind,
            )
        )
    return sorted(grid, key=lambda beat: (beat.time_seconds, beat.beat_index))


def build_music_accents(plan: QuickMontagePlan) -> list[MusicAccent]:
    """Build a deterministic cue sheet from clip timing and semantic importance."""

    if not plan.clips or plan.total_duration_seconds <= 0:
        return []
    accents = _edge_accents(plan.total_duration_seconds)
    transition = _effective_transition(plan)
    clip_starts = _clip_starts(plan)

    scored = [clip.semantic_score for clip in plan.clips if clip.semantic_score is not None]
    highlight_threshold = max(70.0, _percentile(scored, 0.75)) if scored else 101.0
    previous_event = plan.clips[0].event_id
    for index, (clip, start) in enumerate(zip(plan.clips, clip_starts, strict=True)):
        if index:
            event_changed = clip.event_id is not None and clip.event_id != previous_event
            accents.append(
                MusicAccent(
                    time_seconds=min(plan.total_duration_seconds, start),
                    kind="event_change" if event_changed else "scene_change",
                    strength=0.32 if event_changed else 0.16,
                    scene_id=clip.scene_id,
                    label=("Event change" if event_changed else f"Scene change {index + 1}"),
                )
            )
        if clip.semantic_score is not None and clip.semantic_score >= highlight_threshold:
            end_overlap = (
                transition
                if index < len(plan.clips) - 1 and _clip_uses_transition(plan, index + 1)
                else 0.0
            )
            visible_duration = max(0.2, clip.duration_seconds - end_overlap)
            accent_time = start + min(visible_duration * 0.48, visible_duration - 0.1)
            accents.append(
                MusicAccent(
                    time_seconds=min(plan.total_duration_seconds, accent_time),
                    kind="highlight",
                    strength=min(0.45, 0.32 + clip.semantic_score / 800),
                    scene_id=clip.scene_id,
                    label=clip.caption or f"Important scene {index + 1}",
                )
            )
        previous_event = clip.event_id
    return _merge_nearby_accents(accents)


def choose_music_profile(
    scenes: list[Scene],
    settings: QuickMontageSettings,
) -> tuple[MusicProfile, str]:
    if settings.music_profile != "auto":
        return settings.music_profile, "Music profile selected by the user."

    metrics = [
        scene.metadata.get("quality_metrics", {})
        for scene in scenes
        if scene.metadata.get("quality_metrics")
    ]
    brightness = _average(metrics, "brightness", 50)
    saturation = _average(metrics, "saturation", 45)
    sharpness = _average(metrics, "sharpness", 50)
    profile: MusicProfile = "calm"
    return (
        profile,
        "The default AI profile is very calm, low-key music; brighter and more "
        "energetic profiles are used only when selected explicitly. OpenCV metrics "
        f"(brightness {brightness:.0f}, saturation {saturation:.0f}, "
        f"sharpness {sharpness:.0f}).",
    )


def apply_music_accents(
    audio_path: Path,
    *,
    duration_seconds: float,
    accents: list[MusicAccent],
) -> None:
    """Normalize a model WAV to the timeline and add sample-accurate accents."""

    temporary_path = audio_path.with_name(f".{audio_path.stem}.synced.wav")
    try:
        with wave.open(str(audio_path), "rb") as source:
            if source.getsampwidth() != 2:
                raise MusicGenerationError(
                    "The local music model returned an unsupported WAV file."
                )
            sample_rate = source.getframerate()
            source_channels = source.getnchannels()
            if source.getnframes() <= 0 or source_channels <= 0:
                raise MusicGenerationError("The local music model returned an empty WAV file.")
            target_frames = round(duration_seconds * sample_rate)
            with wave.open(str(temporary_path), "wb") as target:
                target.setnchannels(2)
                target.setsampwidth(2)
                target.setframerate(sample_rate)
                chunk_frames = sample_rate * 4
                written = 0
                while written < target_frames:
                    requested = min(chunk_frames, target_frames - written)
                    chunks: list[NDArray[np.float64]] = []
                    collected = 0
                    while collected < requested:
                        raw = source.readframes(requested - collected)
                        frames_read = len(raw) // (2 * source_channels)
                        if frames_read == 0:
                            source.rewind()
                            continue
                        samples = np.frombuffer(raw, dtype="<i2").reshape(
                            frames_read,
                            source_channels,
                        )
                        stereo = (
                            np.repeat(samples, 2, axis=1)
                            if source_channels == 1
                            else samples[:, :2]
                        ).astype(np.float64)
                        chunks.append(stereo)
                        collected += frames_read
                    stereo = np.vstack(chunks)[:requested]
                    time = np.arange(written, written + requested, dtype=np.float64) / sample_rate
                    accent, energy = _accent_layers(time, accents)
                    stereo *= 1 + energy[:, None]
                    stereo += accent[:, None] * 1400
                    fade = np.minimum.reduce(
                        (
                            np.ones_like(time),
                            time / 1.2,
                            (duration_seconds - time) / 1.5,
                        )
                    )
                    stereo *= np.maximum(0, fade[:, None])
                    target.writeframesraw(np.clip(stereo, -32767, 32767).astype("<i2").tobytes())
                    written += requested
        os.replace(temporary_path, audio_path)
    except MusicGenerationError:
        temporary_path.unlink(missing_ok=True)
        raise
    except (OSError, wave.Error, ValueError) as error:
        temporary_path.unlink(missing_ok=True)
        raise MusicGenerationError(
            "Could not sync the generated music with the timeline."
        ) from error


def _music_generation_prompt(
    profile: MusicProfile,
    bpm: int,
    accents: list[MusicAccent],
    cue_sections: list[MusicCueSection],
) -> str:
    profile_text = {
        "calm": "very calm low-register ambient travel music",
        "lounge": "soft low-register melodic lounge",
        "cinematic": "restrained low-register cinematic travel ambience",
        "warm": "soft low-register warm optimistic lounge",
        "energetic": "light but still smooth low-register travel lounge",
    }[profile]
    highlights = sum(cue.kind == "highlight" for cue in accents)
    events = sum(cue.kind == "event_change" for cue in accents)
    section_plan = _cue_sheet_prompt(cue_sections)
    return (
        f"Instrumental {profile_text}, {bpm} BPM, hi-fi travel film underscore, "
        "one coherent recurring melody in a consistent key, warm electric piano, "
        "clean muted guitar, deep soft round bass, very light brushed percussion, "
        "subtle dark atmospheric pads, sparse arrangement with natural variation, "
        "low notes, mellow midrange melody, no high-pitched sounds, no bright bells, "
        "no plucks, no sharp synths, no piercing leads, no cymbal shimmer, "
        "polished mix, mastered with headroom, no distortion, no clipping, "
        "low dynamic range for dialogue ducking, no dramatic build-ups, no loud hits, "
        "no aggressive percussion, elegant professional production, no vocals, "
        "no lyrics, no speech, no lead singer, "
        f"gradual narrative development, {events} section changes and "
        f"{highlights} very restrained musical highlights, gentle resolved ending. "
        f"Arrangement cue sheet: {section_plan}"
    )[:700]


def _music_seed(plan: QuickMontagePlan, profile: MusicProfile) -> int:
    source = "|".join(
        (
            profile,
            f"{plan.total_duration_seconds:.3f}",
            *(
                f"{clip.scene_id}:{clip.duration_seconds:.3f}:{clip.semantic_score}"
                for clip in plan.clips
            ),
        )
    )
    return int.from_bytes(hashlib.sha256(source.encode("utf-8")).digest()[:4], "big")


def _music_cache_key(
    plan: QuickMontagePlan,
    *,
    profile: MusicProfile,
    bpm: int,
    accents: list[MusicAccent],
    cue_sections: list[MusicCueSection],
    generator: str,
    model: str | None,
) -> str:
    payload = {
        "version": ARRANGEMENT_VERSION,
        "profile": profile,
        "bpm": bpm,
        "duration": round(plan.total_duration_seconds, 3),
        "generator": generator,
        "model": model,
        "clips": [
            {
                "scene_id": str(clip.scene_id) if clip.scene_id else None,
                "duration": round(clip.duration_seconds, 3),
                "score": clip.semantic_score,
                "event_id": str(clip.event_id) if clip.event_id else None,
            }
            for clip in plan.clips
        ],
        "accents": [accent.model_dump(mode="json") for accent in accents],
        "cue_sections": [section.model_dump(mode="json") for section in cue_sections],
    }
    serialized = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _cached_music_plan(
    generated_path: Path,
    *,
    cache_key: str,
    expected_generator: str,
) -> MusicPlan | None:
    plan_path = generated_path.parent / "music_plan.json"
    if not generated_path.is_file() or not plan_path.is_file():
        return None
    try:
        cached = MusicPlan.model_validate_json(plan_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if (
        cached.cache_key != cache_key
        or cached.generator != expected_generator
        or cached.fallback_used
        or cached.source_path is None
        or cached.source_content_sha256 is None
        or music_source_content_sha256(generated_path) != cached.source_content_sha256
    ):
        return None
    return cached.model_copy(
        update={
            "source_path": generated_path,
            "reasoning": cached.reasoning + " Composition reused from cache.",
        }
    )


def music_source_content_sha256(path: Path) -> str | None:
    try:
        if not path.is_file() or path.stat().st_size <= 0:
            return None
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _short_error(error: Exception) -> str:
    return str(error).replace("\r", " ").replace("\n", " ")[:160]


def generate_ambient_soundtrack(
    output_path: Path,
    *,
    duration_seconds: float,
    profile: MusicProfile,
    bpm: int,
    accents: list[MusicAccent] | None = None,
    cue_sections: list[MusicCueSection] | None = None,
    sample_rate: int = 44100,
) -> None:
    """Generate one adaptive composition for the complete movie timeline."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    chords = PROFILE_CHORDS[profile]
    total_frames = int(duration_seconds * sample_rate)
    if total_frames <= 0:
        raise MontageError("Cannot create music for an empty timeline.")
    beat_seconds = 60 / bpm
    bar_seconds = beat_seconds * 4
    step_seconds = beat_seconds / 2
    rng = random.Random(f"travelmovieai:{profile}:{duration_seconds:.3f}")
    melody = _build_melody(chords, duration_seconds, step_seconds, rng)
    cue_sheet = accents or _edge_accents(duration_seconds)
    sections = cue_sections or _default_cue_sections(duration_seconds, bpm)
    chord_table = np.asarray(chords, dtype=np.float64)

    with wave.open(str(output_path), "wb") as soundtrack:
        soundtrack.setnchannels(2)
        soundtrack.setsampwidth(2)
        soundtrack.setframerate(sample_rate)
        chunk_frames = sample_rate * 4
        for first_frame in range(0, total_frames, chunk_frames):
            last_frame = min(total_frames, first_frame + chunk_frames)
            time = np.arange(first_frame, last_frame, dtype=np.float64) / sample_rate
            chord_indices = (time / bar_seconds).astype(np.int64) % len(chords)
            chord_indices[time >= max(0.0, duration_seconds - bar_seconds)] = 0
            frequencies = chord_table[chord_indices]
            bar_position = np.mod(time, bar_seconds)
            chord_fade = _soft_envelope_array(
                bar_position,
                bar_seconds,
                0.45,
                0.7,
            )
            pad = np.zeros_like(time)
            keys = np.zeros_like(time)
            for voice in range(chord_table.shape[1]):
                frequency = frequencies[:, voice]
                detuned = frequency * (1 + (voice - 1.5) * 0.0015)
                pad += _warm_tone(detuned, time, phase=voice * 0.21)
                keys += _electric_piano_tone(frequency * 2, time, phase=voice * 0.17)
            pad = np.tanh(pad / chord_table.shape[1] * 0.82)
            keys = np.tanh(keys / chord_table.shape[1] * 0.95) * chord_fade

            beat_position = np.mod(time, beat_seconds)
            bass_envelope = np.exp(-4.2 * beat_position / beat_seconds)
            bass_frequency = frequencies[:, 0]
            bass = (
                np.sin(2 * np.pi * bass_frequency / 2 * time)
                + 0.18 * np.sin(2 * np.pi * bass_frequency * time)
            ) * bass_envelope

            step_indices = np.minimum(
                len(melody) - 1,
                (time / step_seconds).astype(np.int64),
            )
            step_position = np.mod(time, step_seconds)
            melody_envelope = _soft_envelope_array(
                step_position,
                step_seconds,
                0.06,
                0.2,
            )
            melody_frequency = np.take(melody, step_indices)
            lead = _electric_piano_tone(melody_frequency, time, phase=0.11) * melody_envelope
            guitar = _muted_guitar_array(
                time,
                melody_frequency,
                step_seconds,
                sample_rate,
            )

            kick = _kick_array(time, beat_seconds)
            brush = _brush_array(time, beat_seconds, sample_rate)
            hat = _hat_array(time, beat_seconds, sample_rate)
            accent, energy = _accent_layers(time, cue_sheet)
            section_energy, section_lead = _section_layers(time, sections)
            arc = 0.78 + 0.16 * np.sin(np.pi * np.minimum(1.0, time / max(duration_seconds, 0.001)))
            rhythm_level = 0.32 if profile == "calm" else 0.72 if profile == "energetic" else 0.48
            dynamics = (arc + energy) * section_energy
            width = 0.035 * np.sin(2 * np.pi * time / 7.0)
            left = (
                pad * (0.18 + width) * chord_fade
                + keys * 0.105
                + bass * 0.095
                + lead * 0.07 * section_lead
                + guitar * 0.028 * section_lead
                + kick * 0.05 * rhythm_level
                + brush * 0.008 * rhythm_level
                + hat * 0.0008 * rhythm_level
            ) * dynamics + accent * 0.055
            right = (
                pad * (0.18 - width) * chord_fade
                + keys * 0.095
                + bass * 0.095
                + lead * 0.06 * section_lead
                + guitar * 0.032 * section_lead
                + kick * 0.05 * rhythm_level
                + brush * 0.01 * rhythm_level
                - hat * 0.0006 * rhythm_level
            ) * dynamics + accent * 0.05
            fade = np.minimum.reduce(
                (
                    np.ones_like(time),
                    time / 2.5,
                    (duration_seconds - time) / 3,
                )
            )
            stereo = np.column_stack((left, right)) * np.maximum(0.0, fade[:, None])
            if profile == "calm":
                stereo = _mellow_lowpass(stereo)
            pcm = _master_to_pcm(stereo)
            soundtrack.writeframesraw(pcm.tobytes())


def _build_melody(
    chords: tuple[tuple[float, ...], ...],
    duration_seconds: float,
    step_seconds: float,
    rng: random.Random,
) -> list[float]:
    step_count = max(1, math.ceil(duration_seconds / step_seconds))
    melody = []
    previous = chords[0][-1] * 1.5
    for step in range(step_count):
        chord = chords[(step // 8) % len(chords)]
        candidates = [frequency * 1.5 for frequency in chord[1:]]
        if step % 4 in {1, 3} and rng.random() < 0.55:
            candidates.append(previous)
        next_frequency = min(
            candidates,
            key=lambda frequency: abs(frequency - previous) + rng.random() * 35,
        )
        melody.append(next_frequency)
        previous = next_frequency
    return melody


def _soft_envelope_array(
    position: FloatArray,
    duration: float,
    attack_seconds: float,
    release_seconds: float,
) -> FloatArray:
    attack = np.minimum(1.0, position / max(attack_seconds, 0.001))
    release = np.minimum(
        1.0,
        (duration - position) / max(release_seconds, 0.001),
    )
    return cast(FloatArray, np.maximum(0.0, attack * release))


def _warm_tone(
    frequency: FloatArray,
    time: FloatArray,
    *,
    phase: float,
) -> FloatArray:
    base = 2 * np.pi * frequency * time + phase
    tone = np.sin(base) + 0.28 * np.sin(base * 2 + 0.2) + 0.08 * np.sin(base * 3 + 0.5)
    return cast(FloatArray, np.tanh(tone * 0.82))


def _electric_piano_tone(
    frequency: FloatArray,
    time: FloatArray,
    *,
    phase: float,
) -> FloatArray:
    base = 2 * np.pi * frequency * time + phase
    tone = np.sin(base) + 0.22 * np.sin(base * 2 + 0.6) + 0.1 * np.sin(base * 3 + 1.1)
    return cast(FloatArray, np.tanh(tone * 0.78))


def _muted_guitar_array(
    time: FloatArray,
    frequency: FloatArray,
    step_seconds: float,
    sample_rate: int,
) -> FloatArray:
    position = np.mod(time, step_seconds)
    step = (time / step_seconds).astype(np.int64)
    active = np.isin(step % 8, (1, 4, 6))
    envelope = _soft_envelope_array(position, step_seconds, 0.012, step_seconds * 0.42)
    sample = (time * sample_rate).astype(np.int64)
    pick_noise = np.sin((sample * 5.3987 + 17.13) * 24634.6345) * 0.025
    tone = _warm_tone(frequency * 1.005, time, phase=0.37)
    return cast(FloatArray, (tone + pick_noise) * envelope * active)


def _master_to_pcm(stereo: NDArray[np.float64]) -> NDArray[np.int16]:
    limited = np.tanh(stereo * 1.18)
    peak = float(np.max(np.abs(limited))) if limited.size else 0.0
    if peak > 0.96:
        limited *= 0.96 / peak
    return (limited * 30000).astype("<i2")


def _mellow_lowpass(stereo: NDArray[np.float64]) -> NDArray[np.float64]:
    if stereo.shape[0] < 41:
        return stereo
    kernel = np.hanning(41)
    kernel /= np.sum(kernel)
    left = np.convolve(stereo[:, 0], kernel, mode="same")
    right = np.convolve(stereo[:, 1], kernel, mode="same")
    return np.column_stack((left, right))


def _kick_array(time: FloatArray, beat_seconds: float) -> FloatArray:
    position = np.mod(time, beat_seconds)
    envelope = np.exp(-18 * position / beat_seconds)
    frequency = 52 + 38 * envelope
    return cast(FloatArray, np.sin(2 * np.pi * frequency * position) * envelope)


def _brush_array(
    time: FloatArray,
    beat_seconds: float,
    sample_rate: int,
) -> FloatArray:
    beat = (time / beat_seconds).astype(np.int64) % 4
    active = np.isin(beat, (1, 3))
    position = np.mod(time, beat_seconds)
    envelope = np.exp(-15 * position / beat_seconds)
    sample = (time * sample_rate).astype(np.int64)
    noise = np.sin((sample * 12.9898 + 78.233) * 43758.5453)
    return cast(FloatArray, noise * envelope * active)


def _hat_array(
    time: FloatArray,
    beat_seconds: float,
    sample_rate: int,
) -> FloatArray:
    position = np.mod(time, beat_seconds)
    envelope = np.exp(-18 * position / beat_seconds)
    sample = (time * sample_rate).astype(np.int64)
    noise = np.sin((sample * 4.1414 + 31.7) * 15731.743) * 0.25
    tone = np.sin(2 * np.pi * 440 * position) * 0.75
    return cast(FloatArray, (tone + noise) * envelope)


def _accent_layers(
    time: FloatArray,
    accents: list[MusicAccent],
) -> tuple[FloatArray, FloatArray]:
    accent_layer = np.zeros_like(time)
    energy = np.zeros_like(time)
    frequencies = {
        "intro": 164.81,
        "scene_change": 196.00,
        "event_change": 246.94,
        "highlight": 329.63,
        "finale": 164.81,
    }
    for cue in accents:
        if cue.time_seconds < time[0] - 3 or cue.time_seconds > time[-1] + 3:
            continue
        delta = time - cue.time_seconds
        transient = np.exp(-0.5 * np.square(delta / 0.16))
        accent_layer += (
            np.sin(2 * np.pi * frequencies[cue.kind] * delta + np.pi / 2) * transient * cue.strength
        )
        energy += (
            np.exp(-0.5 * np.square(delta / 1.15))
            * cue.strength
            * (0.07 if cue.kind == "highlight" else 0.035)
        )
    return accent_layer, cast(FloatArray, np.minimum(0.12, energy))


def _edge_accents(duration_seconds: float) -> list[MusicAccent]:
    if duration_seconds <= 0:
        return []
    return [
        MusicAccent(
            time_seconds=0,
            kind="intro",
            strength=0.18,
            label="Film opening",
        ),
        MusicAccent(
            time_seconds=max(0.0, duration_seconds - min(1.2, duration_seconds * 0.2)),
            kind="finale",
            strength=0.38,
            label="Final music accent",
        ),
    ]


def _effective_transition(plan: QuickMontagePlan) -> float:
    if len(plan.clips) < 2 or not any(
        _clip_uses_transition(plan, index) for index in range(1, len(plan.clips))
    ):
        return 0.0
    return min(
        plan.settings.transition_duration_seconds,
        min(clip.duration_seconds for clip in plan.clips) * 0.45,
    )


def _clip_uses_transition(plan: QuickMontagePlan, clip_index: int) -> bool:
    settings = plan.settings
    if settings.transition == "none" or settings.transition_duration_seconds <= 0:
        return False
    if settings.transition == "cinematic":
        return plan.clips[clip_index].transition == "fade"
    return settings.transition in {"fade", "wipeleft", "slideright"}


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return ordered[index]


def _merge_nearby_accents(accents: list[MusicAccent]) -> list[MusicAccent]:
    merged: list[MusicAccent] = []
    priority = {
        "intro": 0,
        "scene_change": 1,
        "event_change": 2,
        "highlight": 3,
        "finale": 4,
    }
    for cue in sorted(accents, key=lambda item: item.time_seconds):
        if merged and cue.time_seconds - merged[-1].time_seconds < 0.3:
            previous = merged[-1]
            winner = cue if priority[cue.kind] >= priority[previous.kind] else previous
            merged[-1] = winner.model_copy(
                update={"strength": max(previous.strength, cue.strength)}
            )
        else:
            merged.append(cue)
    return merged


def _nearest_accent(
    time_seconds: float,
    accents: list[MusicAccent],
) -> MusicAccent | None:
    if not accents:
        return None
    nearest = min(accents, key=lambda accent: abs(accent.time_seconds - time_seconds))
    return nearest if abs(nearest.time_seconds - time_seconds) <= 0.18 else None


def _clip_starts(plan: QuickMontagePlan) -> list[float]:
    transition = _effective_transition(plan)
    starts: list[float] = []
    elapsed = 0.0
    for index, clip in enumerate(plan.clips):
        starts.append(elapsed)
        if index < len(plan.clips) - 1:
            overlap = transition if _clip_uses_transition(plan, index + 1) else 0.0
            elapsed += clip.duration_seconds - overlap
    return starts


def _cue_section_role(
    index: int,
    start: float,
    end: float,
    duration: float,
    accents: list[MusicAccent],
) -> Literal["intro", "journey", "highlight", "finale"]:
    midpoint = (start + end) / 2
    if index == 0 or midpoint <= duration * 0.14:
        return "intro"
    if end >= duration * 0.88:
        return "finale"
    if any(accent.kind == "highlight" and start <= accent.time_seconds < end for accent in accents):
        return "highlight"
    return "journey"


def _cue_section_intensity(
    role: Literal["intro", "journey", "highlight", "finale"],
    accents: list[MusicAccent],
) -> float:
    base = {
        "intro": 0.3,
        "journey": 0.45,
        "highlight": 0.58,
        "finale": 0.5,
    }[role]
    accent_boost = min(0.18, sum(accent.strength for accent in accents) * 0.12)
    return min(0.72, base + accent_boost)


def _cue_section_description(
    role: Literal["intro", "journey", "highlight", "finale"],
    accents: list[MusicAccent],
) -> str:
    descriptions = {
        "intro": "Soft opening with a clear recurring motif.",
        "journey": "Gentle travel groove with subtle theme variation.",
        "highlight": "Slightly richer melody for an important visual moment.",
        "finale": "Warm resolved ending without a loud hit.",
    }
    if accents:
        return f"{descriptions[role]} {len(accents)} restrained cue(s)."
    return descriptions[role]


def _cue_sheet_prompt(sections: list[MusicCueSection]) -> str:
    if not sections:
        return "single soft section"
    parts = [
        (
            f"{section.role} {section.start_seconds:.1f}-{section.end_seconds:.1f}s "
            f"intensity {section.intensity:.2f}"
        )
        for section in sections[:8]
    ]
    if len(sections) > 8:
        parts.append(f"{len(sections) - 8} more soft sections")
    return "; ".join(parts)


def _default_cue_sections(duration_seconds: float, bpm: int) -> list[MusicCueSection]:
    return [
        MusicCueSection(
            role="journey",
            start_seconds=0,
            end_seconds=duration_seconds,
            bpm=bpm,
            intensity=0.45,
            accent_count=0,
            description="Continuous soft travel underscore.",
        )
    ]


def _section_layers(
    time: FloatArray,
    sections: list[MusicCueSection],
) -> tuple[FloatArray, FloatArray]:
    energy = np.ones_like(time)
    lead = np.ones_like(time)
    for section in sections:
        active = (time >= section.start_seconds) & (time < section.end_seconds)
        if not np.any(active):
            continue
        section_length = max(0.001, section.end_seconds - section.start_seconds)
        local = (time[active] - section.start_seconds) / section_length
        fade = np.minimum(1.0, np.minimum(local / 0.18, (1 - local) / 0.18))
        fade = np.maximum(0.0, fade)
        target_energy = 0.82 + section.intensity * 0.42
        target_lead = 0.72 + section.intensity * 0.72
        if section.role == "intro":
            target_lead *= 0.82
        elif section.role == "highlight":
            target_lead *= 1.1
        elif section.role == "finale":
            target_energy *= 0.95
        energy[active] = energy[active] * (1 - fade) + target_energy * fade
        lead[active] = lead[active] * (1 - fade) + target_lead * fade
    return energy, lead


def _manual_music(settings: QuickMontageSettings) -> Path:
    if settings.music_path is None:
        raise MontageError("Choose a music file for manual mode.")
    path = settings.music_path.expanduser().resolve()
    if not path.is_file():
        raise MontageError(f"Music file not found: {path}")
    return path


def _select_library_track(
    assets: list[MediaAsset],
    settings: QuickMontageSettings,
    bundled_music_dir: Path,
) -> Path | None:
    style_keywords = STYLE_KEYWORDS[settings.story_style]
    candidates = [
        asset.path
        for asset in assets
        if asset.media_type is MediaType.AUDIO
        and asset.scan_error is None
        and any(
            keyword in asset.path.stem.casefold() for keyword in (*style_keywords, *MUSIC_KEYWORDS)
        )
    ]
    if bundled_music_dir.is_dir():
        candidates.extend(
            path
            for path in bundled_music_dir.iterdir()
            if path.is_file() and path.suffix.casefold() in {".mp3", ".wav", ".flac", ".m4a"}
        )
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda path: (
            not any(keyword in path.stem.casefold() for keyword in style_keywords),
            path.name.casefold(),
        ),
    ).resolve()


def _average(metrics: list[dict[str, object]], key: str, default: float) -> float:
    values = [float(value) for item in metrics if isinstance((value := item.get(key)), int | float)]
    return sum(values) / len(values) if values else default
