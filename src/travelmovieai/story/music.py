"""Local soundtrack planning and deterministic melodic generation."""

import math
import random
import wave
from pathlib import Path
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from travelmovieai.core.exceptions import MontageError
from travelmovieai.domain.enums import MediaType, StoryStyle
from travelmovieai.domain.models import (
    MediaAsset,
    MusicAccent,
    MusicPlan,
    QuickMontagePlan,
    QuickMontageSettings,
    Scene,
)

type MusicProfile = Literal["calm", "lounge", "cinematic", "warm", "energetic"]
type FloatArray = NDArray[np.float64]
ARRANGEMENT_VERSION = "adaptive-lounge-v2"

STYLE_PROFILES: dict[StoryStyle, MusicProfile] = {
    StoryStyle.CINEMATIC: "cinematic",
    StoryStyle.DOCUMENTARY: "lounge",
    StoryStyle.FAMILY: "warm",
    StoryStyle.VLOG: "lounge",
    StoryStyle.ADVENTURE: "energetic",
    StoryStyle.ROMANTIC: "warm",
}
PROFILE_BPM = {
    "calm": 62,
    "lounge": 84,
    "cinematic": 72,
    "warm": 78,
    "energetic": 104,
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
MUSIC_KEYWORDS = ("music", "theme", "soundtrack", "score", "song", "музык", "песн")


def build_music_plan(
    assets: list[MediaAsset],
    scenes: list[Scene],
    settings: QuickMontageSettings,
    bundled_music_dir: Path,
    generated_path: Path,
    montage_plan: QuickMontagePlan,
) -> MusicPlan:
    duration_seconds = montage_plan.total_duration_seconds
    accents = (
        build_music_accents(montage_plan)
        if settings.music_sync
        else _edge_accents(duration_seconds)
    )
    mode = "none" if not settings.music_enabled else settings.music_mode
    if mode == "none":
        return MusicPlan(mode="none", reasoning="Музыка отключена пользователем.")
    if mode == "manual":
        path = _manual_music(settings)
        return MusicPlan(
            mode="manual",
            source_path=path,
            duration_seconds=duration_seconds,
            accents=accents,
            reasoning="Использован выбранный пользователем музыкальный файл.",
        )

    profile, reasoning = choose_music_profile(scenes, settings)
    if mode == "library":
        library_path = _select_library_track(assets, settings, bundled_music_dir)
        if library_path is None:
            raise MontageError("В локальной библиотеке не найден подходящий музыкальный файл.")
        return MusicPlan(
            mode="library",
            source_path=library_path,
            profile=profile,
            duration_seconds=duration_seconds,
            accents=accents,
            reasoning=reasoning + " Выбран трек из локальной библиотеки.",
        )

    if mode == "auto" and settings.music_path is not None:
        path = _manual_music(settings)
        return MusicPlan(
            mode="manual",
            source_path=path,
            profile=profile,
            duration_seconds=duration_seconds,
            accents=accents,
            reasoning=reasoning + " Использован явно указанный файл.",
        )

    generate_ambient_soundtrack(
        generated_path,
        duration_seconds=duration_seconds,
        profile=profile,
        bpm=PROFILE_BPM[profile],
        accents=accents,
    )
    return MusicPlan(
        mode="generated",
        source_path=generated_path,
        profile=profile,
        bpm=PROFILE_BPM[profile],
        duration_seconds=duration_seconds,
        accents=accents,
        arrangement_version=ARRANGEMENT_VERSION,
        reasoning=(
            reasoning + f" Создана единая композиция длиной {duration_seconds:.1f} с "
            f"{len(accents)} синхронизированными музыкальными акцентами."
        ),
        generated=True,
    )


def build_music_accents(plan: QuickMontagePlan) -> list[MusicAccent]:
    """Build a deterministic cue sheet from clip timing and semantic importance."""

    if not plan.clips or plan.total_duration_seconds <= 0:
        return []
    accents = _edge_accents(plan.total_duration_seconds)
    transition = _effective_transition(plan)
    clip_starts: list[float] = []
    elapsed = 0.0
    for index, clip in enumerate(plan.clips):
        clip_starts.append(elapsed)
        if index < len(plan.clips) - 1:
            elapsed += clip.duration_seconds - transition

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
                    strength=0.62 if event_changed else 0.28,
                    scene_id=clip.scene_id,
                    label=("Смена события" if event_changed else f"Смена сцены {index + 1}"),
                )
            )
        if clip.semantic_score is not None and clip.semantic_score >= highlight_threshold:
            visible_duration = max(0.2, clip.duration_seconds - transition)
            accent_time = start + min(visible_duration * 0.48, visible_duration - 0.1)
            accents.append(
                MusicAccent(
                    time_seconds=min(plan.total_duration_seconds, accent_time),
                    kind="highlight",
                    strength=min(1.0, 0.55 + clip.semantic_score / 220),
                    scene_id=clip.scene_id,
                    label=clip.caption or f"Важная сцена {index + 1}",
                )
            )
        previous_event = clip.event_id
    return _merge_nearby_accents(accents)


def choose_music_profile(
    scenes: list[Scene],
    settings: QuickMontageSettings,
) -> tuple[MusicProfile, str]:
    if settings.music_profile != "auto":
        return settings.music_profile, "Музыкальный профиль выбран пользователем."

    metrics = [
        scene.metadata.get("quality_metrics", {})
        for scene in scenes
        if scene.metadata.get("quality_metrics")
    ]
    brightness = _average(metrics, "brightness", 50)
    saturation = _average(metrics, "saturation", 45)
    sharpness = _average(metrics, "sharpness", 50)
    emotions = {str(scene.metadata.get("emotion", "")).casefold() for scene in scenes}
    locations = {str(scene.metadata.get("location_type", "")).casefold() for scene in scenes}
    activities = {str(scene.metadata.get("activity", "")).casefold() for scene in scenes}

    if {"adventurous", "exciting", "energetic"} & emotions or (saturation > 58 and sharpness > 55):
        profile: MusicProfile = "energetic"
    elif {"romantic", "joyful", "emotional"} & emotions or (brightness > 58 and saturation > 48):
        profile = "warm"
    elif brightness < 38 or settings.story_style is StoryStyle.CINEMATIC:
        profile = "cinematic"
    elif (
        {"relaxing"} & emotions
        or {"beach", "sea", "city", "hotel", "restaurant", "park"} & locations
        or {"walking", "dining", "relaxing", "sightseeing"} & activities
    ):
        profile = "lounge"
    else:
        profile = STYLE_PROFILES[settings.story_style]
    return (
        profile,
        "AI-профиль выбран по стилю фильма, эмоциям сцен и OpenCV-метрикам "
        f"(яркость {brightness:.0f}, насыщенность {saturation:.0f}, "
        f"резкость {sharpness:.0f}).",
    )


def generate_ambient_soundtrack(
    output_path: Path,
    *,
    duration_seconds: float,
    profile: MusicProfile,
    bpm: int,
    accents: list[MusicAccent] | None = None,
    sample_rate: int = 44100,
) -> None:
    """Generate one adaptive composition for the complete movie timeline."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    chords = PROFILE_CHORDS[profile]
    total_frames = int(duration_seconds * sample_rate)
    if total_frames <= 0:
        raise MontageError("Невозможно создать музыку для пустого timeline.")
    beat_seconds = 60 / bpm
    bar_seconds = beat_seconds * 4
    step_seconds = beat_seconds / 2
    rng = random.Random(f"travelmovieai:{profile}:{duration_seconds:.3f}")
    melody = _build_melody(chords, duration_seconds, step_seconds, rng)
    cue_sheet = accents or _edge_accents(duration_seconds)
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
            for voice in range(chord_table.shape[1]):
                frequency = frequencies[:, voice]
                pad += np.sin(2 * np.pi * frequency * time + voice * 0.21)
                pad += 0.22 * np.sin(2 * np.pi * frequency * 2 * time)
            pad /= chord_table.shape[1] * 1.22

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
            lead = (
                np.sin(2 * np.pi * melody_frequency * time)
                + 0.16 * np.sin(2 * np.pi * melody_frequency * 2 * time)
            ) * melody_envelope

            kick = _kick_array(time, beat_seconds)
            brush = _brush_array(time, beat_seconds, sample_rate)
            hat = _hat_array(time, beat_seconds, sample_rate)
            accent, energy = _accent_layers(time, cue_sheet)
            arc = 0.78 + 0.16 * np.sin(np.pi * np.minimum(1.0, time / max(duration_seconds, 0.001)))
            rhythm_level = 0.6 if profile == "calm" else 1.15 if profile == "energetic" else 0.85
            dynamics = arc + energy
            left = (
                pad * 0.22 * chord_fade
                + bass * 0.13
                + lead * 0.10
                + kick * 0.09 * rhythm_level
                + brush * 0.035 * rhythm_level
                + hat * 0.018 * rhythm_level
            ) * dynamics + accent * 0.13
            right = (
                pad * 0.22 * chord_fade
                + bass * 0.13
                + lead * 0.08
                + kick * 0.09 * rhythm_level
                + brush * 0.045 * rhythm_level
                - hat * 0.014 * rhythm_level
            ) * dynamics + accent * 0.11
            fade = np.minimum.reduce(
                (
                    np.ones_like(time),
                    time / 2.5,
                    (duration_seconds - time) / 3,
                )
            )
            stereo = np.column_stack((left, right)) * np.maximum(0.0, fade[:, None])
            pcm = (np.tanh(stereo * 1.35) * 32767).astype("<i2")
            soundtrack.writeframesraw(pcm.tobytes())


def _build_melody(
    chords: tuple[tuple[float, ...], ...],
    duration_seconds: float,
    step_seconds: float,
    rng: random.Random,
) -> list[float]:
    step_count = max(1, math.ceil(duration_seconds / step_seconds))
    melody = []
    previous = chords[0][-1] * 2
    for step in range(step_count):
        chord = chords[(step // 8) % len(chords)]
        candidates = [frequency * 2 for frequency in chord[1:]]
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
    position = np.mod(time, beat_seconds / 2)
    envelope = np.exp(-30 * position / beat_seconds)
    sample = (time * sample_rate).astype(np.int64)
    noise = np.sin((sample * 4.1414 + 31.7) * 15731.743)
    return cast(FloatArray, noise * envelope)


def _accent_layers(
    time: FloatArray,
    accents: list[MusicAccent],
) -> tuple[FloatArray, FloatArray]:
    accent_layer = np.zeros_like(time)
    energy = np.zeros_like(time)
    frequencies = {
        "intro": 523.25,
        "scene_change": 659.25,
        "event_change": 783.99,
        "highlight": 1046.50,
        "finale": 523.25,
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
            * (0.18 if cue.kind == "highlight" else 0.1)
        )
    return accent_layer, cast(FloatArray, np.minimum(0.28, energy))


def _edge_accents(duration_seconds: float) -> list[MusicAccent]:
    if duration_seconds <= 0:
        return []
    return [
        MusicAccent(
            time_seconds=0,
            kind="intro",
            strength=0.35,
            label="Начало фильма",
        ),
        MusicAccent(
            time_seconds=max(0.0, duration_seconds - min(1.2, duration_seconds * 0.2)),
            kind="finale",
            strength=0.9,
            label="Финальный музыкальный акцент",
        ),
    ]


def _effective_transition(plan: QuickMontagePlan) -> float:
    if plan.settings.transition == "none" or len(plan.clips) < 2:
        return 0.0
    shortest = min(clip.duration_seconds for clip in plan.clips)
    return min(plan.settings.transition_duration_seconds, shortest * 0.45)


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


def _manual_music(settings: QuickMontageSettings) -> Path:
    if settings.music_path is None:
        raise MontageError("Для ручного режима укажите музыкальный файл.")
    path = settings.music_path.expanduser().resolve()
    if not path.is_file():
        raise MontageError(f"Музыкальный файл не найден: {path}")
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
