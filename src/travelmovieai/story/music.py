"""Local soundtrack planning and deterministic ambient generation."""

import math
import random
import struct
import wave
from pathlib import Path
from typing import Literal

from travelmovieai.core.exceptions import MontageError
from travelmovieai.domain.enums import MediaType, StoryStyle
from travelmovieai.domain.models import (
    MediaAsset,
    MusicPlan,
    QuickMontageSettings,
    Scene,
)

type MusicProfile = Literal["calm", "cinematic", "warm", "energetic"]

STYLE_PROFILES: dict[StoryStyle, MusicProfile] = {
    StoryStyle.CINEMATIC: "cinematic",
    StoryStyle.DOCUMENTARY: "calm",
    StoryStyle.FAMILY: "warm",
    StoryStyle.VLOG: "energetic",
    StoryStyle.ADVENTURE: "energetic",
    StoryStyle.ROMANTIC: "warm",
}
PROFILE_BPM = {"calm": 62, "cinematic": 72, "warm": 78, "energetic": 104}
PROFILE_CHORDS = {
    "calm": ((220.00, 261.63, 329.63), (196.00, 246.94, 293.66)),
    "cinematic": ((146.83, 220.00, 293.66), (130.81, 196.00, 261.63)),
    "warm": ((196.00, 246.94, 329.63), (174.61, 220.00, 293.66)),
    "energetic": ((220.00, 277.18, 329.63), (246.94, 311.13, 369.99)),
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
    duration_seconds: float,
) -> MusicPlan:
    mode = "none" if not settings.music_enabled else settings.music_mode
    if mode == "none":
        return MusicPlan(mode="none", reasoning="Музыка отключена пользователем.")
    if mode == "manual":
        path = _manual_music(settings)
        return MusicPlan(
            mode="manual",
            source_path=path,
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
            reasoning=reasoning + " Выбран трек из локальной библиотеки.",
        )

    if mode == "auto" and settings.music_path is not None:
        path = _manual_music(settings)
        return MusicPlan(
            mode="manual",
            source_path=path,
            profile=profile,
            reasoning=reasoning + " Использован явно указанный файл.",
        )

    generate_ambient_soundtrack(
        generated_path,
        duration_seconds=min(30.0, max(8.0, duration_seconds)),
        profile=profile,
        bpm=PROFILE_BPM[profile],
    )
    return MusicPlan(
        mode="generated",
        source_path=generated_path,
        profile=profile,
        bpm=PROFILE_BPM[profile],
        reasoning=reasoning + " Создан локальный ненавязчивый soundtrack.",
        generated=True,
    )


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

    if {"adventurous", "exciting", "energetic"} & emotions or (saturation > 58 and sharpness > 55):
        profile: MusicProfile = "energetic"
    elif {"romantic", "joyful", "emotional"} & emotions or (brightness > 58 and saturation > 48):
        profile = "warm"
    elif brightness < 38 or settings.story_style is StoryStyle.CINEMATIC:
        profile = "cinematic"
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
    sample_rate: int = 24000,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    chords = PROFILE_CHORDS[profile]
    total_frames = int(duration_seconds * sample_rate)
    beat_seconds = 60 / bpm
    chord_seconds = beat_seconds * 8
    rng = random.Random(f"travelmovieai:{profile}:{duration_seconds:.3f}")

    with wave.open(str(output_path), "wb") as soundtrack:
        soundtrack.setnchannels(2)
        soundtrack.setsampwidth(2)
        soundtrack.setframerate(sample_rate)
        buffer = bytearray()
        for frame in range(total_frames):
            time = frame / sample_rate
            chord = chords[int(time / chord_seconds) % len(chords)]
            local = time % chord_seconds
            envelope = min(1.0, local / 1.5, (chord_seconds - local) / 1.5)
            envelope = max(0.08, envelope)
            pulse = 0.88 + 0.12 * math.sin(2 * math.pi * time / beat_seconds)
            sample = sum(
                math.sin(2 * math.pi * frequency * time + index * 0.4)
                for index, frequency in enumerate(chord)
            ) / len(chord)
            shimmer = math.sin(2 * math.pi * chord[-1] * 2 * time) * 0.06
            noise = (rng.random() * 2 - 1) * 0.008
            value = (sample * 0.18 + shimmer + noise) * envelope * pulse
            fade = min(1.0, time / 2, (duration_seconds - time) / 2)
            pcm = int(max(-1, min(1, value * max(0, fade))) * 32767)
            buffer.extend(struct.pack("<hh", pcm, pcm))
            if len(buffer) >= 65536:
                soundtrack.writeframesraw(buffer)
                buffer.clear()
        if buffer:
            soundtrack.writeframesraw(buffer)


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
