"""Materialize declarative narration cues as one render-ready PCM track."""

import os
import wave
from pathlib import Path
from uuid import uuid4

from travelmovieai.core.exceptions import MontageError
from travelmovieai.domain.models import NarrationAudioCue


def compose_narration_track(
    cues: list[NarrationAudioCue],
    output_path: Path,
) -> float:
    if not cues:
        raise MontageError("Narration track needs at least one timed cue.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.stem}.{uuid4().hex}.tmp.wav")
    sample_rate = 0
    channels = 0
    sample_width = 0
    cursor_frames = 0
    try:
        with wave.open(str(temporary), "wb") as output:
            for cue in cues:
                current_rate, current_channels, current_width, frames, frame_count = (
                    _read_wave_payload(cue.audio_path)
                )
                current_format = (current_rate, current_channels, current_width)
                if sample_rate == 0:
                    sample_rate, channels, sample_width = current_format
                    output.setframerate(sample_rate)
                    output.setnchannels(channels)
                    output.setsampwidth(sample_width)
                elif current_format != (sample_rate, channels, sample_width):
                    raise MontageError("Piper narration lines use inconsistent WAV audio formats.")
                start_frame = round(cue.cue_start_seconds * sample_rate)
                if start_frame < cursor_frames:
                    raise MontageError("Timed narration cues overlap after synthesis.")
                silence_frames = start_frame - cursor_frames
                if silence_frames:
                    output.writeframesraw(b"\0" * silence_frames * channels * sample_width)
                measured_duration = frame_count / sample_rate
                if abs(measured_duration - cue.duration_seconds) > 0.05:
                    raise MontageError(
                        "Piper narration metadata does not match the generated WAV duration."
                    )
                output.writeframesraw(frames)
                cursor_frames = start_frame + frame_count
            output.writeframes(b"")
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return cursor_frames / sample_rate


def narration_track_matches(path: Path, cues: list[NarrationAudioCue]) -> bool:
    if not cues:
        return False
    try:
        with wave.open(str(path), "rb") as audio:
            frame_count = audio.getnframes()
            sample_rate = audio.getframerate()
            valid_format = (
                audio.getcomptype() == "NONE"
                and audio.getnchannels() > 0
                and audio.getsampwidth() > 0
            )
    except (OSError, EOFError, wave.Error):
        return False
    if not valid_format or frame_count <= 0 or sample_rate <= 0:
        return False
    return abs(frame_count / sample_rate - cues[-1].cue_end_seconds) <= 0.05


def _read_wave_payload(path: Path) -> tuple[int, int, int, bytes, int]:
    try:
        with wave.open(str(path), "rb") as source:
            if source.getcomptype() != "NONE":
                raise MontageError("Piper narration WAV must use uncompressed PCM.")
            frame_count = source.getnframes()
            frames = source.readframes(frame_count)
            if frame_count <= 0 or not frames:
                raise MontageError("Piper produced an empty narration line WAV.")
            return (
                source.getframerate(),
                source.getnchannels(),
                source.getsampwidth(),
                frames,
                frame_count,
            )
    except (OSError, EOFError, wave.Error) as error:
        raise MontageError("Piper produced an invalid narration line WAV.") from error
