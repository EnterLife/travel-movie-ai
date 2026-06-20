"""Lazy Faster Whisper provider adapter."""

import importlib
import math
from pathlib import Path
from typing import Any

from travelmovieai.core.exceptions import DependencyUnavailableError, PipelineStageError
from travelmovieai.domain.models import SpeechSegment, SpeechTranscript


class FasterWhisperProvider:
    name = "faster-whisper"

    def __init__(self, model: str, device: str = "auto") -> None:
        self.model = model
        self.device = device
        self._loaded_model: Any = None

    def transcribe(self, audio_path: Path) -> SpeechTranscript:
        model = self._ensure_loaded()
        try:
            segments, info = model.transcribe(
                str(audio_path),
                beam_size=5,
                vad_filter=True,
                condition_on_previous_text=False,
            )
            resolved_segments = list(segments)
        except (OSError, RuntimeError, ValueError) as error:
            raise PipelineStageError(
                f"Whisper не смог распознать звук сцены {audio_path.name}."
            ) from error
        text = " ".join(segment.text.strip() for segment in resolved_segments).strip()
        log_probabilities = [
            float(segment.avg_logprob)
            for segment in resolved_segments
            if getattr(segment, "avg_logprob", None) is not None
        ]
        confidence = (
            min(1.0, max(0.0, math.exp(sum(log_probabilities) / len(log_probabilities))))
            if log_probabilities
            else None
        )
        return SpeechTranscript(
            text=text,
            language=getattr(info, "language", None),
            confidence=confidence,
            segments=[
                SpeechSegment(
                    start_seconds=max(0.0, float(getattr(segment, "start", 0.0))),
                    end_seconds=max(0.0, float(getattr(segment, "end", 0.0))),
                    text=segment.text.strip(),
                    confidence=_segment_confidence(segment),
                )
                for segment in resolved_segments
                if segment.text.strip()
            ],
        )

    def _ensure_loaded(self) -> Any:
        if self._loaded_model is not None:
            return self._loaded_model
        try:
            module = importlib.import_module("faster_whisper")
        except ImportError as error:
            raise DependencyUnavailableError(
                'Для распознавания речи установите python -m pip install -e ".[speech]".'
            ) from error
        device = "cuda" if self.device in {"auto", "cuda"} else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        try:
            self._loaded_model = module.WhisperModel(
                self.model,
                device=device,
                compute_type=compute_type,
            )
        except (OSError, RuntimeError, ValueError) as error:
            if self.device == "auto":
                try:
                    self._loaded_model = module.WhisperModel(
                        self.model,
                        device="cpu",
                        compute_type="int8",
                    )
                    return self._loaded_model
                except (OSError, RuntimeError, ValueError):
                    pass
            raise PipelineStageError(
                f"Не удалось загрузить Faster Whisper модель '{self.model}'."
            ) from error
        return self._loaded_model


def _segment_confidence(segment: object) -> float | None:
    value = getattr(segment, "avg_logprob", None)
    if value is None:
        return None
    return min(1.0, max(0.0, math.exp(float(value))))
