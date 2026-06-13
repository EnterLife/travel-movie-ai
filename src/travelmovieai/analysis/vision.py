"""Vision model orchestration and structured scene understanding."""

import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from travelmovieai.domain.enums import StoryStyle
from travelmovieai.domain.models import Scene, SceneUnderstanding, VisionAnalysisReport
from travelmovieai.infrastructure.vision import PROMPT_VERSION

VisionProgress = Callable[[int, int, str], None]


class VisionProvider(Protocol):
    name: str
    model: str

    def analyze(self, image_path: Path, style: StoryStyle) -> SceneUnderstanding: ...


def analyze_scenes(
    scenes: list[Scene],
    provider: VisionProvider,
    style: StoryStyle,
    progress: VisionProgress | None = None,
) -> VisionAnalysisReport:
    analyzed: list[Scene] = []
    total = len(scenes)
    for index, scene in enumerate(scenes, start=1):
        if scene.keyframe_path is None:
            continue
        cache_key = _vision_cache_key(scene, provider.model, style)
        if (
            scene.caption
            and scene.importance_score is not None
            and scene.metadata.get("vision_cache_key") == cache_key
        ):
            analyzed.append(scene)
            if progress:
                progress(index, total, f"AI-кэш сцены {index}/{total}")
            continue
        if progress:
            progress(index - 1, total, f"AI-анализ сцены {index}/{total}")
        understanding = provider.analyze(scene.keyframe_path, style)
        metadata = {
            **scene.metadata,
            "vision_cache_key": cache_key,
            "location_type": understanding.location_type,
            "activity": understanding.activity,
            "emotion": understanding.emotion,
            "people_count": understanding.people_count,
            "tags": understanding.tags,
            "vision_provider": provider.name,
            "vision_model": provider.model,
            "prompt_version": PROMPT_VERSION,
        }
        analyzed.append(
            scene.model_copy(
                update={
                    "caption": understanding.caption,
                    "importance_score": understanding.importance_score,
                    "metadata": metadata,
                }
            )
        )
    return VisionAnalysisReport(
        created_at=datetime.now(UTC),
        provider=provider.name,
        model=provider.model,
        prompt_version=PROMPT_VERSION,
        scenes=analyzed,
    )


def _vision_cache_key(scene: Scene, model: str, style: StoryStyle) -> str:
    payload = {
        "scene_cache_key": scene.metadata.get("cache_key"),
        "start": scene.start_seconds,
        "end": scene.end_seconds,
        "model": model,
        "style": style.value,
        "prompt": PROMPT_VERSION,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
