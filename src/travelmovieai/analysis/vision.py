"""Vision model orchestration and structured scene understanding."""

import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from travelmovieai.domain.enums import StoryStyle
from travelmovieai.domain.models import (
    Scene,
    SceneUnderstanding,
    VisionAnalysisReport,
    VisionScoreFactors,
)
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
    analyzed_by_index: dict[int, Scene] = {}
    pending: list[tuple[int, Scene, str]] = []
    total = len(scenes)
    cached_count = 0
    for index, scene in enumerate(scenes, start=1):
        if scene.keyframe_path is None:
            continue
        cache_key = _vision_cache_key(scene, provider.model, style)
        if (
            scene.caption
            and scene.importance_score is not None
            and scene.metadata.get("vision_cache_key") == cache_key
        ):
            analyzed_by_index[index] = scene
            cached_count += 1
            if progress:
                progress(index, total, f"AI-кэш сцены {index}/{total}")
            continue
        pending.append((index, scene, cache_key))

    analyze_batch = getattr(provider, "analyze_batch", None)
    batch_size = max(1, int(getattr(provider, "batch_size", 1)))
    for offset in range(0, len(pending), batch_size):
        chunk = pending[offset : offset + batch_size]
        if progress:
            progress(
                offset,
                total,
                f"AI-анализ сцен {offset + 1}-{offset + len(chunk)}/{len(pending)}, "
                f"batch={len(chunk)}",
            )
        image_paths = [
            scene.keyframe_path for _, scene, _ in chunk if scene.keyframe_path is not None
        ]
        if callable(analyze_batch):
            understandings = analyze_batch(image_paths, style)
        else:
            understandings = [provider.analyze(path, style) for path in image_paths]
        for (index, scene, cache_key), understanding in zip(
            chunk,
            understandings,
            strict=True,
        ):
            analyzed_by_index[index] = _scene_with_understanding(
                scene,
                understanding,
                cache_key,
                provider,
            )
        if progress:
            progress(
                min(total, offset + len(chunk)),
                total,
                f"AI готово сцен {offset + len(chunk)}/{len(pending)}, batch={len(chunk)}",
            )

    analyzed = [analyzed_by_index[index] for index in sorted(analyzed_by_index)]
    return VisionAnalysisReport(
        created_at=datetime.now(UTC),
        provider=provider.name,
        model=provider.model,
        prompt_version=PROMPT_VERSION,
        scenes=analyzed,
        analyzed_count=len(pending),
        cached_count=cached_count,
    )


def _scene_with_understanding(
    scene: Scene,
    understanding: SceneUnderstanding,
    cache_key: str,
    provider: VisionProvider,
) -> Scene:
    understanding = _apply_measured_quality(understanding, scene.quality_score)
    metadata = {
        **scene.metadata,
        "vision_cache_key": cache_key,
        "detailed_description": understanding.detailed_description,
        "location_type": understanding.location_type.value,
        "activity": understanding.activity.value,
        "emotion": understanding.emotion.value,
        "people_count": understanding.people_count,
        "people_groups": [group.value for group in understanding.people_groups],
        "landmarks": [
            landmark.model_dump(mode="json") for landmark in understanding.landmarks
        ],
        "tags": understanding.tags,
        "vision_score": understanding.vision_score,
        "vision_score_factors": understanding.score_factors.model_dump(),
        "story_relevance": understanding.story_relevance,
        "vision_provider": provider.name,
        "vision_model": provider.model,
        "prompt_version": PROMPT_VERSION,
    }
    return scene.model_copy(
        update={
            "caption": understanding.caption,
            "importance_score": understanding.vision_score,
            "metadata": metadata,
        }
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


def _apply_measured_quality(
    understanding: SceneUnderstanding,
    quality_score: float | None,
) -> SceneUnderstanding:
    measured_quality = quality_score if quality_score is not None else 50.0
    factors = understanding.score_factors.model_copy(
        update={"visual_quality": measured_quality}
    )
    score = _weighted_vision_score(factors)
    return understanding.model_copy(
        update={
            "score_factors": factors,
            "vision_score": score,
        }
    )


def _weighted_vision_score(factors: VisionScoreFactors) -> float:
    return min(
        100.0,
        factors.uniqueness * 0.24
        + factors.people * 0.12
        + factors.emotion * 0.2
        + factors.visual_quality * 0.16
        + factors.landmark * 0.18
        + factors.unusual_event * 0.1,
    )
