"""Vision model orchestration and structured scene understanding."""

import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from travelmovieai.core.exceptions import DependencyUnavailableError
from travelmovieai.domain.enums import StoryStyle
from travelmovieai.domain.models import (
    Scene,
    SceneUnderstanding,
    VisionAnalysisReport,
    VisionScoreFactors,
)
from travelmovieai.infrastructure.vision import PARSER_VERSION, PROMPT_VERSION

VISION_CACHE_VERSION = "vision-scene-cache-v3-content-identity"
VISION_SCORING_VERSION = "vision-score-v2-measured-quality"
VISION_METADATA_KEYS = frozenset(
    {
        "activity",
        "camera_motion",
        "detailed_description",
        "emotion",
        "focus_point",
        "focus_source",
        "highlight_windows",
        "landmarks",
        "location_type",
        "parser_version",
        "people_count",
        "people_groups",
        "prompt_version",
        "shot_scale",
        "story_relevance",
        "tags",
        "vision_cache_identity",
        "vision_cache_key",
        "vision_error_type",
        "vision_model",
        "vision_provider",
        "vision_score",
        "vision_score_factors",
        "vision_scoring_version",
        "vision_status",
        "vision_style",
    }
)

VisionProgress = Callable[[int, int, str], None]
VisionCheckpoint = Callable[[VisionAnalysisReport], None]


class VisionProvider(Protocol):
    name: str
    model: str

    def analyze(self, image_path: Path, style: StoryStyle) -> SceneUnderstanding: ...


def analyze_scenes(
    scenes: list[Scene],
    provider: VisionProvider,
    style: StoryStyle,
    progress: VisionProgress | None = None,
    *,
    cached_report: VisionAnalysisReport | None = None,
    checkpoint: VisionCheckpoint | None = None,
    max_scene_retries: int = 0,
    allow_degraded_fallback: bool = False,
    allow_content_identity_migration: bool = False,
) -> VisionAnalysisReport:
    analyzed_by_index: dict[int, Scene] = {}
    pending: list[tuple[int, Scene, str]] = []
    total = len(scenes)
    cached_count = 0
    analyzed_count = 0
    degraded_count = 0
    retry_count = 0
    cached_by_id = (
        {scene.id: scene for scene in cached_report.scenes} if cached_report is not None else {}
    )
    for index, scene in enumerate(scenes, start=1):
        if scene.keyframe_path is None:
            analyzed_by_index[index] = scene
            if progress:
                progress(index, total, f"AI skipped: scene {index}/{total} has no frame")
            continue
        cache_key = _vision_cache_key(scene, provider, style)
        cached_scene = cached_by_id.get(scene.id, scene)
        reusable = bool(
            cached_scene.caption
            and cached_scene.importance_score is not None
            and cached_scene.metadata.get("vision_status") != "degraded"
            and (
                cached_scene.metadata.get("vision_cache_key") == cache_key
                or (
                    allow_content_identity_migration
                    and _cached_scene_content_equivalent(
                        scene,
                        cached_scene,
                        provider,
                        style,
                    )
                )
            )
        )
        if reusable:
            analyzed_by_index[index] = _merge_cached_vision_state(
                scene,
                cached_scene,
                cache_key,
                provider,
                style,
            )
            cached_count += 1
            if progress:
                progress(index, total, f"AI cache: scene {index}/{total}")
            continue
        pending.append((index, scene, cache_key))

    if pending:
        prepare = getattr(provider, "prepare", None)
        if callable(prepare):
            prepare()
    batch_size = max(1, int(getattr(provider, "batch_size", 1)))
    processed_count = cached_count + sum(
        1 for scene in analyzed_by_index.values() if scene.keyframe_path is None
    )
    for offset in range(0, len(pending), batch_size):
        chunk = pending[offset : offset + batch_size]
        if progress:
            progress(
                offset,
                total,
                f"AI analysis: scenes {offset + 1}-{offset + len(chunk)}/{len(pending)}, "
                f"batch={len(chunk)}",
            )
        outcomes = _analyze_chunk_with_isolation(
            chunk,
            provider,
            style,
            max_scene_retries=max(0, max_scene_retries),
            allow_degraded_fallback=allow_degraded_fallback,
        )
        for (
            (index, scene, cache_key),
            understanding,
            degraded,
            error_name,
            outcome_retries,
        ) in outcomes:
            analyzed_by_index[index] = _scene_with_understanding(
                scene,
                understanding,
                cache_key,
                provider,
                style,
                degraded=degraded,
                error_name=error_name,
            )
            if degraded:
                degraded_count += 1
            else:
                analyzed_count += 1
            retry_count += outcome_retries
            processed_count += 1
        if checkpoint:
            checkpoint(
                _vision_report(
                    analyzed_by_index,
                    provider,
                    analyzed_count=analyzed_count,
                    cached_count=cached_count,
                    degraded_count=degraded_count,
                    retry_count=retry_count,
                )
            )
        if progress:
            progress(
                min(total, processed_count),
                total,
                f"AI complete: scenes {offset + len(chunk)}/{len(pending)}, batch={len(chunk)}",
            )

    return _vision_report(
        analyzed_by_index,
        provider,
        analyzed_count=analyzed_count,
        cached_count=cached_count,
        degraded_count=degraded_count,
        retry_count=retry_count,
    )


type _PendingVisionScene = tuple[int, Scene, str]
type _VisionOutcome = tuple[_PendingVisionScene, SceneUnderstanding, bool, str | None, int]


def _analyze_chunk_with_isolation(
    chunk: list[_PendingVisionScene],
    provider: VisionProvider,
    style: StoryStyle,
    *,
    max_scene_retries: int,
    allow_degraded_fallback: bool,
    attempts_so_far: int = 0,
) -> list[_VisionOutcome]:
    if len(chunk) > 1:
        try:
            understandings = _call_provider(provider, chunk, style)
        except DependencyUnavailableError:
            raise
        except Exception:
            midpoint = len(chunk) // 2
            return [
                *_analyze_chunk_with_isolation(
                    chunk[:midpoint],
                    provider,
                    style,
                    max_scene_retries=max_scene_retries,
                    allow_degraded_fallback=allow_degraded_fallback,
                    attempts_so_far=attempts_so_far + 1,
                ),
                *_analyze_chunk_with_isolation(
                    chunk[midpoint:],
                    provider,
                    style,
                    max_scene_retries=max_scene_retries,
                    allow_degraded_fallback=allow_degraded_fallback,
                    attempts_so_far=attempts_so_far + 1,
                ),
            ]
        return [
            (pending, understanding, False, None, attempts_so_far)
            for pending, understanding in zip(chunk, understandings, strict=True)
        ]

    pending = chunk[0]
    last_error: Exception | None = None
    for attempt in range(max_scene_retries + 1):
        try:
            understanding = _call_provider(provider, chunk, style)[0]
            return [(pending, understanding, False, None, attempts_so_far + attempt)]
        except DependencyUnavailableError:
            raise
        except Exception as error:
            last_error = error
    if not allow_degraded_fallback or last_error is None:
        if last_error is not None:
            raise last_error
        raise RuntimeError("Vision analysis failed without an error result.")
    return [
        (
            pending,
            _degraded_understanding(pending[1]),
            True,
            type(last_error).__name__,
            attempts_so_far + max_scene_retries,
        )
    ]


def _call_provider(
    provider: VisionProvider,
    chunk: list[_PendingVisionScene],
    style: StoryStyle,
) -> list[SceneUnderstanding]:
    image_paths = [scene.keyframe_path for _, scene, _ in chunk]
    if any(path is None for path in image_paths):
        raise RuntimeError("Vision analysis received a scene without a sampled frame.")
    resolved_paths = [path for path in image_paths if path is not None]
    analyze_batch = getattr(provider, "analyze_batch", None)
    if callable(analyze_batch):
        raw_results = analyze_batch(resolved_paths, style)
    else:
        raw_results = [provider.analyze(path, style) for path in resolved_paths]
    if len(raw_results) != len(chunk):
        raise RuntimeError(
            "Vision provider returned a different number of results than input scenes."
        )
    return [SceneUnderstanding.model_validate(result) for result in raw_results]


def _degraded_understanding(scene: Scene) -> SceneUnderstanding:
    measured_quality = scene.quality_score if scene.quality_score is not None else 50.0
    return SceneUnderstanding(
        caption=scene.caption or "Travel scene pending detailed visual analysis",
        detailed_description=(
            "The local Vision provider could not validate this scene on the current run."
        ),
        vision_score=max(20.0, min(55.0, measured_quality * 0.55)),
        score_factors=VisionScoreFactors(
            uniqueness=30,
            people=20,
            emotion=30,
            visual_quality=measured_quality,
            landmark=0,
            unusual_event=10,
        ),
        story_relevance="Degraded fallback; retry local Vision analysis before final selection.",
        tags=["vision-degraded"],
    )


def _vision_report(
    analyzed_by_index: dict[int, Scene],
    provider: VisionProvider,
    *,
    analyzed_count: int,
    cached_count: int,
    degraded_count: int = 0,
    retry_count: int = 0,
) -> VisionAnalysisReport:
    return VisionAnalysisReport(
        created_at=datetime.now(UTC),
        provider=provider.name,
        model=provider.model,
        prompt_version=PROMPT_VERSION,
        scenes=[analyzed_by_index[index] for index in sorted(analyzed_by_index)],
        analyzed_count=analyzed_count,
        cached_count=cached_count,
        degraded_count=degraded_count,
        retry_count=retry_count,
    )


def _scene_with_understanding(
    scene: Scene,
    understanding: SceneUnderstanding,
    cache_key: str,
    provider: VisionProvider,
    style: StoryStyle,
    *,
    degraded: bool = False,
    error_name: str | None = None,
) -> Scene:
    understanding = _apply_measured_quality(understanding, scene.quality_score)
    base_metadata = {
        key: value
        for key, value in scene.metadata.items()
        if key
        not in {
            "focus_point",
            "focus_source",
            "highlight_windows",
            "vision_status",
            "vision_error_type",
        }
    }
    metadata = {
        **base_metadata,
        "vision_cache_key": cache_key,
        "detailed_description": understanding.detailed_description,
        "location_type": understanding.location_type.value,
        "activity": understanding.activity.value,
        "emotion": understanding.emotion.value,
        "shot_scale": understanding.shot_scale,
        "camera_motion": understanding.camera_motion,
        "people_count": understanding.people_count,
        "people_groups": [group.value for group in understanding.people_groups],
        "landmarks": [landmark.model_dump(mode="json") for landmark in understanding.landmarks],
        "tags": understanding.tags,
        "vision_score": understanding.vision_score,
        "vision_score_factors": understanding.score_factors.model_dump(),
        "story_relevance": understanding.story_relevance,
        "highlight_windows": [
            window.model_dump(mode="json") for window in understanding.highlight_windows
        ],
        "vision_provider": provider.name,
        "vision_model": provider.model,
        "vision_cache_identity": _provider_cache_identity(provider),
        "prompt_version": PROMPT_VERSION,
        "parser_version": PARSER_VERSION,
        "vision_scoring_version": VISION_SCORING_VERSION,
        "vision_status": "degraded" if degraded else "analyzed",
        "vision_style": style.value,
    }
    if error_name:
        metadata["vision_error_type"] = error_name
    if understanding.focus_x is not None and understanding.focus_y is not None:
        metadata["focus_point"] = {
            "x": understanding.focus_x,
            "y": understanding.focus_y,
        }
        metadata["focus_source"] = understanding.focus_source
    return scene.model_copy(
        update={
            "caption": understanding.caption,
            "importance_score": understanding.vision_score,
            "metadata": metadata,
        }
    )


def _vision_cache_key(scene: Scene, provider: VisionProvider, style: StoryStyle) -> str:
    payload = {
        "cache_version": VISION_CACHE_VERSION,
        "scene_cache_key": scene.metadata.get("cache_key"),
        "start": scene.start_seconds,
        "end": scene.end_seconds,
        "quality_score": scene.quality_score,
        "provider": _provider_cache_identity(provider),
        "style": style.value,
        "prompt": PROMPT_VERSION,
        "parser": PARSER_VERSION,
        "scoring": VISION_SCORING_VERSION,
        "contact_sheet": scene_vision_input_identity(scene),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def scene_vision_input_identity(scene: Scene) -> dict[str, object]:
    contact_sheet = scene.metadata.get("contact_sheet")
    metadata = contact_sheet if isinstance(contact_sheet, dict) else {}
    path = scene.keyframe_path
    file_identity: dict[str, object] = {"exists": False}
    if path is not None:
        try:
            stat = path.stat()
            file_identity = {
                "exists": True,
                "name": path.name,
                "size_bytes": stat.st_size,
                "content_sha256": _file_sha256(path),
            }
        except OSError:
            file_identity = {"exists": False, "name": path.name}
    return {
        "schema_version": metadata.get("schema_version"),
        "sample_count": metadata.get("sample_count"),
        "sample_positions": metadata.get("sample_positions"),
        "file": file_identity,
    }


def _cached_scene_content_equivalent(
    scene: Scene,
    cached_scene: Scene,
    provider: VisionProvider,
    style: StoryStyle,
) -> bool:
    current_sheet = scene.metadata.get("contact_sheet")
    cached_sheet = cached_scene.metadata.get("contact_sheet")
    current_identity = scene_vision_input_identity(scene)
    file_identity = current_identity.get("file")
    expected_sha = current_sheet.get("content_sha256") if isinstance(current_sheet, dict) else None
    cached_style = cached_scene.metadata.get("vision_style")
    return (
        scene.asset_id == cached_scene.asset_id
        and scene.start_seconds == cached_scene.start_seconds
        and scene.end_seconds == cached_scene.end_seconds
        and scene.metadata.get("cache_key") == cached_scene.metadata.get("cache_key")
        and scene.quality_score == cached_scene.quality_score
        and scene.metadata.get("quality_metrics") == cached_scene.metadata.get("quality_metrics")
        and isinstance(current_sheet, dict)
        and current_sheet == cached_sheet
        and isinstance(expected_sha, str)
        and len(expected_sha) == 64
        and isinstance(file_identity, dict)
        and file_identity.get("exists") is True
        and file_identity.get("content_sha256") == expected_sha
        and cached_scene.metadata.get("vision_cache_identity") == _provider_cache_identity(provider)
        and cached_scene.metadata.get("vision_provider") == provider.name
        and cached_scene.metadata.get("vision_model") == provider.model
        and cached_scene.metadata.get("prompt_version") == PROMPT_VERSION
        and cached_scene.metadata.get("parser_version") == PARSER_VERSION
        and cached_scene.metadata.get("vision_scoring_version") == VISION_SCORING_VERSION
        and (cached_style is None or cached_style == style.value)
    )


def _merge_cached_vision_state(
    scene: Scene,
    cached_scene: Scene,
    cache_key: str,
    provider: VisionProvider,
    style: StoryStyle,
) -> Scene:
    metadata = {
        key: value for key, value in scene.metadata.items() if key not in VISION_METADATA_KEYS
    }
    metadata.update(
        {key: value for key, value in cached_scene.metadata.items() if key in VISION_METADATA_KEYS}
    )
    metadata.update(
        {
            "vision_cache_key": cache_key,
            "vision_cache_identity": _provider_cache_identity(provider),
            "vision_style": style.value,
        }
    )
    return scene.model_copy(
        update={
            "caption": cached_scene.caption,
            "importance_score": cached_scene.importance_score,
            "metadata": metadata,
        }
    )


def _provider_cache_identity(provider: VisionProvider) -> dict[str, object]:
    identity_factory = getattr(provider, "cache_identity", None)
    identity = identity_factory() if callable(identity_factory) else None
    if not isinstance(identity, dict):
        identity = {"provider": provider.name, "model": provider.model}
    normalized = json.loads(json.dumps(identity, sort_keys=True, default=str))
    if not isinstance(normalized, dict):
        return {"provider": provider.name, "model": provider.model}
    return {str(key): value for key, value in normalized.items()}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _apply_measured_quality(
    understanding: SceneUnderstanding,
    quality_score: float | None,
) -> SceneUnderstanding:
    measured_quality = quality_score if quality_score is not None else 50.0
    factors = understanding.score_factors.model_copy(update={"visual_quality": measured_quality})
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
