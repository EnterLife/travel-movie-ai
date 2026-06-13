"""Explainable scene ranking for semantic montage selection."""

from travelmovieai.domain.models import Scene


def rank_scenes(scenes: list[Scene]) -> list[Scene]:
    """Return scenes ordered by semantic score with a small diversity bonus."""
    tag_frequency: dict[str, int] = {}
    for scene in scenes:
        for tag in _semantic_tags(scene):
            tag_frequency[tag] = tag_frequency.get(tag, 0) + 1

    scored: list[Scene] = []
    for scene in scenes:
        base = scene.importance_score if scene.importance_score is not None else 50.0
        quality = scene.quality_score if scene.quality_score is not None else 60.0
        event_importance = float(scene.metadata.get("event_importance", base))
        landmark_bonus = min(8.0, len(scene.metadata.get("landmarks", [])) * 4.0)
        tags = _semantic_tags(scene)
        rarity = sum(1 / tag_frequency[tag] for tag in tags) / len(tags) if tags else 0.0
        diversity_bonus = min(8.0, rarity * 4)
        score = min(
            100.0,
            base * 0.62
            + quality * 0.16
            + event_importance * 0.08
            + diversity_bonus
            + landmark_bonus,
        )
        scored.append(
            scene.model_copy(
                update={
                    "metadata": {
                        **scene.metadata,
                        "ranking_score": score,
                        "ranking_factors": {
                            "vision_importance": base,
                            "visual_quality": quality,
                            "event_importance": event_importance,
                            "landmark_bonus": landmark_bonus,
                            "diversity_bonus": diversity_bonus,
                        },
                    }
                }
            )
        )
    return sorted(
        scored,
        key=lambda scene: (
            -float(scene.metadata["ranking_score"]),
            scene.start_seconds,
        ),
    )


def _semantic_tags(scene: Scene) -> set[str]:
    values = {
        str(scene.metadata.get("location_type", "")).strip().casefold(),
        str(scene.metadata.get("activity", "")).strip().casefold(),
        str(scene.metadata.get("emotion", "")).strip().casefold(),
    }
    values.update(str(tag).strip().casefold() for tag in scene.metadata.get("tags", []))
    return {value for value in values if value and value != "unknown"}
