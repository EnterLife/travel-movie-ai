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
        people_bonus = _people_bonus(scene)
        audio_bonus, audio_penalty, audio_reasons = _audio_factors(scene)
        technical_reasons = list(scene.metadata.get("technical_rejection_reasons", []))
        technical_penalty = min(24.0, len(technical_reasons) * 8.0)
        duplicate_penalty = 35.0 if scene.metadata.get("duplicate_status") == "duplicate" else 0.0
        tags = _semantic_tags(scene)
        rarity = sum(1 / tag_frequency[tag] for tag in tags) / len(tags) if tags else 0.0
        diversity_bonus = min(8.0, rarity * 4)
        score = max(
            0.0,
            min(
                100.0,
                base * 0.62
                + quality * 0.16
                + event_importance * 0.08
                + diversity_bonus
                + landmark_bonus
                + people_bonus
                + audio_bonus
                - technical_penalty
                - duplicate_penalty
                - audio_penalty,
            ),
        )
        reasons = [
            f"vision {base:.0f}",
            f"quality {quality:.0f}",
            f"event {event_importance:.0f}",
        ]
        if landmark_bonus:
            reasons.append("landmark")
        if people_bonus:
            reasons.append("people")
        reasons.extend(audio_reasons)
        if diversity_bonus:
            reasons.append("semantic diversity")
        if technical_reasons:
            reasons.append(f"technical issues: {', '.join(technical_reasons)}")
        if duplicate_penalty:
            reasons.append("near duplicate")
        scored.append(
            scene.model_copy(
                update={
                    "metadata": {
                        **scene.metadata,
                        "ranking_score": score,
                        "ranking_reasons": reasons,
                        "ranking_factors": {
                            "vision_importance": base,
                            "visual_quality": quality,
                            "event_importance": event_importance,
                            "landmark_bonus": landmark_bonus,
                            "people_bonus": people_bonus,
                            "audio_bonus": audio_bonus,
                            "audio_penalty": audio_penalty,
                            "diversity_bonus": diversity_bonus,
                            "technical_penalty": technical_penalty,
                            "duplicate_penalty": duplicate_penalty,
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


def _people_bonus(scene: Scene) -> float:
    people_count = _float_value(scene.metadata.get("people_count"))
    groups = scene.metadata.get("people_groups", [])
    has_group = isinstance(groups, list) and any(
        str(group).strip().casefold() not in {"", "none"} for group in groups
    )
    if people_count <= 0 and not has_group:
        return 0.0
    return min(8.0, 3.0 + people_count * 1.5 + (1.5 if has_group else 0.0))


def _semantic_tags(scene: Scene) -> set[str]:
    values = {
        str(scene.metadata.get("location_type", "")).strip().casefold(),
        str(scene.metadata.get("activity", "")).strip().casefold(),
        str(scene.metadata.get("emotion", "")).strip().casefold(),
    }
    values.update(str(tag).strip().casefold() for tag in scene.metadata.get("tags", []))
    return {value for value in values if value and value != "unknown"}


def _audio_factors(scene: Scene) -> tuple[float, float, list[str]]:
    features = scene.metadata.get("audio_features", {})
    if not isinstance(features, dict):
        return 0.0, 0.0, []
    label = str(features.get("primary_label", "unknown"))
    speech = _float_value(features.get("speech_likelihood"))
    noise = _float_value(features.get("noise_score"))
    ambience = _float_value(features.get("ambience_score"))
    bonus = min(7.0, speech * 3.0 + ambience / 100 * 4.0)
    penalty = 0.0
    reasons: list[str] = []
    if speech >= 0.55:
        reasons.append("speech protected")
    if label in {"water", "crowd", "music"} or ambience >= 68:
        reasons.append(f"audio ambience {label}")
    if label in {"wind", "transport"}:
        penalty += 6.0
        reasons.append(f"audio noise {label}")
    if noise >= 72:
        penalty += min(8.0, (noise - 72) * 0.28)
        if not any(reason.startswith("audio noise") for reason in reasons):
            reasons.append("audio noise")
    return bonus, min(14.0, penalty), reasons


def _float_value(value: object) -> float:
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0
