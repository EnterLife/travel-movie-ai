"""Multimodal scene descriptions used by the story builder."""

from datetime import UTC, datetime
from typing import Literal

from travelmovieai.domain.models import (
    MultimodalDescriptionReport,
    MultimodalSceneDescription,
    Scene,
)


def build_multimodal_descriptions(
    scenes: list[Scene],
) -> MultimodalDescriptionReport:
    """Combine available modality outputs without inventing missing context."""
    descriptions: list[MultimodalSceneDescription] = []
    for scene in scenes:
        vision = str(
            scene.metadata.get("detailed_description") or scene.caption or ""
        ).strip()
        if not vision:
            continue
        parts = [vision]
        modalities: list[Literal["vision", "speech", "opencv", "audio"]] = ["vision"]
        transcript = scene.transcript.strip() if scene.transcript else None
        if transcript:
            parts.append(f"Speech: {transcript}")
            modalities.append("speech")
        if scene.quality_score is not None:
            parts.append(f"Visual quality: {scene.quality_score:.0f}/100.")
            modalities.append("opencv")
        audio_context = [
            str(value).strip()
            for value in scene.metadata.get("audio_context", [])
            if str(value).strip()
        ]
        if audio_context:
            parts.append(f"Audio context: {', '.join(audio_context)}.")
            modalities.append("audio")
        descriptions.append(
            MultimodalSceneDescription(
                scene_id=scene.id,
                description=" ".join(parts),
                vision_caption=scene.caption or vision,
                transcript=transcript,
                quality_score=scene.quality_score,
                audio_context=audio_context,
                source_modalities=modalities,
            )
        )
    return MultimodalDescriptionReport(
        created_at=datetime.now(UTC),
        descriptions=descriptions,
    )
