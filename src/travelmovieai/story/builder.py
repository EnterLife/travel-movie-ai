"""Multimodal scene descriptions used by the story builder."""

from datetime import UTC, datetime
from typing import Literal
from uuid import UUID

from travelmovieai.domain.enums import StoryStyle
from travelmovieai.domain.models import (
    Event,
    MultimodalDescriptionReport,
    MultimodalSceneDescription,
    Scene,
    Storyboard,
    StorySection,
)


def build_multimodal_descriptions(
    scenes: list[Scene],
) -> MultimodalDescriptionReport:
    """Combine available modality outputs without inventing missing context."""
    descriptions: list[MultimodalSceneDescription] = []
    for scene in scenes:
        vision = str(scene.metadata.get("detailed_description") or scene.caption or "").strip()
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


def build_storyboard(
    events: list[Event],
    scenes: list[Scene],
    style: StoryStyle,
) -> Storyboard:
    """Build a deterministic story skeleton before timeline decisions."""
    if not events:
        return Storyboard(title="Travel Movie", style=style)
    scene_ids_by_event: dict[str, list[UUID]] = {}
    for scene in scenes:
        event_id = str(scene.metadata.get("event_id", ""))
        scene_ids_by_event.setdefault(event_id, []).append(scene.id)

    sections: list[StorySection] = []
    first = events[0]
    sections.append(
        StorySection(
            role="opening",
            title=first.title,
            event_ids=[first.id],
            scene_ids=scene_ids_by_event.get(str(first.id), []),
        )
    )
    middle = events[1:-1]
    if middle:
        highlight = max(middle, key=lambda event: event.importance_score)
        journey = [event for event in middle if event.id != highlight.id]
        if journey:
            sections.append(
                StorySection(
                    role="journey",
                    title="The Journey",
                    event_ids=[event.id for event in journey],
                    scene_ids=[
                        scene_id
                        for event in journey
                        for scene_id in scene_ids_by_event.get(str(event.id), [])
                    ],
                )
            )
        sections.append(
            StorySection(
                role="highlight",
                title=highlight.title,
                event_ids=[highlight.id],
                scene_ids=scene_ids_by_event.get(str(highlight.id), []),
            )
        )
    if len(events) > 1:
        last = events[-1]
        sections.append(
            StorySection(
                role="finale",
                title=last.title,
                event_ids=[last.id],
                scene_ids=scene_ids_by_event.get(str(last.id), []),
            )
        )
    return Storyboard(
        title=_story_title(events, style),
        style=style,
        event_ids=[event.id for event in events],
        sections=sections,
    )


def _story_title(events: list[Event], style: StoryStyle) -> str:
    landmark = next((event.landmarks[0] for event in events if event.landmarks), None)
    if landmark:
        return f"{landmark}: A {style.value.title()} Journey"
    return f"Our {style.value.title()} Journey"
