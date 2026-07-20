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
from travelmovieai.story.editorial import clean_caption

_EDGE_EVENT_RATIO = 0.15
_HIGHLIGHT_EVENT_RATIO = 0.25


def build_multimodal_descriptions(
    scenes: list[Scene],
) -> MultimodalDescriptionReport:
    """Combine available modality outputs without inventing missing context."""
    descriptions: list[MultimodalSceneDescription] = []
    for scene in scenes:
        vision = clean_caption(scene.metadata.get("detailed_description"), max_characters=500)
        caption = clean_caption(scene.caption, max_characters=500)
        vision = vision or caption
        if vision is None:
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
                vision_caption=caption or vision,
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

    opening, journey, highlights, finale = _story_event_groups(events)
    sections: list[StorySection] = []
    first = opening[0]
    sections.append(
        StorySection(
            role="opening",
            title=first.title,
            event_ids=[event.id for event in opening],
            scene_ids=_section_scene_ids(opening, scene_ids_by_event),
        )
    )
    if journey:
        sections.append(
            StorySection(
                role="journey",
                title="The Journey",
                event_ids=[event.id for event in journey],
                scene_ids=_section_scene_ids(journey, scene_ids_by_event),
            )
        )
    if highlights:
        lead_highlight = max(highlights, key=lambda event: event.importance_score)
        sections.append(
            StorySection(
                role="highlight",
                title=lead_highlight.title,
                event_ids=[event.id for event in highlights],
                scene_ids=_section_scene_ids(highlights, scene_ids_by_event),
            )
        )
    if finale:
        last = finale[-1]
        sections.append(
            StorySection(
                role="finale",
                title=last.title,
                event_ids=[event.id for event in finale],
                scene_ids=_section_scene_ids(finale, scene_ids_by_event),
            )
        )
    return Storyboard(
        title=_story_title(events, style),
        style=style,
        event_ids=[event.id for event in events],
        sections=sections,
    )


def _story_event_groups(
    events: list[Event],
) -> tuple[list[Event], list[Event], list[Event], list[Event]]:
    """Partition a longer trip into a balanced four-part editorial arc."""

    if len(events) == 1:
        return list(events), [], [], []
    if len(events) == 2:
        return [events[0]], [], [], [events[1]]

    edge_count = min(
        max(1, int(len(events) * _EDGE_EVENT_RATIO + 0.5)),
        (len(events) - 1) // 2,
    )
    opening = list(events[:edge_count])
    finale = list(events[-edge_count:])
    middle = list(events[edge_count:-edge_count])
    highlight_count = min(
        len(middle),
        max(1, int(len(events) * _HIGHLIGHT_EVENT_RATIO + 0.5)),
    )
    highlight_ids = {
        event.id
        for _, event in sorted(
            enumerate(middle),
            key=lambda item: (-item[1].importance_score, item[0]),
        )[:highlight_count]
    }
    highlights = [event for event in middle if event.id in highlight_ids]
    journey = [event for event in middle if event.id not in highlight_ids]
    return opening, journey, highlights, finale


def _section_scene_ids(
    events: list[Event],
    scene_ids_by_event: dict[str, list[UUID]],
) -> list[UUID]:
    return [scene_id for event in events for scene_id in scene_ids_by_event.get(str(event.id), [])]


def _story_title(events: list[Event], style: StoryStyle) -> str:
    landmark = next((event.landmarks[0] for event in events if event.landmarks), None)
    if landmark:
        return f"{landmark}: A {style.value.title()} Journey"
    return f"Our {style.value.title()} Journey"
