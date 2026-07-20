"""Deterministic local narration text generation."""

from datetime import UTC, datetime
from typing import Literal

from travelmovieai.domain.models import Event, NarrationLine, NarrationReport, Storyboard

type NarrationRole = Literal["opening", "journey", "highlight", "finale"]


def build_narration(
    storyboard: Storyboard,
    events: list[Event],
    *,
    target_duration_seconds: float = 90.0,
    characters_per_second: float = 14.0,
) -> NarrationReport:
    if characters_per_second <= 0:
        raise ValueError("Narration speech rate must be greater than zero.")
    events_by_id = {event.id: event for event in events}
    drafts: list[tuple[NarrationRole, str]] = []
    for section in storyboard.sections:
        section_events = [
            events_by_id[event_id] for event_id in section.event_ids if event_id in events_by_id
        ]
        if not section_events:
            continue
        summaries = [event.summary.strip() for event in section_events if event.summary.strip()]
        if section.role == "opening":
            text = f"Our journey begins with {section.title}."
        elif section.role == "highlight":
            text = f"The heart of the journey is {section.title}."
        elif section.role == "finale":
            text = f"We finish our journey with {section.title}."
        else:
            titles = ", ".join(event.title for event in section_events[:3])
            text = f"Along the way: {titles}."
        if summaries:
            text = f"{text} {summaries[0]}"
        drafts.append((section.role, text[:1000]))

    lines: list[NarrationLine] = []
    if drafts:
        duration = max(0.1, target_duration_seconds)
        section_duration = duration / len(drafts)
        padding = min(1.0, section_duration * 0.1)
        for index, (role, text) in enumerate(drafts):
            start = index * section_duration + padding
            end = (index + 1) * section_duration - padding
            if end <= start:
                start = index * section_duration
                end = (index + 1) * section_duration
            text = _fit_text_to_speech_budget(
                text,
                max_characters=max(1, int((end - start) * characters_per_second)),
            )
            lines.append(
                NarrationLine(
                    section_role=role,
                    text=text,
                    cue_start_seconds=start,
                    cue_end_seconds=end,
                )
            )
    return NarrationReport(created_at=datetime.now(UTC), lines=lines)


def _fit_text_to_speech_budget(text: str, *, max_characters: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_characters:
        return normalized
    if max_characters <= 3:
        return normalized[:max_characters]
    body_limit = max_characters - 3
    body = normalized[:body_limit].rstrip()
    word_boundary = body.rfind(" ")
    if word_boundary >= max(1, body_limit // 2):
        body = body[:word_boundary].rstrip()
    return f"{body.rstrip(' ,.;:!?')}..."
