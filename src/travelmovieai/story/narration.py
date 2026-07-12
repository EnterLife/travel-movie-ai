"""Deterministic local narration text generation."""

from datetime import UTC, datetime

from travelmovieai.domain.models import Event, NarrationLine, NarrationReport, Storyboard


def build_narration(storyboard: Storyboard, events: list[Event]) -> NarrationReport:
    events_by_id = {event.id: event for event in events}
    lines: list[NarrationLine] = []
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
        lines.append(NarrationLine(section_role=section.role, text=text[:1000]))
    return NarrationReport(created_at=datetime.now(UTC), lines=lines)
