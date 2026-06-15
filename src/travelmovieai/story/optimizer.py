"""Story-aware timeline metadata and ordering helpers."""

from travelmovieai.domain.models import Scene, Storyboard

ROLE_ORDER = {
    "opening": 0,
    "journey": 1,
    "highlight": 2,
    "finale": 3,
}


def apply_story_structure(
    scenes: list[Scene],
    storyboard: Storyboard,
) -> list[Scene]:
    """Annotate scenes with storyboard section metadata for timeline ordering."""

    section_by_scene: dict[str, tuple[int, str, str]] = {}
    for index, section in enumerate(storyboard.sections):
        for scene_id in section.scene_ids:
            section_by_scene[str(scene_id)] = (index, section.role, section.title)

    updated: list[Scene] = []
    for scene in scenes:
        section_data = section_by_scene.get(str(scene.id))
        if section_data is None:
            updated.append(scene)
            continue
        index, role, title = section_data
        updated.append(
            scene.model_copy(
                update={
                    "metadata": {
                        **scene.metadata,
                        "story_section_index": index,
                        "story_section_role": role,
                        "story_section_title": title,
                        "story_role_order": ROLE_ORDER.get(role, 99),
                    }
                }
            )
        )
    return updated
