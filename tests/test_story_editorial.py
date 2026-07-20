from travelmovieai.story.editorial import (
    clean_caption,
    clean_title,
    is_generic_caption,
)


def test_clean_caption_removes_model_contact_sheet_boilerplate() -> None:
    raw = "A series of frames showing a coastline at sunset"

    assert clean_caption(raw) == "A coastline at sunset."
    assert is_generic_caption(raw) is True


def test_clean_caption_rejects_empty_visual_description() -> None:
    assert clean_caption("A series of images") is None
    assert clean_caption("12345") is None


def test_clean_title_rejects_generic_visual_guess() -> None:
    assert clean_title("town") is None
    assert clean_title("Large letter 'V'") is None
    assert clean_title("Sochi Olympic Park") == "Sochi Olympic Park"
