"""Small deterministic editorial checks for generated captions and titles."""

import re

_CAPTION_PREFIXES = (
    re.compile(
        r"^(?:a|the)\s+(?:series|sequence|collection)\s+of\s+"
        r"(?:frames|images|photos|shots)(?:\s+(?:showing|shows?|depicting|depicts?))?"
        r"\s*[:;,\-]?\s*",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:these|the)\s+(?:frames|images|photos|shots)\s+"
        r"(?:showing|shows?|depicting|depicts?)\s*[:;,\-]?\s*",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:this|the)\s+(?:image|frame|photo|shot|scene)\s+"
        r"(?:showing|shows?|depicting|depicts?)\s*[:;,\-]?\s*",
        re.IGNORECASE,
    ),
)
_GENERIC_CAPTIONS = {
    "a beautiful view",
    "a scenic view",
    "a series of frames",
    "a series of images",
    "scenic view",
    "travel scene",
    "unknown scene",
}
_GENERIC_TITLES = {
    "building",
    "city",
    "large letter v",
    "landmark",
    "letter v",
    "location",
    "logo",
    "place",
    "road",
    "scenery",
    "sign",
    "street",
    "town",
    "unknown",
    "village",
}
_SINGLE_LETTER_TITLE = re.compile(
    r"^(?:large\s+)?letter\s+['\"]?[a-z0-9]['\"]?$",
    re.IGNORECASE,
)


def clean_caption(value: object, *, max_characters: int = 240) -> str | None:
    """Return concise semantic caption text or ``None`` for boilerplate."""

    if not isinstance(value, str):
        return None
    text = " ".join(value.split()).strip(" -:;,.")
    if not text:
        return None
    for pattern in _CAPTION_PREFIXES:
        cleaned = pattern.sub("", text, count=1).strip(" -:;,.")
        if cleaned != text:
            text = cleaned
            break
    if not text or _normalized(text) in _GENERIC_CAPTIONS:
        return None
    if not any(character.isalpha() for character in text):
        return None
    text = text[:max_characters].rstrip(" -:;,")
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    if text and text[-1] not in ".!?":
        text += "."
    return text or None


def is_generic_caption(value: object) -> bool:
    """Report whether text has no usable editorial caption after cleanup."""

    if not isinstance(value, str):
        return True
    text = " ".join(value.split())
    return any(pattern.match(text) for pattern in _CAPTION_PREFIXES) or clean_caption(text) is None


def clean_title(value: object, *, max_characters: int = 100) -> str | None:
    """Return a safe concise title, rejecting generic model guesses."""

    if not isinstance(value, str):
        return None
    text = " ".join(value.split()).strip(" -:;,.")
    normalized = _normalized(text)
    if (
        not text
        or normalized in _GENERIC_TITLES
        or _SINGLE_LETTER_TITLE.fullmatch(text) is not None
        or not any(character.isalpha() for character in text)
    ):
        return None
    return text[:max_characters].rstrip(" -:;,.") or None


def is_generic_title(value: object) -> bool:
    """Report whether text is unsuitable as an on-screen event title."""

    return clean_title(value) is None


def _normalized(value: str) -> str:
    return " ".join(value.casefold().split()).strip(" -:;,.")
