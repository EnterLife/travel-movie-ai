"""Deterministic local soundtrack selection."""

from pathlib import Path

from travelmovieai.core.exceptions import MontageError
from travelmovieai.domain.enums import MediaType, StoryStyle
from travelmovieai.domain.models import MediaAsset, QuickMontageSettings

STYLE_KEYWORDS: dict[StoryStyle, tuple[str, ...]] = {
    StoryStyle.CINEMATIC: ("cinematic", "epic", "score"),
    StoryStyle.DOCUMENTARY: ("documentary", "ambient", "calm"),
    StoryStyle.FAMILY: ("family", "happy", "warm"),
    StoryStyle.VLOG: ("vlog", "travel", "upbeat"),
    StoryStyle.ADVENTURE: ("adventure", "energy", "action"),
    StoryStyle.ROMANTIC: ("romantic", "love", "emotional"),
}
MUSIC_KEYWORDS = ("music", "theme", "soundtrack", "score", "song", "музык", "песн")


def select_music(
    assets: list[MediaAsset],
    settings: QuickMontageSettings,
    bundled_music_dir: Path,
) -> Path | None:
    if not settings.music_enabled:
        return None
    if settings.music_path is not None:
        path = settings.music_path.expanduser().resolve()
        if not path.is_file():
            raise MontageError(f"Музыкальный файл не найден: {path}")
        return path

    style_keywords = STYLE_KEYWORDS[settings.story_style]
    source_candidates = [
        asset.path
        for asset in assets
        if asset.media_type is MediaType.AUDIO
        and asset.scan_error is None
        and any(
            keyword in asset.path.stem.casefold() for keyword in (*style_keywords, *MUSIC_KEYWORDS)
        )
    ]
    library_candidates: list[Path] = []
    if bundled_music_dir.is_dir():
        library_candidates.extend(
            path
            for path in bundled_music_dir.iterdir()
            if path.is_file() and path.suffix.casefold() in {".mp3", ".wav", ".flac", ".m4a"}
        )
    candidates = source_candidates + library_candidates
    if not candidates:
        return None

    return min(
        candidates,
        key=lambda path: (
            not any(keyword in path.stem.casefold() for keyword in style_keywords),
            path.name.casefold(),
        ),
    ).resolve()
