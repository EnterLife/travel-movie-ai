"""Shared cleanup rules for state owned by optional pipeline stages."""

from dataclasses import dataclass

from travelmovieai.application.context import ProjectContext
from travelmovieai.infrastructure.database import MediaAssetRepository


@dataclass(frozen=True, slots=True)
class StageOwnedState:
    artifact_names: tuple[str, ...]
    metadata_keys: tuple[str, ...] = ()
    clear_quality_score: bool = False
    clear_transcript: bool = False
    candidate_window_sources: tuple[str, ...] = ()


QUALITY_STATE = StageOwnedState(
    artifact_names=("quality_analysis.json", "quality_analysis.cache.json"),
    metadata_keys=("quality_metrics", "technical_rejection_reasons"),
    clear_quality_score=True,
)
SPEECH_STATE = StageOwnedState(
    artifact_names=("speech_analysis.json", "speech_analysis.cache.json"),
    metadata_keys=(
        "speech_cache_key",
        "speech_provider",
        "speech_model",
        "speech_language",
        "speech_confidence",
        "speech_segments",
    ),
    clear_transcript=True,
)
AUDIO_STATE = StageOwnedState(
    artifact_names=("audio_analysis.json", "audio_analysis.cache.json"),
    metadata_keys=("audio_analysis", "audio_context", "audio_features"),
    candidate_window_sources=("audio_analysis",),
)
DUPLICATE_STATE = StageOwnedState(
    artifact_names=("duplicates.json", "duplicates.cache.json"),
    metadata_keys=(
        "perceptual_hash",
        "duplicate_of",
        "duplicate_similarity",
        "duplicate_status",
    ),
)


def clear_stage_owned_state(context: ProjectContext, ownership: StageOwnedState) -> None:
    """Remove stale artifacts and only the scene fields owned by one stage."""
    context.prepare()
    for name in ownership.artifact_names:
        (context.artifacts_dir / name).unlink(missing_ok=True)

    with MediaAssetRepository(context.database_path) as repository:
        repository.initialize()
        scenes = repository.list_scenes()
        updated = []
        changed = False
        owned_window_sources = set(ownership.candidate_window_sources)
        for scene in scenes:
            metadata = dict(scene.metadata)
            for key in ownership.metadata_keys:
                metadata.pop(key, None)
            if owned_window_sources:
                windows = metadata.get("candidate_windows")
                if isinstance(windows, list):
                    retained = [
                        window
                        for window in windows
                        if not (
                            isinstance(window, dict)
                            and window.get("source") in owned_window_sources
                        )
                    ]
                    if retained:
                        metadata["candidate_windows"] = retained
                    else:
                        metadata.pop("candidate_windows", None)
            replacements: dict[str, object] = {"metadata": metadata}
            if ownership.clear_quality_score:
                replacements["quality_score"] = None
            if ownership.clear_transcript:
                replacements["transcript"] = None
            cleaned = scene.model_copy(update=replacements)
            changed = changed or cleaned != scene
            updated.append(cleaned)
        if changed:
            repository.synchronize_scenes(updated)
