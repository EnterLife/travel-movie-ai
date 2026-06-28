"""Pipeline stage that plans soundtrack cues for the montage."""

from pathlib import Path

from pydantic import ValidationError

from travelmovieai.application.context import ProjectContext
from travelmovieai.core.exceptions import MontageError
from travelmovieai.domain.enums import PipelineStage
from travelmovieai.domain.models import MusicPlan, QuickMontageSettings, StageResult
from travelmovieai.editing.timeline import build_semantic_montage_plan
from travelmovieai.infrastructure.artifacts import (
    artifact_fingerprint,
    stage_cache_manifest_matches,
    write_json_atomic,
    write_stage_cache_manifest,
)
from travelmovieai.infrastructure.database import MediaAssetRepository
from travelmovieai.pipeline.base import Stage
from travelmovieai.story.music import build_music_plan

ARTIFACT_SCHEMA_VERSION = "music-selection-v1"


class MusicSelectionStage(Stage):
    name = PipelineStage.MUSIC_SELECTION

    def run(self, context: ProjectContext) -> StageResult:
        repository = MediaAssetRepository(context.database_path)
        repository.initialize()
        assets = repository.list_assets()
        scenes = repository.list_scenes()
        if not assets or not scenes:
            return StageResult(
                stage=self.name,
                skipped=True,
                message="Music selection needs media assets and ranked scenes.",
            )

        settings = _semantic_montage_settings(context)
        if not settings.music_enabled or settings.music_mode == "none":
            music_plan = MusicPlan(
                mode="none",
                duration_seconds=0,
                reasoning="Music disabled by montage settings.",
            )
            music_artifact = context.artifacts_dir / "music_plan.json"
            write_json_atomic(music_artifact, music_plan)
            return StageResult(
                stage=self.name,
                skipped=True,
                artifacts=[music_artifact],
                message="Music selection disabled by montage settings.",
            )

        draft_plan = build_semantic_montage_plan(assets, scenes, settings)
        music_artifact = context.artifacts_dir / "music_plan.json"
        cache_artifact = context.artifacts_dir / "music_plan.cache.json"
        input_fingerprint = artifact_fingerprint(
            assets,
            scenes,
            settings,
            _music_source_fingerprints(
                context.settings.music_library.expanduser().resolve(),
                settings.music_path,
            ),
        )
        config_fingerprint = artifact_fingerprint(
            {
                "generated_music_filename": context.settings.generated_music_filename,
                "music_library": context.settings.music_library.expanduser().resolve(),
                "schema": ARTIFACT_SCHEMA_VERSION,
            }
        )
        if stage_cache_manifest_matches(
            cache_artifact,
            stage=self.name,
            artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
            input_fingerprint=input_fingerprint,
            config_fingerprint=config_fingerprint,
            artifacts=[music_artifact],
        ) and _cached_music_artifact_valid(music_artifact):
            return StageResult(
                stage=self.name,
                skipped=True,
                artifacts=[music_artifact, cache_artifact],
                message="Music selection reused cached soundtrack metadata.",
            )

        music_plan = build_music_plan(
            assets,
            scenes,
            settings,
            context.settings.music_library.expanduser().resolve(),
            context.artifacts_dir / context.settings.generated_music_filename,
            draft_plan,
            neural_generator=None,
        )
        if not _music_plan_source_available(music_plan):
            raise MontageError(
                f"Music selection produced a {music_plan.mode} plan without an available "
                "soundtrack file."
            )
        write_json_atomic(music_artifact, music_plan)
        write_stage_cache_manifest(
            cache_artifact,
            stage=self.name,
            artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
            input_fingerprint=input_fingerprint,
            config_fingerprint=config_fingerprint,
            artifacts=[music_artifact],
        )
        return StageResult(
            stage=self.name,
            artifacts=[music_artifact, cache_artifact],
            message=f"Music selection prepared {music_plan.mode} soundtrack metadata.",
        )


def _cached_music_artifact_valid(music_artifact: Path) -> bool:
    try:
        music_plan = MusicPlan.model_validate_json(music_artifact.read_text(encoding="utf-8"))
    except (OSError, ValidationError):
        return False
    return _music_plan_source_available(music_plan)


def _music_plan_source_available(music_plan: MusicPlan) -> bool:
    if music_plan.mode == "none":
        return True
    return music_plan.source_path is not None and music_plan.source_path.is_file()


def _semantic_montage_settings(context: ProjectContext) -> QuickMontageSettings:
    if context.montage_settings is None:
        return QuickMontageSettings(semantic_analysis=True, story_style=context.style)
    return context.montage_settings.model_copy(
        update={"semantic_analysis": True, "story_style": context.style}
    )


def _music_source_fingerprints(
    music_library: Path, manual_music_path: Path | None
) -> list[dict[str, object]]:
    paths: list[Path] = []
    if music_library.is_dir():
        paths.extend(
            path
            for path in music_library.iterdir()
            if path.is_file() and path.suffix.casefold() in {".mp3", ".wav", ".flac", ".m4a"}
        )
    if manual_music_path is not None:
        paths.append(manual_music_path.expanduser().resolve())
    return [_path_fingerprint(path) for path in sorted(paths, key=lambda item: item.as_posix())]


def _path_fingerprint(path: Path) -> dict[str, object]:
    try:
        stat = path.stat()
    except OSError:
        return {"path": path.as_posix(), "missing": True}
    return {
        "path": path.as_posix(),
        "size_bytes": stat.st_size,
        "modified_ns": stat.st_mtime_ns,
    }
