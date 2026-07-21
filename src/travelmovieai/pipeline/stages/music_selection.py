"""Pipeline stage that plans soundtrack cues for the montage."""

from collections.abc import Callable
from pathlib import Path
from typing import cast

from pydantic import ValidationError

from travelmovieai.application.context import ProjectContext
from travelmovieai.core.exceptions import MontageError
from travelmovieai.domain.enums import PipelineStage, StageStatus
from travelmovieai.domain.models import (
    MusicPlan,
    QuickMontageSettings,
    StageExecutionMetadata,
    StageResult,
)
from travelmovieai.editing.timeline import build_semantic_montage_plan
from travelmovieai.infrastructure.artifacts import (
    artifact_fingerprint,
    stage_cache_manifest_matches,
    write_json_atomic,
    write_stage_cache_manifest,
)
from travelmovieai.infrastructure.database import MediaAssetRepository
from travelmovieai.infrastructure.music_generation import (
    AceStepMusicGenerator,
    resolve_local_music_model,
)
from travelmovieai.infrastructure.system import detect_resource_profile
from travelmovieai.pipeline.base import Stage
from travelmovieai.story.music import (
    MusicPlanExecution,
    NeuralMusicGenerator,
    build_music_plan,
    music_source_content_sha256,
)

ARTIFACT_SCHEMA_VERSION = "music-selection-v5-tail-audit"
MusicGeneratorFactory = Callable[
    [ProjectContext, QuickMontageSettings],
    NeuralMusicGenerator | None,
]


class MusicSelectionStage(Stage):
    name = PipelineStage.MUSIC_SELECTION

    def __init__(self, generator_factory: MusicGeneratorFactory | None = None) -> None:
        self._generator_factory = generator_factory

    def run(self, context: ProjectContext) -> StageResult:
        repository = MediaAssetRepository(context.database_path)
        repository.initialize()
        assets = repository.list_assets()
        scenes = repository.list_scenes()
        music_artifact = context.artifacts_dir / "music_plan.json"
        cache_artifact = context.artifacts_dir / "music_plan.cache.json"
        if not assets or not scenes:
            music_artifact.unlink(missing_ok=True)
            cache_artifact.unlink(missing_ok=True)
            return StageResult(
                stage=self.name,
                status=StageStatus.NO_INPUT,
                message="Music selection needs media assets and ranked scenes.",
            )

        settings = _semantic_montage_settings(context)
        if not settings.music_enabled or settings.music_mode == "none":
            music_plan = MusicPlan(
                mode="none",
                duration_seconds=0,
                reasoning="Music disabled by montage settings.",
            )
            write_json_atomic(music_artifact, music_plan)
            cache_artifact.unlink(missing_ok=True)
            return StageResult(
                stage=self.name,
                status=StageStatus.DISABLED,
                artifacts=[music_artifact],
                message="Music selection disabled by montage settings.",
            )

        draft_plan = build_semantic_montage_plan(assets, scenes, settings)
        input_fingerprint = artifact_fingerprint(
            assets,
            scenes,
            settings,
            _music_source_fingerprints(
                context.settings.music_library.expanduser().resolve(),
                settings.music_path,
                settings.music_reference_path,
                settings.music_lora_path,
            ),
        )
        config_fingerprint = artifact_fingerprint(
            {
                "generated_music_filename": context.settings.generated_music_filename,
                "music_library": context.settings.music_library.expanduser().resolve(),
                "schema": ARTIFACT_SCHEMA_VERSION,
            }
        )
        cached_music_plan = _read_music_artifact(music_artifact)
        if (
            stage_cache_manifest_matches(
                cache_artifact,
                stage=self.name,
                artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
                input_fingerprint=input_fingerprint,
                config_fingerprint=config_fingerprint,
                artifacts=[music_artifact],
            )
            and cached_music_plan is not None
            and _music_plan_cache_valid(cached_music_plan)
        ):
            return StageResult(
                stage=self.name,
                status=StageStatus.CACHED,
                artifacts=[music_artifact, cache_artifact],
                message="Music selection reused cached soundtrack metadata.",
                execution=_music_execution_metadata(cached_music_plan),
            )

        neural_generator = (
            self._generator_factory(context, settings)
            if self._generator_factory is not None
            else _neural_music_generator(context, settings)
        )
        plan_execution = MusicPlanExecution()
        music_plan = build_music_plan(
            assets,
            scenes,
            settings,
            context.settings.music_library.expanduser().resolve(),
            context.artifacts_dir / context.settings.generated_music_filename,
            draft_plan,
            neural_generator=neural_generator,
            ffmpeg_binary=context.settings.ffmpeg_binary,
            progress=context.progress,
            execution=plan_execution,
        )
        if not _music_plan_source_available(music_plan):
            raise MontageError(
                f"Music selection produced a {music_plan.mode} plan without an available "
                "soundtrack file."
            )
        music_plan = _music_plan_with_source_revision(music_plan)
        write_json_atomic(music_artifact, music_plan)
        if music_plan.fallback_used:
            cache_artifact.unlink(missing_ok=True)
        else:
            write_stage_cache_manifest(
                cache_artifact,
                stage=self.name,
                artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
                input_fingerprint=input_fingerprint,
                config_fingerprint=config_fingerprint,
                artifacts=[music_artifact],
            )
        artifacts = [music_artifact]
        if not music_plan.fallback_used:
            artifacts.append(cache_artifact)
        return StageResult(
            stage=self.name,
            status=(
                StageStatus.DEGRADED
                if music_plan.fallback_used
                else StageStatus.CACHED
                if plan_execution.cache_hit
                else StageStatus.COMPLETED
            ),
            cache_hit=plan_execution.cache_hit,
            artifacts=artifacts,
            message=(
                f"Music selection prepared {music_plan.mode} soundtrack metadata"
                + (" with procedural fallback." if music_plan.fallback_used else ".")
            ),
            execution=_music_execution_metadata(
                music_plan,
                primary_provider=(
                    neural_generator.name
                    if neural_generator is not None
                    else "ace-step"
                    if music_plan.fallback_used and settings.music_engine in {"auto", "ace-step"}
                    else None
                ),
                primary_model=(neural_generator.model if neural_generator is not None else None),
            ),
        )


def _cached_music_artifact_valid(music_artifact: Path) -> bool:
    music_plan = _read_music_artifact(music_artifact)
    return music_plan is not None and _music_plan_cache_valid(music_plan)


def _read_music_artifact(music_artifact: Path) -> MusicPlan | None:
    try:
        return MusicPlan.model_validate_json(music_artifact.read_text(encoding="utf-8"))
    except (OSError, ValidationError):
        return None


def _music_plan_cache_valid(music_plan: MusicPlan) -> bool:
    if music_plan.fallback_used or not _music_plan_source_available(music_plan):
        return False
    if music_plan.mode != "generated":
        return True
    return (
        music_plan.source_path is not None
        and music_plan.source_content_sha256 is not None
        and music_source_content_sha256(music_plan.source_path) == music_plan.source_content_sha256
    )


def _music_plan_with_source_revision(music_plan: MusicPlan) -> MusicPlan:
    if music_plan.mode != "generated" or music_plan.source_path is None:
        return music_plan
    content_revision = music_source_content_sha256(music_plan.source_path)
    if content_revision is None:
        raise MontageError("Generated soundtrack could not be fingerprinted safely.")
    return music_plan.model_copy(update={"source_content_sha256": content_revision})


def _music_execution_metadata(
    music_plan: MusicPlan,
    *,
    primary_provider: str | None = None,
    primary_model: str | None = None,
) -> StageExecutionMetadata:
    return StageExecutionMetadata(
        fallback_count=int(music_plan.fallback_used),
        provider=primary_provider if music_plan.fallback_used else music_plan.generator,
        fallback_provider=music_plan.generator if music_plan.fallback_used else None,
        model=primary_model if music_plan.fallback_used else music_plan.model,
    )


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


def _neural_music_generator(
    context: ProjectContext,
    settings: QuickMontageSettings,
) -> NeuralMusicGenerator | None:
    if settings.music_mode not in {"auto", "generated"} or settings.music_engine == "procedural":
        return None
    resources = context.resources or detect_resource_profile(
        context.settings.ffmpeg_binary,
        worker_override=context.settings.workers,
        batch_override=context.settings.batch_size,
        resource_mode=context.settings.resource_mode,
        gpu_memory_reserve_mb=context.settings.gpu_memory_reserve_mb,
        max_gpu_processes=context.settings.max_gpu_processes,
    )
    model = resolve_local_music_model(
        settings.music_model or context.settings.music_model,
        gpu_memory_mb=resources.gpu_memory_mb,
        quality=settings.music_quality,
    )
    return cast(
        NeuralMusicGenerator,
        AceStepMusicGenerator(
            model,
            runtime_dir=Path(".cache/ace-step").resolve(),
            model_cache=(context.settings.model_cache / "ace-step").expanduser().resolve(),
            ffmpeg_binary=context.settings.ffmpeg_binary,
            allow_download=context.settings.allow_model_download,
            device=context.settings.device,
            gpu_memory_mb=resources.gpu_memory_mb,
            ffmpeg_timeout_seconds=context.settings.render_timeout_seconds,
            cancel_requested=(
                (lambda: _progress_heartbeat(context.progress))
                if context.progress is not None
                else None
            ),
            quality=settings.music_quality,
            reference_audio=(
                settings.music_reference_path.expanduser().resolve()
                if settings.music_reference_path is not None
                else None
            ),
            reference_strength=settings.music_reference_strength,
            lora_path=(
                settings.music_lora_path.expanduser().resolve()
                if settings.music_lora_path is not None
                else None
            ),
            lora_strength=settings.music_lora_strength,
        ),
    )


def _progress_heartbeat(progress: Callable[[int, int, str], None] | None) -> bool:
    if progress is not None:
        progress(1, 4, "ACE-Step: generation is still running")
    return False


def _music_source_fingerprints(
    music_library: Path,
    manual_music_path: Path | None,
    reference_music_path: Path | None,
    lora_path: Path | None,
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
    if reference_music_path is not None:
        paths.append(reference_music_path.expanduser().resolve())
    if lora_path is not None:
        resolved_lora = lora_path.expanduser().resolve()
        if resolved_lora.is_dir():
            paths.extend(
                path
                for path in resolved_lora.rglob("*")
                if path.is_file()
                and path.suffix.casefold() in {".safetensors", ".bin", ".pt", ".json"}
            )
        else:
            paths.append(resolved_lora)
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
