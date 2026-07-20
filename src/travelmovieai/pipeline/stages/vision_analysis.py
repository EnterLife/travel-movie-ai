"""Pipeline stage for structured local Vision AI scene understanding."""

from collections.abc import Callable
from pathlib import Path

from travelmovieai.analysis.vision import (
    VISION_CACHE_VERSION,
    VISION_METADATA_KEYS,
    VISION_SCORING_VERSION,
    VisionProvider,
    analyze_scenes,
    scene_vision_input_identity,
)
from travelmovieai.application.context import ProjectContext
from travelmovieai.domain.enums import PipelineStage, StageStatus
from travelmovieai.domain.models import (
    Scene,
    StageCacheManifest,
    StageExecutionMetadata,
    StageResult,
    VisionAnalysisReport,
)
from travelmovieai.infrastructure.artifacts import (
    artifact_fingerprint,
    stage_cache_manifest_matches,
    write_json_atomic,
    write_stage_cache_manifest,
)
from travelmovieai.infrastructure.database import MediaAssetRepository
from travelmovieai.infrastructure.system import ResourceProfile, detect_resource_profile
from travelmovieai.infrastructure.vision import (
    PARSER_VERSION,
    PROMPT_VERSION,
    build_vision_provider,
    resolve_vision_provider_identity,
)
from travelmovieai.pipeline.base import Stage

ARTIFACT_SCHEMA_VERSION = "vision-analysis-v3-temporal-content"
MAX_SCENE_RETRIES = 2
ALLOW_DEGRADED_FALLBACK = True
VisionProviderFactory = Callable[[ProjectContext, str, str, ResourceProfile], VisionProvider]


class VisionAnalysisStage(Stage):
    name = PipelineStage.VISION_ANALYSIS

    def __init__(self, provider_factory: VisionProviderFactory | None = None) -> None:
        self._provider_factory = provider_factory

    def run(self, context: ProjectContext) -> StageResult:
        montage_settings = context.montage_settings
        provider_name = (
            montage_settings.vision_provider
            if montage_settings is not None
            else context.settings.vision_provider
        )
        model_name = (
            montage_settings.vision_model
            if montage_settings is not None and montage_settings.vision_model is not None
            else context.settings.vision_model
        )
        repository = MediaAssetRepository(context.database_path)
        repository.initialize()
        scenes = repository.list_scenes()
        artifact = context.artifacts_dir / "vision_analysis.json"
        resources = context.resources or detect_resource_profile(
            context.settings.ffmpeg_binary,
            worker_override=context.settings.workers,
            batch_override=context.settings.batch_size,
            resource_mode=context.settings.resource_mode,
            gpu_memory_reserve_mb=context.settings.gpu_memory_reserve_mb,
            max_gpu_processes=context.settings.max_gpu_processes,
        )
        provider_device = (
            resources.device
            if context.settings.device == "auto"
            else "cpu"
            if context.settings.device == "directml"
            else context.settings.device
        )
        input_fingerprint = artifact_fingerprint(_vision_inputs(scenes))
        config_fingerprint = artifact_fingerprint(
            {
                "provider": provider_name,
                "model": model_name,
                "resolved_provider": resolve_vision_provider_identity(
                    provider=provider_name,
                    model=model_name,
                    device=provider_device,
                    gpu_memory_mb=resources.gpu_memory_mb,
                    system_memory_mb=resources.memory_mb,
                    model_batch_size=resources.model_batch_size,
                ),
                "device": provider_device,
                "allow_model_download": context.settings.allow_model_download,
                "model_batch_size": resources.model_batch_size,
                "style": context.style,
                "prompt_version": PROMPT_VERSION,
                "parser_version": PARSER_VERSION,
                "scoring_version": VISION_SCORING_VERSION,
                "scene_cache_version": VISION_CACHE_VERSION,
                "max_scene_retries": MAX_SCENE_RETRIES,
                "allow_degraded_fallback": ALLOW_DEGRADED_FALLBACK,
                "schema": ARTIFACT_SCHEMA_VERSION,
            }
        )
        cache_artifact = context.artifacts_dir / "vision_analysis.cache.json"
        cached_report = _read_vision_analysis(artifact)
        content_migration_allowed = _vision_cache_manifest_config_matches(
            cache_artifact,
            config_fingerprint=config_fingerprint,
            artifact=artifact,
        )
        if (
            stage_cache_manifest_matches(
                cache_artifact,
                stage=self.name,
                artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
                input_fingerprint=input_fingerprint,
                config_fingerprint=config_fingerprint,
                artifacts=[artifact],
            )
            and cached_report is not None
            and _vision_analysis_cache_valid(
                cached_report,
                scenes,
            )
        ):
            restored_scenes = _restore_cached_vision_state(scenes, cached_report.scenes)
            if restored_scenes != scenes:
                repository.synchronize_scenes(restored_scenes)
            return StageResult(
                stage=self.name,
                status=StageStatus.CACHED,
                artifacts=[context.database_path, artifact, cache_artifact],
                message="Vision AI reused cached analysis artifacts.",
                execution=StageExecutionMetadata(
                    retry_count=cached_report.retry_count,
                    fallback_count=cached_report.degraded_count,
                    provider=cached_report.provider,
                    model=cached_report.model,
                ),
            )

        provider: VisionProvider = (
            self._provider_factory(context, provider_name, model_name, resources)
            if self._provider_factory is not None
            else build_vision_provider(
                provider=provider_name,
                model=model_name,
                device=provider_device,
                cache_dir=context.settings.model_cache.expanduser().resolve(),
                allow_download=context.settings.allow_model_download,
                gpu_memory_mb=resources.gpu_memory_mb,
                system_memory_mb=resources.memory_mb,
                model_batch_size=resources.model_batch_size,
                model_pool_size=context.settings.vision_model_pool_size,
            )
        )
        try:
            report = analyze_scenes(
                scenes,
                provider,
                context.style,
                cached_report=cached_report,
                checkpoint=lambda partial: write_json_atomic(artifact, partial),
                progress=context.progress,
                max_scene_retries=MAX_SCENE_RETRIES,
                allow_degraded_fallback=ALLOW_DEGRADED_FALLBACK,
                allow_content_identity_migration=content_migration_allowed,
            )
        finally:
            release = getattr(provider, "release", None)
            if callable(release):
                release()
        repository.synchronize_scenes(report.scenes)
        write_json_atomic(artifact, report)
        write_stage_cache_manifest(
            cache_artifact,
            stage=self.name,
            artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
            input_fingerprint=input_fingerprint,
            config_fingerprint=config_fingerprint,
            artifacts=[artifact],
        )
        return StageResult(
            stage=self.name,
            status=(
                StageStatus.CACHED
                if report.analyzed_count == 0
                and report.cached_count > 0
                and report.degraded_count == 0
                else StageStatus.NO_INPUT
                if not report.scenes
                else StageStatus.DEGRADED
                if report.degraded_count > 0
                else StageStatus.COMPLETED
            ),
            artifacts=[context.database_path, artifact, cache_artifact],
            message=(
                f"Vision AI analyzed {report.analyzed_count} scene(s), "
                f"{report.cached_count} cached, {report.degraded_count} degraded, "
                f"model {report.model}."
            ),
            execution=StageExecutionMetadata(
                retry_count=report.retry_count,
                fallback_count=report.degraded_count,
                provider=report.provider,
                fallback_provider="deterministic" if report.degraded_count else None,
                model=report.model,
            ),
        )


def _cached_vision_analysis_valid(artifact: Path, scenes: list[Scene]) -> bool:
    report = _read_vision_analysis(artifact)
    return report is not None and _vision_analysis_cache_valid(report, scenes)


def _read_vision_analysis(artifact: Path) -> VisionAnalysisReport | None:
    try:
        return VisionAnalysisReport.model_validate_json(artifact.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _vision_cache_manifest_config_matches(
    path: Path,
    *,
    config_fingerprint: str,
    artifact: Path,
) -> bool:
    if not path.is_file() or not artifact.is_file():
        return False
    try:
        manifest = StageCacheManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        manifest.stage is PipelineStage.VISION_ANALYSIS
        and manifest.artifact_schema_version == ARTIFACT_SCHEMA_VERSION
        and manifest.config_fingerprint == config_fingerprint
    )


def _vision_analysis_cache_valid(
    report: VisionAnalysisReport,
    scenes: list[Scene],
) -> bool:
    if {scene.id for scene in report.scenes} != {scene.id for scene in scenes}:
        return False
    if report.degraded_count > 0:
        return False
    return all(
        scene.keyframe_path is None
        or (
            scene.caption
            and scene.importance_score is not None
            and scene.metadata.get("vision_status") != "degraded"
        )
        for scene in report.scenes
    )


def _vision_inputs(scenes: list[Scene]) -> list[dict[str, object]]:
    return [
        {
            "id": str(scene.id),
            "asset_id": str(scene.asset_id),
            "start_seconds": scene.start_seconds,
            "end_seconds": scene.end_seconds,
            "keyframe_path": scene.keyframe_path,
            "quality_score": scene.quality_score,
            "scene_cache_key": scene.metadata.get("cache_key"),
            "quality_metrics": scene.metadata.get("quality_metrics"),
            "vision_input_identity": scene_vision_input_identity(scene),
        }
        for scene in sorted(scenes, key=lambda item: str(item.id))
    ]


def _restore_cached_vision_state(
    current_scenes: list[Scene],
    cached_scenes: list[Scene],
) -> list[Scene]:
    cached_by_id = {scene.id: scene for scene in cached_scenes}
    restored: list[Scene] = []
    for scene in current_scenes:
        cached = cached_by_id[scene.id]
        metadata = {
            key: value for key, value in scene.metadata.items() if key not in VISION_METADATA_KEYS
        }
        metadata.update(
            {key: value for key, value in cached.metadata.items() if key in VISION_METADATA_KEYS}
        )
        restored.append(
            scene.model_copy(
                update={
                    "caption": cached.caption,
                    "importance_score": cached.importance_score,
                    "metadata": metadata,
                }
            )
        )
    return restored
