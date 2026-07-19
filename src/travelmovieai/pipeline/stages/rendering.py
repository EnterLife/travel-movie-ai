"""Pipeline stage that renders the declarative montage timeline with FFmpeg."""

from pathlib import Path

from pydantic import ValidationError

from travelmovieai.application.context import ProjectContext
from travelmovieai.application.disk_space import ensure_render_disk_space
from travelmovieai.application.validation import validate_output_path
from travelmovieai.core.exceptions import MontageError, TravelMovieError
from travelmovieai.domain.enums import PipelineStage, StageStatus
from travelmovieai.domain.models import MontageQualityReport, QuickMontagePlan, StageResult
from travelmovieai.editing.quality_report import (
    build_montage_quality_report,
    enforce_montage_quality,
    enrich_montage_quality_report_with_render,
)
from travelmovieai.editing.renderer import QuickMontageRenderer
from travelmovieai.infrastructure.artifacts import (
    artifact_fingerprint,
    stage_cache_manifest_matches,
    write_json_atomic,
    write_stage_cache_manifest,
)
from travelmovieai.infrastructure.database import MediaAssetRepository
from travelmovieai.infrastructure.ffmpeg import FFprobeClient
from travelmovieai.infrastructure.system import detect_resource_profile
from travelmovieai.pipeline.base import Stage

ARTIFACT_SCHEMA_VERSION = "rendering-v6-media-revisions"
RENDERER_BEHAVIOR_VERSION = "safe-transitions-quality-gate-v8-portable-overlays"


class RenderingStage(Stage):
    name = PipelineStage.RENDERING

    def run(self, context: ProjectContext) -> StageResult:
        timeline_artifact = context.artifacts_dir / "quick_timeline.json"
        if not timeline_artifact.is_file():
            return StageResult(
                stage=self.name,
                status=StageStatus.NO_INPUT,
                message="Rendering needs quick_timeline.json.",
            )

        try:
            plan = QuickMontagePlan.model_validate_json(
                timeline_artifact.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError) as error:
            raise MontageError("Could not read the timeline for rendering.") from error
        if not plan.clips:
            return StageResult(
                stage=self.name,
                status=StageStatus.NO_INPUT,
                artifacts=[timeline_artifact],
                message="Rendering skipped because the timeline has no clips.",
            )

        output_path = validate_output_path(
            context.output_path or context.artifacts_dir / "final.mp4",
            context.input_path,
            workspace=context.workspace,
            database_path=context.database_path,
        )
        quality_artifact = context.artifacts_dir / "montage_quality_report.json"
        cache_artifact = context.artifacts_dir / "rendering.cache.json"
        input_fingerprint = artifact_fingerprint(
            plan,
            _file_revision(plan.music_path),
            _file_revision(plan.narration_path),
        )
        config_fingerprint = artifact_fingerprint(
            {
                "ffmpeg_binary": context.settings.ffmpeg_binary,
                "ffprobe_binary": context.settings.ffprobe_binary,
                "output_path": output_path,
                "workers": context.settings.workers,
                "batch_size": context.settings.batch_size,
                "render_timeout_seconds": context.settings.render_timeout_seconds,
                "renderer_behavior": RENDERER_BEHAVIOR_VERSION,
                "schema": ARTIFACT_SCHEMA_VERSION,
            }
        )
        if stage_cache_manifest_matches(
            cache_artifact,
            stage=self.name,
            artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
            input_fingerprint=input_fingerprint,
            config_fingerprint=config_fingerprint,
            artifacts=[output_path, quality_artifact],
        ) and _cached_render_artifacts_valid(
            quality_artifact,
            output_path,
            context.settings.ffprobe_binary,
        ):
            return StageResult(
                stage=self.name,
                status=StageStatus.CACHED,
                artifacts=[output_path, quality_artifact, cache_artifact],
                message="Rendering reused cached movie and quality report.",
            )

        ensure_render_disk_space(
            workspace=context.workspace,
            output_path=output_path,
            settings=plan.settings,
            reserve_mb=context.settings.render_disk_reserve_mb,
            safety_factor=context.settings.render_disk_safety_factor,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        resources = detect_resource_profile(
            context.settings.ffmpeg_binary,
            worker_override=context.settings.workers,
            batch_override=context.settings.batch_size,
            resource_mode=context.settings.resource_mode,
            gpu_memory_reserve_mb=context.settings.gpu_memory_reserve_mb,
            max_gpu_processes=context.settings.max_gpu_processes,
        )
        renderer = QuickMontageRenderer(
            context.settings.ffmpeg_binary,
            context.settings.ffprobe_binary,
            workers=resources.render_workers,
            ffmpeg_threads=resources.ffmpeg_threads,
            timeout_seconds=context.settings.render_timeout_seconds,
        )
        encoder = (
            renderer.render(plan, output_path, context.cache_dir, context.progress)
            if context.progress is not None
            else renderer.render(plan, output_path, context.cache_dir)
        )

        repository = MediaAssetRepository(context.database_path)
        repository.initialize()
        quality_report = build_montage_quality_report(plan, repository.list_scenes())
        quality_report = enrich_montage_quality_report_with_render(
            quality_report,
            output_path,
            ffprobe_binary=context.settings.ffprobe_binary,
            ffmpeg_binary=context.settings.ffmpeg_binary,
            timeout_seconds=context.settings.render_timeout_seconds,
        )
        quality_report = quality_report.model_copy(update={"render_encoder": encoder})
        write_json_atomic(quality_artifact, quality_report)
        enforce_montage_quality(quality_report)
        repository.record_timeline_version(
            plan,
            phase="rendered",
            variant_name=context.variant_name,
            variant_slug=context.variant_slug,
            output_path=output_path,
        )
        write_stage_cache_manifest(
            cache_artifact,
            stage=self.name,
            artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
            input_fingerprint=input_fingerprint,
            config_fingerprint=config_fingerprint,
            artifacts=[output_path, quality_artifact],
        )
        return StageResult(
            stage=self.name,
            artifacts=[output_path, quality_artifact, cache_artifact],
            message=f"Rendering produced {output_path} with {encoder}.",
        )


def _cached_render_artifacts_valid(
    quality_artifact: Path,
    output_path: Path,
    ffprobe_binary: str,
) -> bool:
    try:
        report = MontageQualityReport.model_validate_json(
            quality_artifact.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError):
        return False
    if (
        not output_path.is_file()
        or report.rendered_path != output_path
        or report.rendered_has_video is not True
        or report.rendered_has_audio is not True
    ):
        return False
    try:
        probe = FFprobeClient(ffprobe_binary).probe(output_path)
    except TravelMovieError:
        return False
    stream_types = {
        stream.get("codec_type")
        for stream in probe.metadata.get("streams", [])
        if isinstance(stream, dict)
    }
    if "video" not in stream_types or "audio" not in stream_types:
        return False
    if probe.duration_seconds is None or probe.duration_seconds <= 0:
        return False
    return not (
        report.rendered_duration_seconds is not None
        and abs(probe.duration_seconds - report.rendered_duration_seconds) > 0.5
    )


def _file_revision(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    try:
        stat = path.stat()
    except OSError as error:
        raise MontageError("Could not inspect a timeline audio source.") from error
    return {
        "path": path,
        "size": stat.st_size,
        "modified_ns": stat.st_mtime_ns,
    }
