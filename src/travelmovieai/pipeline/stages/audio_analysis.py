"""Pipeline stage for scene-level audio context classification."""

from pathlib import Path

from travelmovieai.analysis.audio import analyze_audio
from travelmovieai.application.context import ProjectContext
from travelmovieai.domain.enums import MediaType, PipelineStage, StageStatus
from travelmovieai.domain.models import AudioAnalysisReport, MediaAsset, Scene, StageResult
from travelmovieai.infrastructure.artifacts import (
    artifact_fingerprint,
    stage_cache_manifest_matches,
    write_json_atomic,
    write_stage_cache_manifest,
)
from travelmovieai.infrastructure.database import MediaAssetRepository
from travelmovieai.pipeline.base import Stage
from travelmovieai.pipeline.state import AUDIO_STATE, clear_stage_owned_state

ARTIFACT_SCHEMA_VERSION = "audio-analysis-v3"


class AudioAnalysisStage(Stage):
    name = PipelineStage.AUDIO_ANALYSIS

    def run(self, context: ProjectContext) -> StageResult:
        if context.montage_settings is not None and not context.montage_settings.audio_analysis:
            clear_stage_owned_state(context, AUDIO_STATE)
            return StageResult(
                stage=self.name,
                status=StageStatus.DISABLED,
                message="Audio analysis disabled by montage settings.",
            )

        with MediaAssetRepository(context.database_path) as repository:
            repository.initialize()
            scenes = repository.list_scenes()
            assets = repository.list_assets()
            artifact = context.artifacts_dir / "audio_analysis.json"
            cache_artifact = context.artifacts_dir / "audio_analysis.cache.json"
            if not _has_eligible_audio_scene(scenes, assets):
                clear_stage_owned_state(context, AUDIO_STATE)
                return StageResult(
                    stage=self.name,
                    status=StageStatus.NO_INPUT,
                    message="Audio analysis needs a video scene with an audio stream.",
                )
            input_fingerprint = artifact_fingerprint(
                _audio_scene_inputs(scenes), _asset_inputs(assets)
            )
            config_fingerprint = artifact_fingerprint(
                {
                    "ffmpeg_binary": context.settings.ffmpeg_binary,
                    "timeout_seconds": context.settings.frame_extraction_timeout_seconds,
                    "schema": ARTIFACT_SCHEMA_VERSION,
                }
            )
            if stage_cache_manifest_matches(
                cache_artifact,
                stage=self.name,
                artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
                input_fingerprint=input_fingerprint,
                config_fingerprint=config_fingerprint,
                artifacts=[artifact],
            ) and _cached_audio_analysis_valid(artifact, scenes):
                return StageResult(
                    stage=self.name,
                    status=StageStatus.CACHED,
                    artifacts=[context.database_path, artifact, cache_artifact],
                    message="Audio analysis reused cached scene audio metadata.",
                )

            report = analyze_audio(
                scenes,
                assets,
                context.settings.ffmpeg_binary,
                timeout_seconds=context.settings.frame_extraction_timeout_seconds,
                progress=context.progress,
            )
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
            artifacts=[context.database_path, artifact, cache_artifact],
            message=(
                f"Audio analysis classified {report.analyzed_count} scene(s), "
                f"{report.skipped_count} skipped."
            ),
        )


def _cached_audio_analysis_valid(artifact: Path, scenes: list[Scene]) -> bool:
    try:
        report = AudioAnalysisReport.model_validate_json(artifact.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if {scene.id for scene in report.scenes} != {scene.id for scene in scenes}:
        return False
    reported = {scene.id: scene for scene in report.scenes}
    return all(
        scene.metadata.get("audio_analysis") == reported[scene.id].metadata.get("audio_analysis")
        for scene in scenes
    )


def _audio_scene_inputs(scenes: list[Scene]) -> list[dict[str, object]]:
    return [
        {
            "id": str(scene.id),
            "asset_id": str(scene.asset_id),
            "start_seconds": scene.start_seconds,
            "end_seconds": scene.end_seconds,
            "transcript": scene.transcript,
            "speech_cache_key": scene.metadata.get("speech_cache_key"),
            "scene_cache_key": scene.metadata.get("cache_key"),
        }
        for scene in sorted(scenes, key=lambda item: str(item.id))
    ]


def _asset_inputs(assets: list[MediaAsset]) -> list[dict[str, object]]:
    return [
        {
            "id": str(asset.id),
            "path": asset.path,
            "size_bytes": asset.size_bytes,
            "modified_ns": asset.modified_ns,
            "duration_seconds": asset.duration_seconds,
            "streams": asset.probe_metadata.get("streams"),
        }
        for asset in sorted(assets, key=lambda item: str(item.id))
    ]


def _has_eligible_audio_scene(scenes: list[Scene], assets: list[MediaAsset]) -> bool:
    eligible_assets = {
        asset.id
        for asset in assets
        if asset.media_type is MediaType.VIDEO
        and any(
            isinstance(stream, dict) and stream.get("codec_type") == "audio"
            for stream in asset.probe_metadata.get("streams", [])
        )
    }
    return any(scene.asset_id in eligible_assets for scene in scenes)
