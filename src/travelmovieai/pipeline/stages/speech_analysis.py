"""Pipeline stage for optional scene-level Faster Whisper transcription."""

from pathlib import Path

from travelmovieai.analysis.speech import analyze_speech
from travelmovieai.application.context import ProjectContext
from travelmovieai.domain.enums import MediaType, PipelineStage, StageStatus
from travelmovieai.domain.models import MediaAsset, Scene, SpeechAnalysisReport, StageResult
from travelmovieai.infrastructure.artifacts import (
    artifact_fingerprint,
    stage_cache_manifest_matches,
    write_json_atomic,
    write_stage_cache_manifest,
)
from travelmovieai.infrastructure.database import MediaAssetRepository
from travelmovieai.infrastructure.whisper import FasterWhisperProvider
from travelmovieai.pipeline.base import Stage
from travelmovieai.pipeline.state import SPEECH_STATE, clear_stage_owned_state

ARTIFACT_SCHEMA_VERSION = "speech-analysis-v2"


class SpeechAnalysisStage(Stage):
    name = PipelineStage.SPEECH_ANALYSIS

    def run(self, context: ProjectContext) -> StageResult:
        if context.montage_settings is not None and not context.montage_settings.speech_analysis:
            clear_stage_owned_state(context, SPEECH_STATE)
            return StageResult(
                stage=self.name,
                status=StageStatus.DISABLED,
                message="Speech analysis disabled by montage settings.",
            )

        repository = MediaAssetRepository(context.database_path)
        repository.initialize()
        scenes = repository.list_scenes()
        assets = repository.list_assets()
        artifact = context.artifacts_dir / "speech_analysis.json"
        cache_artifact = context.artifacts_dir / "speech_analysis.cache.json"
        if not _has_eligible_audio_scene(scenes, assets):
            clear_stage_owned_state(context, SPEECH_STATE)
            return StageResult(
                stage=self.name,
                status=StageStatus.NO_INPUT,
                message="Speech analysis needs a video scene with an audio stream.",
            )
        input_fingerprint = artifact_fingerprint(
            _speech_scene_inputs(scenes), _asset_inputs(assets)
        )
        config_fingerprint = artifact_fingerprint(
            {
                "whisper_model": context.settings.whisper_model,
                "device": context.settings.device,
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
        ) and _cached_speech_analysis_valid(artifact, scenes):
            return StageResult(
                stage=self.name,
                status=StageStatus.CACHED,
                artifacts=[context.database_path, artifact, cache_artifact],
                message="Speech analysis reused cached transcripts.",
            )

        provider = FasterWhisperProvider(
            context.settings.whisper_model,
            context.settings.device,
            cache_dir=(context.settings.model_cache / "faster-whisper").expanduser().resolve(),
            allow_download=context.settings.allow_model_download,
        )
        try:
            report = analyze_speech(
                scenes,
                assets,
                provider,
                context.settings.ffmpeg_binary,
                context.cache_dir / "speech",
                timeout_seconds=context.settings.frame_extraction_timeout_seconds,
                progress=context.progress,
            )
        finally:
            provider.release()
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
                if report.transcribed_count == 0 and report.cached_count > 0
                else StageStatus.COMPLETED
            ),
            artifacts=[context.database_path, artifact, cache_artifact],
            message=(
                f"Speech analysis transcribed {report.transcribed_count} scene(s), "
                f"{report.cached_count} cached."
            ),
        )


def _cached_speech_analysis_valid(artifact: Path, scenes: list[Scene]) -> bool:
    try:
        report = SpeechAnalysisReport.model_validate_json(artifact.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if {scene.id for scene in report.scenes} != {scene.id for scene in scenes}:
        return False
    reported = {scene.id: scene for scene in report.scenes}
    return all(
        scene.transcript == reported[scene.id].transcript
        and scene.metadata.get("speech_cache_key")
        == reported[scene.id].metadata.get("speech_cache_key")
        for scene in scenes
    )


def _speech_scene_inputs(scenes: list[Scene]) -> list[dict[str, object]]:
    return [
        {
            "id": str(scene.id),
            "asset_id": str(scene.asset_id),
            "start_seconds": scene.start_seconds,
            "end_seconds": scene.end_seconds,
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
