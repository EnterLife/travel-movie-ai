"""Pipeline stage that detects video scenes and persists their boundaries."""

from datetime import UTC, datetime

from travelmovieai.analysis.scenes import SceneDetector, scene_cache_key
from travelmovieai.application.context import ProjectContext
from travelmovieai.domain.enums import MediaType, PipelineStage, StageStatus
from travelmovieai.domain.models import (
    MediaAsset,
    QuickMontageSettings,
    Scene,
    SceneDetectionReport,
    StageResult,
)
from travelmovieai.infrastructure.artifacts import write_json_atomic
from travelmovieai.infrastructure.database import MediaAssetRepository
from travelmovieai.media.proxy import AnalysisMedia, AnalysisProxyManager
from travelmovieai.pipeline.base import Stage


class SceneDetectionStage(Stage):
    name = PipelineStage.SCENE_DETECTION

    def __init__(
        self,
        settings: QuickMontageSettings | None = None,
        detector: SceneDetector | None = None,
        proxy_manager: AnalysisProxyManager | None = None,
    ) -> None:
        self._settings = settings
        self._detector = detector or SceneDetector()
        self._proxy_manager = proxy_manager

    def run(self, context: ProjectContext) -> StageResult:
        settings = (
            self._settings
            or context.montage_settings
            or QuickMontageSettings(story_style=context.style)
        )
        repository = MediaAssetRepository(context.database_path)
        repository.initialize()
        assets = repository.list_assets()
        existing = _group_scenes(repository.list_scenes())
        scenes: list[Scene] = []
        cached_count = 0
        fallback_count = 0
        generated_proxies = 0
        cached_proxies = 0
        proxy_manager = self._proxy_manager or AnalysisProxyManager(
            context.cache_dir / "proxies",
            ffmpeg_binary=context.settings.ffmpeg_binary,
            ffprobe_binary=context.settings.ffprobe_binary,
            mode=context.settings.analysis_proxy_mode,
            max_dimension=context.settings.analysis_proxy_max_dimension,
            video_bitrate_mbps=context.settings.analysis_proxy_video_bitrate_mbps,
            timeout_seconds=context.settings.analysis_proxy_timeout_seconds,
        )
        proxy_cache_identity = proxy_manager.cache_identity()

        total_assets = len(assets)
        for index, asset in enumerate(assets, start=1):
            if asset.scan_error or asset.media_type not in {MediaType.VIDEO, MediaType.PHOTO}:
                if context.progress is not None:
                    context.progress(
                        index,
                        total_assets,
                        f"Scene detection: {index}/{total_assets}",
                    )
                continue
            cached = existing.get(str(asset.id), [])
            expected_key = scene_cache_key(
                asset,
                settings,
                analysis_fingerprint=proxy_cache_identity,
            )
            if cached and all(scene.metadata.get("cache_key") == expected_key for scene in cached):
                scenes.extend(cached)
                cached_count += len(cached)
                if context.progress is not None:
                    context.progress(
                        index,
                        total_assets,
                        f"Scene detection cache: {index}/{total_assets}",
                    )
                continue
            analysis_media = proxy_manager.resolve(asset)
            analysis_asset = _analysis_asset(
                asset,
                analysis_media,
                proxy_cache_identity=proxy_cache_identity,
            )
            if analysis_media.proxied:
                if analysis_media.cache_hit:
                    cached_proxies += 1
                else:
                    generated_proxies += 1
            detected, used_fallback = self._detector.detect(analysis_asset, settings)
            scenes.extend(detected)
            fallback_count += len(detected) if used_fallback else 0
            if context.progress is not None:
                context.progress(
                    index,
                    total_assets,
                    f"Scene detection: {index}/{total_assets}",
                )

        repository.synchronize_scenes(scenes)
        report = SceneDetectionReport(
            created_at=datetime.now(UTC),
            scenes=scenes,
            detected_count=len(scenes) - cached_count,
            cached_count=cached_count,
            fallback_count=fallback_count,
        )
        artifact = context.artifacts_dir / "scenes.json"
        write_json_atomic(artifact, report)
        return StageResult(
            stage=self.name,
            status=(
                StageStatus.CACHED
                if report.detected_count == 0 and cached_count > 0
                else StageStatus.NO_INPUT
                if not scenes
                else StageStatus.COMPLETED
            ),
            artifacts=[context.database_path, artifact],
            message=(
                f"Scene detection produced {len(scenes)} scene(s): "
                f"{report.detected_count} detected, {cached_count} cached, "
                f"{fallback_count} from fallback; proxies={generated_proxies} generated/"
                f"{cached_proxies} cached."
            ),
        )


def _group_scenes(scenes: list[Scene]) -> dict[str, list[Scene]]:
    grouped: dict[str, list[Scene]] = {}
    for scene in scenes:
        grouped.setdefault(str(scene.asset_id), []).append(scene)
    return grouped


def _analysis_asset(
    asset: MediaAsset,
    media: AnalysisMedia,
    *,
    proxy_cache_identity: str,
) -> MediaAsset:
    updates: dict[str, object] = {
        "probe_metadata": {
            **asset.probe_metadata,
            "scene_analysis_fingerprint": proxy_cache_identity,
        }
    }
    if media.proxied:
        updates.update(
            {
                "path": media.analysis_path,
                "width": media.width,
                "height": media.height,
                "duration_seconds": media.duration_seconds or asset.duration_seconds,
                "probe_metadata": {
                    **asset.probe_metadata,
                    "scene_analysis_fingerprint": proxy_cache_identity,
                    "analysis_proxy_fingerprint": media.cache_key,
                    "video_duration_seconds": media.duration_seconds,
                },
            }
        )
    return asset.model_copy(update=updates)
