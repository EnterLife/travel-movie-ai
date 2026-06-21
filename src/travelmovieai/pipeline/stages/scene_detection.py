"""Pipeline stage that detects video scenes and persists their boundaries."""

from datetime import UTC, datetime

from travelmovieai.analysis.scenes import SceneDetector, scene_cache_key
from travelmovieai.application.context import ProjectContext
from travelmovieai.domain.enums import MediaType, PipelineStage
from travelmovieai.domain.models import (
    QuickMontageSettings,
    Scene,
    SceneDetectionReport,
    StageResult,
)
from travelmovieai.infrastructure.artifacts import write_json_atomic
from travelmovieai.infrastructure.database import MediaAssetRepository
from travelmovieai.pipeline.base import Stage


class SceneDetectionStage(Stage):
    name = PipelineStage.SCENE_DETECTION

    def __init__(
        self,
        settings: QuickMontageSettings | None = None,
        detector: SceneDetector | None = None,
    ) -> None:
        self._settings = settings
        self._detector = detector or SceneDetector()

    def run(self, context: ProjectContext) -> StageResult:
        settings = self._settings or context.montage_settings or QuickMontageSettings(
            story_style=context.style
        )
        repository = MediaAssetRepository(context.database_path)
        repository.initialize()
        assets = repository.list_assets()
        existing = _group_scenes(repository.list_scenes())
        scenes: list[Scene] = []
        cached_count = 0
        fallback_count = 0

        for asset in assets:
            if asset.scan_error or asset.media_type not in {MediaType.VIDEO, MediaType.PHOTO}:
                continue
            cached = existing.get(str(asset.id), [])
            expected_key = scene_cache_key(asset, settings)
            if cached and all(scene.metadata.get("cache_key") == expected_key for scene in cached):
                scenes.extend(cached)
                cached_count += len(cached)
                continue
            detected, used_fallback = self._detector.detect(asset, settings)
            scenes.extend(detected)
            fallback_count += len(detected) if used_fallback else 0

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
            skipped=report.detected_count == 0,
            artifacts=[context.database_path, artifact],
            message=(
                f"Scene detection produced {len(scenes)} scene(s): "
                f"{report.detected_count} detected, {cached_count} cached, "
                f"{fallback_count} from fallback."
            ),
        )


def _group_scenes(scenes: list[Scene]) -> dict[str, list[Scene]]:
    grouped: dict[str, list[Scene]] = {}
    for scene in scenes:
        grouped.setdefault(str(scene.asset_id), []).append(scene)
    return grouped
