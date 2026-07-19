"""Pipeline stage for local semantic scene embeddings and FAISS indexing."""

from collections.abc import Callable
from importlib.util import find_spec
from pathlib import Path

from pydantic import ValidationError

from travelmovieai.analysis.embeddings import EMBEDDING_BACKEND, embed_scenes
from travelmovieai.application.context import ProjectContext
from travelmovieai.domain.enums import PipelineStage, StageStatus
from travelmovieai.domain.models import (
    EmbeddingAnalysisReport,
    Scene,
    SemanticIndexManifest,
    StageResult,
)
from travelmovieai.infrastructure.artifacts import (
    artifact_fingerprint,
    stage_cache_manifest_matches,
    write_json_atomic,
    write_stage_cache_manifest,
)
from travelmovieai.infrastructure.database import MediaAssetRepository
from travelmovieai.infrastructure.embeddings import (
    SentenceTransformerEmbeddingProvider,
    TextEmbeddingProvider,
)
from travelmovieai.infrastructure.semantic_index import build_semantic_index
from travelmovieai.pipeline.base import Stage

ARTIFACT_SCHEMA_VERSION = "embeddings-v2"
EmbeddingProviderFactory = Callable[[ProjectContext], TextEmbeddingProvider]


class EmbeddingsStage(Stage):
    name = PipelineStage.EMBEDDINGS

    def __init__(self, provider_factory: EmbeddingProviderFactory | None = None) -> None:
        self._provider_factory = provider_factory

    def run(self, context: ProjectContext) -> StageResult:
        repository = MediaAssetRepository(context.database_path)
        repository.initialize()
        scenes = repository.list_scenes()
        artifact = context.artifacts_dir / "embeddings.json"
        cache_artifact = context.artifacts_dir / "embeddings.cache.json"
        index_artifact = context.artifacts_dir / "embeddings.faiss"
        index_manifest = context.artifacts_dir / "embeddings.index.json"
        if not scenes:
            _remove_stale_artifacts(
                artifact,
                cache_artifact,
                index_artifact,
                index_manifest,
            )
            return StageResult(
                stage=self.name,
                status=StageStatus.NO_INPUT,
                message="Embeddings need analyzed scenes.",
            )

        configured_backend = context.settings.embedding_backend
        backend = (
            EMBEDDING_BACKEND if configured_backend == "feature-hash" else "sentence-transformers"
        )
        model = (
            context.settings.embedding_model
            if configured_backend == "sentence-transformers"
            else None
        )
        index_enabled = _index_enabled(context.settings.embedding_index)
        expected_artifacts = [artifact]
        if index_enabled:
            expected_artifacts.extend([index_artifact, index_manifest])
        input_fingerprint = artifact_fingerprint(_embedding_inputs(scenes))
        config_fingerprint = artifact_fingerprint(
            {
                "backend": backend,
                "model": model,
                "batch_size": context.settings.embedding_batch_size,
                "index_enabled": index_enabled,
                "schema": ARTIFACT_SCHEMA_VERSION,
            }
        )
        if (
            stage_cache_manifest_matches(
                cache_artifact,
                stage=self.name,
                artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
                input_fingerprint=input_fingerprint,
                config_fingerprint=config_fingerprint,
                artifacts=expected_artifacts,
            )
            and _cached_embeddings_valid(artifact, scenes, backend=backend, model=model)
            and (
                not index_enabled
                or _cached_index_valid(index_manifest, index_artifact, artifact, scenes)
            )
        ):
            return StageResult(
                stage=self.name,
                status=StageStatus.CACHED,
                artifacts=[context.database_path, *expected_artifacts, cache_artifact],
                message="Embeddings reused cached semantic features and index.",
            )

        provider = (
            self._provider(context) if configured_backend == "sentence-transformers" else None
        )
        if context.progress is not None:
            context.progress(0, 2, "Embeddings: encoding scene metadata")
        try:
            report, updated = embed_scenes(scenes, provider)
        finally:
            if provider is not None:
                provider.release()
        if context.progress is not None:
            context.progress(1, 2, "Embeddings: updating the local archive")
        repository.synchronize_scenes(updated)
        if index_enabled:
            manifest = build_semantic_index(
                report,
                index_path=index_artifact,
                manifest_path=index_manifest,
            )
            report = report.model_copy(
                update={
                    "index_path": index_artifact.resolve(),
                    "indexed_count": len(manifest.scene_ids),
                }
            )
        else:
            index_artifact.unlink(missing_ok=True)
            index_manifest.unlink(missing_ok=True)
        write_json_atomic(artifact, report)
        write_stage_cache_manifest(
            cache_artifact,
            stage=self.name,
            artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
            input_fingerprint=input_fingerprint,
            config_fingerprint=config_fingerprint,
            artifacts=expected_artifacts,
        )
        if context.progress is not None:
            context.progress(2, 2, "Embeddings: complete")
        return StageResult(
            stage=self.name,
            artifacts=[context.database_path, *expected_artifacts, cache_artifact],
            message=(
                f"Embeddings prepared {len(report.embeddings)} semantic vector(s); "
                f"FAISS index={'enabled' if index_enabled else 'disabled'}."
            ),
        )

    def _provider(self, context: ProjectContext) -> TextEmbeddingProvider:
        if self._provider_factory is not None:
            return self._provider_factory(context)
        return SentenceTransformerEmbeddingProvider(
            model=context.settings.embedding_model,
            cache_dir=context.settings.model_cache / "sentence-transformers",
            device=context.settings.device,
            allow_download=context.settings.allow_model_download,
            batch_size=context.settings.embedding_batch_size,
        )


def _cached_embeddings_valid(
    path: Path,
    scenes: list[Scene],
    *,
    backend: str,
    model: str | None,
) -> bool:
    try:
        report = EmbeddingAnalysisReport.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError):
        return False
    return (
        report.backend == backend
        and report.model == model
        and {item.scene_id for item in report.embeddings} == {scene.id for scene in scenes}
        and all(len(item.vector) == report.dimensions for item in report.embeddings)
        and all(
            scene.metadata.get("embedding_backend") == backend
            and scene.metadata.get("embedding_model") == model
            for scene in scenes
        )
    )


def _cached_index_valid(
    manifest_path: Path,
    index_path: Path,
    report_path: Path,
    scenes: list[Scene],
) -> bool:
    try:
        manifest = SemanticIndexManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        report = EmbeddingAnalysisReport.model_validate_json(
            report_path.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError):
        return False
    report_scene_ids = [embedding.scene_id for embedding in report.embeddings]
    return (
        manifest.index_path.resolve() == index_path.resolve()
        and manifest.backend == report.backend
        and manifest.model == report.model
        and manifest.dimensions == report.dimensions
        and manifest.scene_ids == report_scene_ids
        and set(report_scene_ids) == {scene.id for scene in scenes}
        and manifest.source_fingerprint
        == artifact_fingerprint(
            report.backend,
            report.model,
            report.dimensions,
            report.embeddings,
        )
        and index_path.is_file()
        and index_path.stat().st_size > 0
    )


def _embedding_inputs(scenes: list[Scene]) -> list[dict[str, object]]:
    return [
        {
            "id": str(scene.id),
            "caption": scene.caption,
            "transcript": scene.transcript,
            "description": scene.metadata.get("detailed_description"),
            "location_type": scene.metadata.get("location_type"),
            "activity": scene.metadata.get("activity"),
            "emotion": scene.metadata.get("emotion"),
            "tags": scene.metadata.get("tags"),
        }
        for scene in sorted(scenes, key=lambda item: str(item.id))
    ]


def _index_enabled(mode: str) -> bool:
    if mode == "disabled":
        return False
    if mode == "faiss":
        return True
    return find_spec("faiss") is not None


def _remove_stale_artifacts(*paths: Path) -> None:
    for path in paths:
        path.unlink(missing_ok=True)
