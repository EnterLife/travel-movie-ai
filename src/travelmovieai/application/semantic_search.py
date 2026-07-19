"""Local semantic archive-search use case."""

from pathlib import Path

from pydantic import ValidationError

from travelmovieai.analysis.embeddings import EMBEDDING_BACKEND, feature_hash_embedding
from travelmovieai.application.context import ProjectContext
from travelmovieai.core.exceptions import PipelineStageError
from travelmovieai.domain.models import EmbeddingAnalysisReport, SemanticSearchReport
from travelmovieai.infrastructure.embeddings import SentenceTransformerEmbeddingProvider
from travelmovieai.infrastructure.semantic_index import search_semantic_index


def search_project_scenes(
    context: ProjectContext,
    query: str,
    *,
    limit: int = 10,
) -> SemanticSearchReport:
    normalized_query = " ".join(query.split())
    if not normalized_query:
        raise PipelineStageError("Semantic search query must not be empty.")
    if len(normalized_query) > 1000:
        raise PipelineStageError("Semantic search query must not exceed 1000 characters.")
    if limit < 1 or limit > 1000:
        raise PipelineStageError("Semantic search limit must be between 1 and 1000.")
    report_path = context.artifacts_dir / "embeddings.json"
    report = _read_embedding_report(report_path)
    index_path = context.artifacts_dir / "embeddings.faiss"
    manifest_path = context.artifacts_dir / "embeddings.index.json"
    if not index_path.is_file() or not manifest_path.is_file():
        raise PipelineStageError(
            "Semantic archive search needs a completed FAISS embeddings stage."
        )

    if report.backend == EMBEDDING_BACKEND:
        query_vector = feature_hash_embedding(normalized_query)
    elif report.backend == "sentence-transformers":
        if not report.model:
            raise PipelineStageError("Sentence embedding report is missing its model name.")
        provider = SentenceTransformerEmbeddingProvider(
            model=report.model,
            cache_dir=context.settings.model_cache / "sentence-transformers",
            device=context.settings.device,
            allow_download=context.settings.allow_model_download,
            batch_size=context.settings.embedding_batch_size,
        )
        try:
            encoded = provider.encode([normalized_query])
        finally:
            provider.release()
        query_vector = encoded[0]
    else:
        raise PipelineStageError(
            f"Semantic search does not support embedding backend {report.backend!r}."
        )
    hits = search_semantic_index(
        query_vector,
        index_path=index_path,
        manifest_path=manifest_path,
        limit=limit,
        expected_report=report,
    )
    return SemanticSearchReport(
        backend=report.backend,
        model=report.model,
        query=normalized_query,
        hits=hits,
    )


def _read_embedding_report(path: Path) -> EmbeddingAnalysisReport:
    try:
        return EmbeddingAnalysisReport.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as error:
        raise PipelineStageError("Could not read embeddings.json for semantic search.") from error
