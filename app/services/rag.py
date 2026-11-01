import time
import uuid
from datetime import UTC, datetime

from app.logging import get_logger
from app.schemas import DocumentInfo, QueryRequest, QueryResponse, Source
from app.services.cache import SemanticCache, scope_key
from app.services.chunking import split_text
from app.services.embeddings import Embedder
from app.services.llm import LLMError, LLMProvider
from app.services.parsing import DocumentParsingError, detect_content_type, parse_document
from app.services.vector_store import SearchHit, VectorStore

logger = get_logger(__name__)

NO_CONTEXT_ANSWER = "No indexed documents are relevant to this question."


class RagService:
    """Orchestrates ingestion (parse -> chunk -> embed -> index) and querying
    (cache -> retrieve -> generate, with fallback)."""

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        llm: LLMProvider,
        fallback_llm: LLMProvider,
        cache: SemanticCache | None,
        chunk_size: int,
        chunk_overlap: int,
        default_top_k: int,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.llm = llm
        self.fallback_llm = fallback_llm
        self.cache = cache
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.default_top_k = default_top_k

    def ingest(self, filename: str, data: bytes) -> DocumentInfo:
        content_type = detect_content_type(filename)
        text = parse_document(filename, data)
        chunks = split_text(text, self.chunk_size, self.chunk_overlap)
        if not chunks:
            raise DocumentParsingError(f"No extractable text found in '{filename}'")

        info = DocumentInfo(
            id=uuid.uuid4().hex,
            filename=filename,
            content_type=content_type,
            chunks=len(chunks),
            characters=len(text),
            uploaded_at=datetime.now(UTC),
        )
        started = time.perf_counter()
        embeddings = self.embedder.embed(chunks)
        self.store.add_chunks(
            info.id,
            chunks,
            embeddings,
            metadata={
                "filename": info.filename,
                "content_type": info.content_type,
                "total_chunks": info.chunks,
                "characters": info.characters,
                "uploaded_at": info.uploaded_at.isoformat(),
            },
        )
        self._invalidate_cache("document_indexed")
        logger.info(
            "document_indexed",
            document_id=info.id,
            filename=filename,
            chunks=info.chunks,
            characters=info.characters,
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return info

    def delete(self, document_id: str) -> bool:
        removed = self.store.delete_document(document_id)
        if removed:
            self._invalidate_cache("document_deleted")
            logger.info("document_deleted", document_id=document_id, chunks=removed)
        return removed > 0

    def query(self, request: QueryRequest) -> QueryResponse:
        started = time.perf_counter()
        top_k = request.top_k or self.default_top_k
        scope = scope_key(top_k, request.document_ids)
        [embedding] = self.embedder.embed([request.question])

        def elapsed_ms() -> float:
            return round((time.perf_counter() - started) * 1000, 1)

        if self.cache is not None and request.use_cache:
            cached = self.cache.lookup(embedding, scope)
            if cached is not None:
                logger.info(
                    "cache_hit", similarity=round(cached.similarity, 4), matched=cached.question
                )
                return QueryResponse(
                    answer=cached.answer,
                    sources=cached.sources,
                    model=cached.model,
                    cached=True,
                    latency_ms=elapsed_ms(),
                )

        hits = self.store.search(embedding, top_k, request.document_ids)
        if not hits:
            logger.info("no_context_found", top_k=top_k)
            return QueryResponse(
                answer=NO_CONTEXT_ANSWER, sources=[], model="none", latency_ms=elapsed_ms()
            )

        sources = [_to_source(hit) for hit in hits]
        fallback = False
        try:
            answer = self.llm.generate(request.question, hits)
        except LLMError as exc:
            logger.warning(
                "llm_failed_using_fallback",
                provider=self.llm.name,
                fallback_provider=self.fallback_llm.name,
                error=str(exc),
            )
            answer = self.fallback_llm.generate(request.question, hits)
            fallback = True

        # Never cache degraded answers: the next request should retry the LLM.
        if self.cache is not None and request.use_cache and not fallback:
            self.cache.store(request.question, embedding, answer.text, sources, answer.model, scope)

        response = QueryResponse(
            answer=answer.text,
            sources=sources,
            model=answer.model,
            fallback=fallback,
            latency_ms=elapsed_ms(),
        )
        logger.info(
            "query_answered",
            model=answer.model,
            fallback=fallback,
            sources=len(sources),
            top_score=round(hits[0].score, 4),
            latency_ms=response.latency_ms,
        )
        return response

    def clear_cache(self) -> None:
        self._invalidate_cache("manual")

    def _invalidate_cache(self, reason: str) -> None:
        # The corpus changed, so previously cached answers may be stale.
        if self.cache is not None and self.cache.count():
            self.cache.clear()
            logger.info("cache_cleared", reason=reason)


def _to_source(hit: SearchHit) -> Source:
    return Source(
        document_id=str(hit.metadata["document_id"]),
        filename=str(hit.metadata["filename"]),
        chunk_index=int(hit.metadata["chunk_index"]),
        score=round(hit.score, 4),
        text=hit.text,
    )
