from dataclasses import dataclass

from fastapi import Request

from app.config import Settings
from app.services.cache import SemanticCache
from app.services.embeddings import build_embedder
from app.services.llm import AnthropicProvider, ExtractiveProvider, LLMProvider
from app.services.rag import RagService
from app.services.vector_store import VectorStore, build_chroma_client


@dataclass(frozen=True, slots=True)
class Container:
    settings: Settings
    rag: RagService


def build_container(settings: Settings) -> Container:
    client = build_chroma_client(settings)
    embedder = build_embedder(settings.embedding_provider)
    fallback = ExtractiveProvider()

    llm: LLMProvider
    if settings.llm_provider == "anthropic":
        llm = AnthropicProvider(
            model=settings.anthropic_model,
            max_tokens=settings.anthropic_max_tokens,
            effort=settings.anthropic_effort,
            timeout_seconds=settings.anthropic_timeout_seconds,
            server_fallback=settings.anthropic_server_fallback,
        )
    else:
        llm = fallback

    cache = (
        SemanticCache(
            client,
            settings.cache_collection,
            similarity_threshold=settings.cache_similarity_threshold,
            ttl_seconds=settings.cache_ttl_seconds,
        )
        if settings.cache_enabled
        else None
    )

    rag = RagService(
        store=VectorStore(client, settings.documents_collection),
        embedder=embedder,
        llm=llm,
        fallback_llm=fallback,
        cache=cache,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        default_top_k=settings.top_k,
    )
    return Container(settings=settings, rag=rag)


def get_container(request: Request) -> Container:
    container: Container = request.app.state.container
    return container


def get_rag_service(request: Request) -> RagService:
    return get_container(request).rag
