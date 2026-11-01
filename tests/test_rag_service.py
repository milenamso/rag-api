import pytest

from app.schemas import QueryRequest
from app.services.parsing import UnsupportedFileTypeError
from app.services.rag import NO_CONTEXT_ANSWER, RagService
from tests.conftest import FakeLLM

DOC = b"""# Installation

Run pip install rag-api to install the project dependencies.

# Deployment

Use docker compose up to start the API together with ChromaDB.
"""


def test_ingest_indexes_chunks(rag: RagService):
    info = rag.ingest("guide.md", DOC)
    assert info.chunks >= 1
    assert info.content_type == "text/markdown"
    assert rag.store.count() == info.chunks


def test_ingest_rejects_unsupported_type(rag: RagService):
    with pytest.raises(UnsupportedFileTypeError):
        rag.ingest("photo.png", b"\x89PNG")


def test_query_without_documents_skips_llm(rag: RagService, fake_llm: FakeLLM):
    response = rag.query(QueryRequest(question="How do I install it?"))
    assert response.answer == NO_CONTEXT_ANSWER
    assert response.sources == []
    assert fake_llm.calls == []


def test_query_returns_llm_answer_with_sources(rag: RagService, fake_llm: FakeLLM):
    rag.ingest("guide.md", DOC)
    response = rag.query(QueryRequest(question="How do I install the project dependencies?"))
    assert response.answer.startswith("Answer to:")
    assert response.model == "fake-model"
    assert not response.cached
    assert not response.fallback
    assert response.sources[0].filename == "guide.md"
    assert "pip install" in fake_llm.calls[0][1][0].text


def test_second_similar_query_is_served_from_cache(rag: RagService, fake_llm: FakeLLM):
    rag.ingest("guide.md", DOC)
    rag.query(QueryRequest(question="How do I install the project dependencies?"))
    second = rag.query(QueryRequest(question="how do I install the project dependencies"))
    assert second.cached
    assert len(fake_llm.calls) == 1


def test_use_cache_false_bypasses_cache(rag: RagService, fake_llm: FakeLLM):
    rag.ingest("guide.md", DOC)
    question = "How do I install the project dependencies?"
    rag.query(QueryRequest(question=question))
    response = rag.query(QueryRequest(question=question, use_cache=False))
    assert not response.cached
    assert len(fake_llm.calls) == 2


def test_new_document_invalidates_cache(rag: RagService, fake_llm: FakeLLM):
    rag.ingest("guide.md", DOC)
    question = "How do I install the project dependencies?"
    rag.query(QueryRequest(question=question))
    rag.ingest("other.md", b"Unrelated content about databases.")
    assert not rag.query(QueryRequest(question=question)).cached


def test_llm_failure_falls_back_to_extractive_answer(rag: RagService, fake_llm: FakeLLM):
    fake_llm.fail = True
    rag.ingest("guide.md", DOC)
    question = "How do I install the project dependencies?"
    response = rag.query(QueryRequest(question=question))
    assert response.fallback
    assert response.model == "extractive"
    assert "pip install" in response.answer
    # Degraded answers are not cached, so the LLM is retried next time.
    fake_llm.fail = False
    retry = rag.query(QueryRequest(question=question))
    assert not retry.cached
    assert not retry.fallback


def test_delete_document(rag: RagService):
    info = rag.ingest("guide.md", DOC)
    assert rag.delete(info.id)
    assert not rag.delete(info.id)
    assert rag.store.count() == 0
