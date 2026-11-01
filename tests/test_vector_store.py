from datetime import UTC, datetime

from app.services.embeddings import HashEmbedder
from app.services.vector_store import VectorStore


def index(store: VectorStore, embedder: HashEmbedder, doc_id: str, chunks: list[str]) -> None:
    store.add_chunks(
        doc_id,
        chunks,
        embedder.embed(chunks),
        metadata={
            "filename": f"{doc_id}.md",
            "content_type": "text/markdown",
            "total_chunks": len(chunks),
            "characters": sum(map(len, chunks)),
            "uploaded_at": datetime.now(UTC).isoformat(),
        },
    )


def test_search_returns_most_similar_chunk_first(store: VectorStore, embedder: HashEmbedder):
    index(store, embedder, "a", ["python fastapi web framework", "cooking pasta with tomatoes"])
    [query] = embedder.embed(["fastapi python framework"])
    hits = store.search(query, top_k=2)
    assert hits[0].text == "python fastapi web framework"
    assert hits[0].score > hits[1].score
    assert hits[0].metadata["document_id"] == "a"


def test_search_on_empty_store_returns_nothing(store: VectorStore, embedder: HashEmbedder):
    assert store.search(embedder.embed(["anything"])[0], top_k=3) == []


def test_search_can_filter_by_document(store: VectorStore, embedder: HashEmbedder):
    index(store, embedder, "a", ["vector databases store embeddings"])
    index(store, embedder, "b", ["vector databases index embeddings"])
    [query] = embedder.embed(["vector databases"])
    hits = store.search(query, top_k=5, document_ids=["b"])
    assert {h.metadata["document_id"] for h in hits} == {"b"}


def test_list_and_delete_documents(store: VectorStore, embedder: HashEmbedder):
    index(store, embedder, "a", ["one", "two"])
    index(store, embedder, "b", ["three"])
    docs = {d.id: d for d in store.list_documents()}
    assert docs["a"].chunks == 2
    assert docs["b"].filename == "b.md"

    assert store.delete_document("a") == 2
    assert store.delete_document("a") == 0
    assert [d.id for d in store.list_documents()] == ["b"]
    assert store.count() == 1


def test_store_reports_healthy(store: VectorStore):
    assert store.is_healthy()
