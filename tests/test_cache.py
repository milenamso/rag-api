from chromadb.api import ClientAPI

from app.schemas import Source
from app.services.cache import SemanticCache, scope_key
from app.services.embeddings import HashEmbedder

SOURCE = Source(document_id="d1", filename="a.md", chunk_index=0, score=0.9, text="ctx")


class FakeClock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


def test_hit_for_semantically_equivalent_question(cache: SemanticCache, embedder: HashEmbedder):
    scope = scope_key(3, None)
    [stored] = embedder.embed(["how do I install the project"])
    cache.store("how do I install the project", stored, "pip install", [SOURCE], "m", scope)

    [same] = embedder.embed(["How do I install the project?"])
    hit = cache.lookup(same, scope)
    assert hit is not None
    assert hit.answer == "pip install"
    assert hit.sources == [SOURCE]
    assert hit.similarity >= cache.similarity_threshold


def test_miss_for_different_question(cache: SemanticCache, embedder: HashEmbedder):
    scope = scope_key(3, None)
    cache.store("install steps", embedder.embed(["install steps"])[0], "a", [], "m", scope)
    assert cache.lookup(embedder.embed(["revenue in europe last quarter"])[0], scope) is None


def test_miss_for_different_scope(cache: SemanticCache, embedder: HashEmbedder):
    [vector] = embedder.embed(["install steps"])
    cache.store("install steps", vector, "a", [], "m", scope_key(3, None))
    assert cache.lookup(vector, scope_key(3, ["doc-1"])) is None
    assert cache.lookup(vector, scope_key(5, None)) is None


def test_expired_entries_are_evicted(
    chroma_client: ClientAPI, collection_name: str, embedder: HashEmbedder
):
    clock = FakeClock()
    cache = SemanticCache(
        chroma_client, collection_name, similarity_threshold=0.9, ttl_seconds=10, clock=clock
    )
    [vector] = embedder.embed(["question"])
    cache.store("question", vector, "answer", [], "m", "s")
    assert cache.lookup(vector, "s") is not None

    clock.now += 11
    assert cache.lookup(vector, "s") is None
    assert cache.count() == 0


def test_clear_removes_everything(cache: SemanticCache, embedder: HashEmbedder):
    cache.store("q", embedder.embed(["q"])[0], "a", [], "m", "s")
    cache.clear()
    assert cache.count() == 0


def test_scope_key_ignores_document_order():
    assert scope_key(3, ["a", "b"]) == scope_key(3, ["b", "a"])
