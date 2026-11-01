import hashlib
import json
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from chromadb.api import ClientAPI

from app.schemas import Source
from app.services.vector_store import get_cosine_collection


@dataclass(frozen=True, slots=True)
class CachedAnswer:
    question: str
    answer: str
    sources: list[Source]
    model: str
    similarity: float


def scope_key(top_k: int, document_ids: Sequence[str] | None) -> str:
    """Answers are only reusable for the same retrieval parameters."""
    raw = json.dumps({"top_k": top_k, "docs": sorted(document_ids or [])})
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class SemanticCache:
    """Reuses answers for questions whose embeddings are close to a previous one.

    "Como instalo o projeto?" and "Como faço para instalar o projeto?" map to
    nearby vectors, so the second question is served without calling the LLM.
    """

    def __init__(
        self,
        client: ClientAPI,
        collection_name: str,
        similarity_threshold: float,
        ttl_seconds: int,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._client = client
        self._name = collection_name
        self._collection = get_cosine_collection(client, collection_name)
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self._clock = clock

    def lookup(self, embedding: Sequence[float], scope: str) -> CachedAnswer | None:
        if self._collection.count() == 0:
            return None
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=1,
            where={"scope": scope},
            include=["documents", "metadatas", "distances"],
        )
        if not result["ids"][0]:
            return None

        entry_id = result["ids"][0][0]
        meta = (result["metadatas"] or [[]])[0][0]
        similarity = 1.0 - (result["distances"] or [[]])[0][0]

        if self._clock() - float(str(meta["created_at"])) > self.ttl_seconds:
            self._collection.delete(ids=[entry_id])
            return None
        if similarity < self.similarity_threshold:
            return None

        return CachedAnswer(
            question=(result["documents"] or [[]])[0][0] or "",
            answer=str(meta["answer"]),
            sources=[Source.model_validate(s) for s in json.loads(str(meta["sources"]))],
            model=str(meta["model"]),
            similarity=similarity,
        )

    def store(
        self,
        question: str,
        embedding: Sequence[float],
        answer: str,
        sources: Sequence[Source],
        model: str,
        scope: str,
    ) -> None:
        self._collection.add(
            ids=[uuid.uuid4().hex],
            documents=[question],
            embeddings=[embedding],
            metadatas=[
                {
                    "answer": answer,
                    "sources": json.dumps([s.model_dump() for s in sources]),
                    "model": model,
                    "scope": scope,
                    "created_at": self._clock(),
                }
            ],
        )

    def clear(self) -> None:
        self._client.delete_collection(self._name)
        self._collection = get_cosine_collection(self._client, self._name)

    def count(self) -> int:
        return self._collection.count()
