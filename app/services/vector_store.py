from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import chromadb
from chromadb.api import ClientAPI

from app.config import Settings
from app.schemas import DocumentInfo

Metadata = Mapping[str, str | int | float | bool]


@dataclass(frozen=True, slots=True)
class SearchHit:
    id: str
    text: str
    metadata: dict[str, Any]
    score: float  # cosine similarity in [-1, 1]; higher is closer


def build_chroma_client(settings: Settings) -> ClientAPI:
    if settings.chroma_mode == "http":
        return chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
    if settings.chroma_mode == "memory":
        return chromadb.EphemeralClient()
    return chromadb.PersistentClient(path=settings.chroma_path)


def get_cosine_collection(client: ClientAPI, name: str) -> chromadb.Collection:
    # We compute embeddings ourselves, so Chroma must not attach its own function.
    return client.get_or_create_collection(
        name=name, embedding_function=None, metadata={"hnsw:space": "cosine"}
    )


class VectorStore:
    """Document chunks stored in a ChromaDB collection, searchable by cosine similarity."""

    def __init__(self, client: ClientAPI, collection_name: str) -> None:
        self._client = client
        self._collection = get_cosine_collection(client, collection_name)

    def add_chunks(
        self,
        document_id: str,
        chunks: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadata: Metadata,
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunks:
            return
        self._collection.add(
            ids=[f"{document_id}:{i}" for i in range(len(chunks))],
            documents=list(chunks),
            embeddings=list(embeddings),
            metadatas=[
                {**metadata, "document_id": document_id, "chunk_index": i}
                for i in range(len(chunks))
            ],
        )

    def search(
        self,
        embedding: Sequence[float],
        top_k: int,
        document_ids: Sequence[str] | None = None,
    ) -> list[SearchHit]:
        if self.count() == 0:
            return []
        where: dict[str, Any] | None = None
        if document_ids:
            where = {"document_id": {"$in": list(document_ids)}}
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        ids = result["ids"][0]
        documents = (result["documents"] or [[]])[0]
        metadatas = (result["metadatas"] or [[]])[0]
        distances = (result["distances"] or [[]])[0]
        return [
            SearchHit(id=id_, text=doc or "", metadata=dict(meta or {}), score=1.0 - dist)
            for id_, doc, meta, dist in zip(ids, documents, metadatas, distances, strict=True)
        ]

    def list_documents(self) -> list[DocumentInfo]:
        result = self._collection.get(include=["metadatas"])
        documents: dict[str, DocumentInfo] = {}
        for meta in result["metadatas"] or []:
            doc_id = str(meta["document_id"])
            if doc_id not in documents:
                documents[doc_id] = DocumentInfo(
                    id=doc_id,
                    filename=str(meta["filename"]),
                    content_type=str(meta["content_type"]),
                    chunks=int(str(meta["total_chunks"])),
                    characters=int(str(meta["characters"])),
                    uploaded_at=datetime.fromisoformat(str(meta["uploaded_at"])),
                )
        return sorted(documents.values(), key=lambda d: d.uploaded_at, reverse=True)

    def delete_document(self, document_id: str) -> int:
        existing = self._collection.get(where={"document_id": document_id}, include=[])
        if existing["ids"]:
            self._collection.delete(ids=existing["ids"])
        return len(existing["ids"])

    def count(self) -> int:
        return self._collection.count()

    def is_healthy(self) -> bool:
        try:
            self._client.heartbeat()
        except Exception:
            return False
        return True
