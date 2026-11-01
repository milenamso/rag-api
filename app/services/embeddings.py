import hashlib
import math
import re
from collections.abc import Sequence
from typing import Protocol

_TOKEN = re.compile(r"\w+", re.UNICODE)


class Embedder(Protocol):
    """Turns texts into fixed-size vectors. Implementations must be deterministic."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class ChromaDefaultEmbedder:
    """Local ONNX ``all-MiniLM-L6-v2`` model shipped with ChromaDB (no API calls).

    The model (~80 MB) is downloaded on first use and cached on disk.
    """

    def __init__(self) -> None:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        self._fn = DefaultEmbeddingFunction()

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._fn(list(texts))
        return [[float(x) for x in vector] for vector in vectors]


class HashEmbedder:
    """Dependency-free bag-of-words embedder using the hashing trick.

    Lexically similar texts get similar vectors, which is enough for tests and
    offline demos. Not a substitute for a real semantic model in production.
    """

    def __init__(self, dimensions: int = 256) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in _TOKEN.findall(text.lower()):
            digest = hashlib.md5(token.encode(), usedforsecurity=False).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(x * x for x in vector))
        if norm == 0:
            # Chroma rejects zero vectors under cosine distance; use a fixed unit vector.
            vector[0] = 1.0
            return vector
        return [x / norm for x in vector]


def build_embedder(provider: str) -> Embedder:
    if provider == "hash":
        return HashEmbedder()
    return ChromaDefaultEmbedder()
