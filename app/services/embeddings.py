import hashlib
import math
import re
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

_TOKEN = re.compile(r"\w+", re.UNICODE)


class Embedder(Protocol):
    """Turns texts into fixed-size vectors. Implementations must be deterministic."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


def mean_pool(hidden: npt.NDArray[np.float32], mask: npt.NDArray[np.int64]) -> list[list[float]]:
    """Average token vectors, ignoring padding, then L2-normalize (sentence-transformers style)."""
    weights = mask[..., np.newaxis].astype(np.float32)
    pooled = (hidden * weights).sum(axis=1) / np.clip(weights.sum(axis=1), 1e-9, None)
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    return [[float(x) for x in row] for row in pooled / np.clip(norms, 1e-12, None)]


class MiniLMEmbedder:
    """Local ``all-MiniLM-L6-v2`` on ONNX Runtime (no API calls).

    Reuses the model files ChromaDB downloads (~80 MB, cached on disk), but pads each batch
    only to its longest text. Chroma's ``DefaultEmbeddingFunction`` pads every input to 256
    tokens, so a 15-token question costs as much as a full chunk. The vectors are the same
    (see ``benchmarks/``), so existing indexes stay valid.
    """

    def __init__(self, batch_size: int = 32) -> None:
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

        self._fn = ONNXMiniLM_L6_V2(preferred_providers=["CPUExecutionProvider"])
        self._batch_size = batch_size
        self._tokenizer: Any = None

    def _load_tokenizer(self) -> Any:
        if self._tokenizer is None:
            self._fn(["warm-up"])  # downloads the model on first use
            tokenizer = self._fn.tokenizer.__class__.from_str(self._fn.tokenizer.to_str())
            tokenizer.no_padding()
            tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")  # pad to the batch's longest
            self._tokenizer = tokenizer
        return self._tokenizer

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        tokenizer = self._load_tokenizer()
        # Batching texts of similar length keeps padding to a minimum.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        vectors: list[list[float]] = [[] for _ in texts]
        for start in range(0, len(order), self._batch_size):
            batch = order[start : start + self._batch_size]
            encoded = tokenizer.encode_batch([texts[i] for i in batch])
            ids = np.array([e.ids for e in encoded], dtype=np.int64)
            mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
            hidden = self._fn.model.run(
                None,
                {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)},
            )[0]
            for i, vector in zip(batch, mean_pool(hidden, mask), strict=True):
                vectors[i] = vector
        return vectors


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
    return MiniLMEmbedder()
