import math

import numpy as np
import pytest

from app.services.embeddings import HashEmbedder, mean_pool


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_vectors_are_deterministic_and_normalized():
    embedder = HashEmbedder(dimensions=64)
    [first] = embedder.embed(["the quick brown fox"])
    [second] = embedder.embed(["the quick brown fox"])
    assert first == second
    assert len(first) == 64
    assert math.isclose(math.sqrt(sum(x * x for x in first)), 1.0)


def test_similar_texts_are_closer_than_unrelated_ones():
    embedder = HashEmbedder()
    base, similar, unrelated = embedder.embed(
        [
            "how do I install the project dependencies",
            "how to install project dependencies",
            "quarterly revenue grew in europe",
        ]
    )
    assert cosine(base, similar) > cosine(base, unrelated)


def test_empty_text_yields_valid_unit_vector():
    [vector] = HashEmbedder(dimensions=8).embed([""])
    assert vector == [1.0] + [0.0] * 7


def test_invalid_dimensions():
    with pytest.raises(ValueError, match="dimensions"):
        HashEmbedder(dimensions=0)


def test_mean_pool_ignores_padding_and_normalizes():
    hidden = np.array([[[3.0, 4.0], [100.0, 100.0]]], dtype=np.float32)
    mask = np.array([[1, 0]], dtype=np.int64)
    [vector] = mean_pool(hidden, mask)
    assert vector == pytest.approx([0.6, 0.8])
