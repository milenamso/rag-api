from itertools import pairwise

import pytest

from app.services.chunking import split_text


def test_short_text_is_a_single_chunk():
    assert split_text("Hello world.", chunk_size=100, chunk_overlap=10) == ["Hello world."]


def test_empty_text_produces_no_chunks():
    assert split_text("   \n\n  ", chunk_size=100, chunk_overlap=10) == []


def test_chunks_respect_max_size():
    text = " ".join(f"word{i}" for i in range(500))
    chunks = split_text(text, chunk_size=120, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(len(chunk) <= 120 for chunk in chunks)


def test_chunks_prefer_paragraph_boundaries():
    first = "A" * 70 + "."
    second = "B" * 70 + "."
    chunks = split_text(f"{first}\n\n{second}", chunk_size=100, chunk_overlap=10)
    assert chunks[0] == first
    assert chunks[-1].endswith(second)


def test_consecutive_chunks_overlap():
    text = " ".join(f"token{i:03d}" for i in range(200))
    chunks = split_text(text, chunk_size=100, chunk_overlap=30)
    for previous, current in pairwise(chunks):
        assert previous.split()[-1] in current


def test_all_content_is_preserved():
    words = [f"w{i}" for i in range(300)]
    chunks = split_text(" ".join(words), chunk_size=80, chunk_overlap=15)
    assert set(" ".join(chunks).split()) == set(words)


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (100, 100), (100, -1)])
def test_invalid_parameters_raise(size: int, overlap: int):
    with pytest.raises(ValueError, match="chunk_"):
        split_text("text", chunk_size=size, chunk_overlap=overlap)
