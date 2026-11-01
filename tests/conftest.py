import uuid
from collections.abc import Iterator, Sequence

import chromadb
import pytest
from chromadb.api import ClientAPI
from fastapi.testclient import TestClient

from app.config import Settings
from app.container import Container
from app.main import create_app
from app.services.cache import SemanticCache
from app.services.embeddings import HashEmbedder
from app.services.llm import ExtractiveProvider, LLMAnswer, LLMError
from app.services.rag import RagService
from app.services.vector_store import SearchHit, VectorStore


class FakeLLM:
    """Records calls and returns a canned answer, or raises to simulate an outage."""

    name = "fake"

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[str, list[SearchHit]]] = []

    def generate(self, question: str, contexts: Sequence[SearchHit]) -> LLMAnswer:
        self.calls.append((question, list(contexts)))
        if self.fail:
            raise LLMError("simulated outage")
        return LLMAnswer(text=f"Answer to: {question} [1]", model="fake-model")


def make_pdf(text: str) -> bytes:
    """Build a minimal single-page PDF containing ``text``."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    return bytes(out)


@pytest.fixture
def chroma_client() -> ClientAPI:
    return chromadb.EphemeralClient()


@pytest.fixture
def collection_name() -> str:
    # Ephemeral clients share in-process state, so isolate each test by name.
    return f"test-{uuid.uuid4().hex}"


@pytest.fixture
def embedder() -> HashEmbedder:
    return HashEmbedder()


@pytest.fixture
def store(chroma_client: ClientAPI, collection_name: str) -> VectorStore:
    return VectorStore(chroma_client, f"{collection_name}-docs")


@pytest.fixture
def cache(chroma_client: ClientAPI, collection_name: str) -> SemanticCache:
    return SemanticCache(
        chroma_client, f"{collection_name}-cache", similarity_threshold=0.9, ttl_seconds=60
    )


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def rag(
    store: VectorStore, embedder: HashEmbedder, fake_llm: FakeLLM, cache: SemanticCache
) -> RagService:
    return RagService(
        store=store,
        embedder=embedder,
        llm=fake_llm,
        fallback_llm=ExtractiveProvider(),
        cache=cache,
        chunk_size=200,
        chunk_overlap=40,
        default_top_k=3,
    )


@pytest.fixture
def settings() -> Settings:
    return Settings(
        chroma_mode="memory",
        embedding_provider="hash",
        llm_provider="extractive",
        log_json=False,
        log_level="WARNING",
        max_upload_mb=1,
        chunk_size=200,
        chunk_overlap=40,
    )


@pytest.fixture
def client(settings: Settings, rag: RagService) -> Iterator[TestClient]:
    app = create_app(container=Container(settings=settings, rag=rag))
    with TestClient(app) as test_client:
        yield test_client
