from fastapi.testclient import TestClient

from app.config import Settings
from tests.conftest import make_pdf

MARKDOWN = b"# FastAPI\n\nFastAPI is a modern Python web framework for building APIs."


def upload(client: TestClient, name: str, data: bytes, content_type: str = "text/markdown"):
    return client.post("/documents", files={"file": (name, data, content_type)})


def test_health(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["vector_store"] == "up"
    assert body["llm_provider"] == "fake"


def test_request_id_is_propagated(client: TestClient):
    response = client.get("/health", headers={"x-request-id": "abc123"})
    assert response.headers["x-request-id"] == "abc123"


def test_upload_markdown_and_list(client: TestClient):
    response = upload(client, "fastapi.md", MARKDOWN)
    assert response.status_code == 201
    doc = response.json()
    assert doc["filename"] == "fastapi.md"
    assert doc["chunks"] >= 1

    listing = client.get("/documents").json()
    assert listing["total"] == 1
    assert listing["documents"][0]["id"] == doc["id"]


def test_upload_pdf(client: TestClient):
    response = upload(client, "paper.pdf", make_pdf("Vector search with Chroma"), "application/pdf")
    assert response.status_code == 201
    assert response.json()["content_type"] == "application/pdf"


def test_upload_unsupported_type(client: TestClient):
    response = upload(client, "image.png", b"\x89PNG", "image/png")
    assert response.status_code == 415
    assert "Unsupported" in response.json()["detail"]


def test_upload_empty_document(client: TestClient):
    assert upload(client, "empty.md", b"   ").status_code == 422


def test_upload_too_large(client: TestClient, settings: Settings):
    data = b"a" * (settings.max_upload_bytes + 1)
    assert upload(client, "big.txt", data, "text/plain").status_code == 413


def test_query_flow_with_cache(client: TestClient):
    upload(client, "fastapi.md", MARKDOWN)
    payload = {"question": "What is FastAPI?"}

    first = client.post("/query", json=payload)
    assert first.status_code == 200
    body = first.json()
    assert body["answer"] == "Answer to: What is FastAPI? [1]"
    assert body["sources"][0]["filename"] == "fastapi.md"
    assert body["cached"] is False

    second = client.post("/query", json=payload).json()
    assert second["cached"] is True

    assert client.delete("/cache").status_code == 204
    assert client.post("/query", json=payload).json()["cached"] is False


def test_query_validation(client: TestClient):
    assert client.post("/query", json={"question": "hi"}).status_code == 422
    assert client.post("/query", json={"question": "valid?", "top_k": 0}).status_code == 422
    assert client.post("/query", json={"question": "valid?", "extra": 1}).status_code == 422
    # Strict mode: no silent coercion from string to int.
    assert client.post("/query", json={"question": "valid?", "top_k": "3"}).status_code == 422


def test_delete_document(client: TestClient):
    doc_id = upload(client, "fastapi.md", MARKDOWN).json()["id"]
    assert client.delete(f"/documents/{doc_id}").status_code == 204
    assert client.delete(f"/documents/{doc_id}").status_code == 404
    assert client.get("/documents").json()["total"] == 0
