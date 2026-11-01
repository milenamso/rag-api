from types import SimpleNamespace
from typing import Any

import anthropic
import httpx
import pytest

from app.services.llm import (
    SERVER_FALLBACK_BETA,
    AnthropicProvider,
    ExtractiveProvider,
    LLMError,
    build_user_prompt,
)
from app.services.vector_store import SearchHit

HITS = [
    SearchHit(id="d:0", text="Chroma is a vector DB.", metadata={"filename": "a.md"}, score=0.9),
    SearchHit(id="d:1", text="FastAPI is a framework.", metadata={"filename": "b.md"}, score=0.8),
]


def message(text: str = "Answer [1]", stop_reason: str = "end_turn") -> SimpleNamespace:
    return SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text=text),
        ],
        stop_reason=stop_reason,
        model="claude-opus-5-5",
    )


class FakeMessages:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.kwargs: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def fake_client(result: Any) -> Any:
    messages = FakeMessages(result)
    return SimpleNamespace(messages=messages, beta=SimpleNamespace(messages=messages))


def provider(client: Any, server_fallback: bool = True) -> AnthropicProvider:
    return AnthropicProvider(
        model="claude-opus-5-5",
        max_tokens=1000,
        effort="medium",
        timeout_seconds=5,
        server_fallback=server_fallback,
        client=client,
    )


def test_prompt_numbers_documents_and_includes_question():
    prompt = build_user_prompt("What is Chroma?", HITS)
    assert '<document index="1" source="a.md">' in prompt
    assert '<document index="2" source="b.md">' in prompt
    assert prompt.endswith("Question: What is Chroma?")


def test_generate_returns_text_blocks_only():
    client = fake_client(message("Chroma stores vectors [1]."))
    answer = provider(client).generate("What is Chroma?", HITS)
    assert answer.text == "Chroma stores vectors [1]."
    assert answer.model == "claude-opus-5-5"


def test_generate_sends_server_side_fallback_options():
    client = fake_client(message())
    provider(client).generate("q", HITS)
    kwargs = client.beta.messages.kwargs
    assert kwargs["betas"] == [SERVER_FALLBACK_BETA]
    assert kwargs["fallbacks"] == "default"
    assert kwargs["output_config"] == {"effort": "medium"}
    assert kwargs["model"] == "claude-opus-5-5"


def test_generate_without_server_fallback_uses_standard_endpoint():
    client = fake_client(message())
    provider(client, server_fallback=False).generate("q", HITS)
    assert "betas" not in client.messages.kwargs


def test_refusal_raises_llm_error():
    with pytest.raises(LLMError, match="declined"):
        provider(fake_client(message(stop_reason="refusal"))).generate("q", HITS)


def test_empty_text_raises_llm_error():
    with pytest.raises(LLMError, match="Empty"):
        provider(fake_client(message(text="  "))).generate("q", HITS)


def test_api_errors_are_wrapped():
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    error = anthropic.APIConnectionError(request=request)
    with pytest.raises(LLMError, match="Could not reach"):
        provider(fake_client(error)).generate("q", HITS)


def test_extractive_provider_quotes_top_passages():
    answer = ExtractiveProvider(max_passages=1).generate("q", HITS)
    assert "[1] Chroma is a vector DB." in answer.text
    assert "FastAPI" not in answer.text
    assert answer.model == "extractive"


def test_extractive_provider_without_context():
    assert "No relevant" in ExtractiveProvider().generate("q", []).text
