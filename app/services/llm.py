from collections.abc import Sequence
from dataclasses import dataclass
from html import escape
from typing import Protocol

import anthropic
from anthropic.types import Message, MessageParam
from anthropic.types.beta import BetaMessage, BetaMessageParam

from app.config import Effort
from app.services.vector_store import SearchHit

SYSTEM_PROMPT = """\
You answer questions using only the documents provided in the user message.

- Base every statement on the documents. If they do not contain the answer, say so plainly \
instead of guessing.
- Cite the documents you used with their index in square brackets, e.g. [1] or [2][3].
- Answer in the same language as the question, concisely and in plain prose."""

SERVER_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMError(RuntimeError):
    """The LLM could not produce a usable answer (network, API, refusal, empty)."""


@dataclass(frozen=True, slots=True)
class LLMAnswer:
    text: str
    model: str


class LLMProvider(Protocol):
    name: str

    def generate(self, question: str, contexts: Sequence[SearchHit]) -> LLMAnswer: ...


def build_user_prompt(question: str, contexts: Sequence[SearchHit]) -> str:
    documents = "\n".join(
        f'<document index="{i}" source="{escape(str(hit.metadata.get("filename", "")))}">\n'
        f"{hit.text}\n</document>"
        for i, hit in enumerate(contexts, start=1)
    )
    return f"<documents>\n{documents}\n</documents>\n\nQuestion: {question}"


class AnthropicProvider:
    """Generates grounded answers with Claude via the official Anthropic SDK."""

    name = "anthropic"

    def __init__(
        self,
        model: str,
        max_tokens: int,
        effort: Effort,
        timeout_seconds: float,
        server_fallback: bool = True,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.server_fallback = server_fallback
        self._client = client or anthropic.Anthropic(timeout=timeout_seconds, max_retries=2)

    def generate(self, question: str, contexts: Sequence[SearchHit]) -> LLMAnswer:
        prompt = build_user_prompt(question, contexts)
        response: BetaMessage | Message
        try:
            if self.server_fallback:
                # On a safety refusal, the API transparently retries on a fallback model.
                beta_messages: list[BetaMessageParam] = [{"role": "user", "content": prompt}]
                response = self._client.beta.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=SYSTEM_PROMPT,
                    messages=beta_messages,
                    output_config={"effort": self.effort},
                    betas=[SERVER_FALLBACK_BETA],
                    fallbacks="default",
                )
            else:
                messages: list[MessageParam] = [{"role": "user", "content": prompt}]
                response = self._client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=SYSTEM_PROMPT,
                    messages=messages,
                    output_config={"effort": self.effort},
                )
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"Could not reach the Anthropic API: {exc}") from exc

        if response.stop_reason == "refusal":
            raise LLMError("The model declined to answer")
        text = "".join(block.text for block in response.content if block.type == "text").strip()
        if not text:
            raise LLMError(f"Empty response (stop_reason={response.stop_reason})")
        return LLMAnswer(text=text, model=response.model)


class ExtractiveProvider:
    """LLM-free answer built from the retrieved passages.

    Used as the fallback when the LLM is unavailable, and as a provider for
    running the API fully offline.
    """

    name = "extractive"

    def __init__(self, max_passages: int = 3, max_chars: int = 500) -> None:
        self.max_passages = max_passages
        self.max_chars = max_chars

    def generate(self, question: str, contexts: Sequence[SearchHit]) -> LLMAnswer:
        if not contexts:
            return LLMAnswer(text="No relevant passages were found.", model=self.name)
        passages = []
        for i, hit in enumerate(contexts[: self.max_passages], start=1):
            snippet = hit.text if len(hit.text) <= self.max_chars else hit.text[: self.max_chars]
            passages.append(f"[{i}] {snippet.strip()}")
        text = "Most relevant passages from the indexed documents:\n\n" + "\n\n".join(passages)
        return LLMAnswer(text=text, model=self.name)
