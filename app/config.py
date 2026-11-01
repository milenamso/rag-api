from functools import lru_cache
from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Effort = Literal["low", "medium", "high", "xhigh", "max"]


class Settings(BaseSettings):
    """Application settings, read from environment variables prefixed with ``RAG_``.

    The Anthropic credential is read by the SDK itself from ``ANTHROPIC_API_KEY``.
    """

    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    app_name: str = "rag-api"
    log_level: str = "INFO"
    log_json: bool = True

    # Vector store
    chroma_mode: Literal["persistent", "http", "memory"] = "persistent"
    chroma_path: str = "./data/chroma"
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    documents_collection: str = "documents"
    cache_collection: str = "semantic_cache"

    # Ingestion
    embedding_provider: Literal["default", "hash"] = "default"
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)
    max_upload_mb: int = Field(default=20, gt=0)

    # Retrieval & generation
    top_k: int = Field(default=4, ge=1, le=20)
    llm_provider: Literal["anthropic", "extractive"] = "anthropic"
    anthropic_model: str = "claude-opus-5-5"
    anthropic_effort: Effort = "medium"
    anthropic_max_tokens: int = Field(default=16000, gt=0)
    anthropic_timeout_seconds: float = Field(default=60.0, gt=0)
    anthropic_server_fallback: bool = True

    # Semantic cache
    cache_enabled: bool = True
    cache_similarity_threshold: float = Field(default=0.92, ge=0.0, le=1.0)
    cache_ttl_seconds: int = Field(default=3600, gt=0)

    @model_validator(mode="after")
    def _check_overlap(self) -> Self:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
