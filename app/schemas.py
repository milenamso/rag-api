from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    """Base for request bodies: no type coercion and no unknown fields."""

    model_config = ConfigDict(strict=True, extra="forbid")


class DocumentInfo(BaseModel):
    id: str
    filename: str
    content_type: str
    chunks: int = Field(ge=0)
    characters: int = Field(ge=0)
    uploaded_at: datetime


class DocumentList(BaseModel):
    documents: list[DocumentInfo]
    total: int


class QueryRequest(StrictModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    document_ids: list[str] | None = Field(default=None, min_length=1)
    use_cache: bool = True


class Source(BaseModel):
    document_id: str
    filename: str
    chunk_index: int
    score: float
    text: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    model: str
    cached: bool = False
    fallback: bool = False
    latency_ms: float


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    vector_store: Literal["up", "down"]
    indexed_chunks: int
    llm_provider: str


class ErrorResponse(BaseModel):
    detail: str
