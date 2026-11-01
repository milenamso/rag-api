from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile, status

from app import __version__
from app.container import Container, get_container, get_rag_service
from app.schemas import (
    DocumentInfo,
    DocumentList,
    ErrorResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
)
from app.services.rag import RagService

router = APIRouter()

RagDep = Annotated[RagService, Depends(get_rag_service)]
ContainerDep = Annotated[Container, Depends(get_container)]


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health(container: ContainerDep) -> HealthResponse:
    store = container.rag.store
    healthy = store.is_healthy()
    return HealthResponse(
        status="ok" if healthy else "degraded",
        version=__version__,
        vector_store="up" if healthy else "down",
        indexed_chunks=store.count() if healthy else 0,
        llm_provider=container.rag.llm.name,
    )


@router.post(
    "/documents",
    response_model=DocumentInfo,
    status_code=status.HTTP_201_CREATED,
    tags=["documents"],
    responses={
        413: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
)
def upload_document(
    container: ContainerDep,
    file: Annotated[UploadFile, File(description="PDF, Markdown or plain-text file")],
) -> DocumentInfo:
    """Upload a document; it is parsed, chunked, embedded and indexed synchronously."""
    if not file.filename:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "File name is required")

    limit = container.settings.max_upload_bytes
    data = file.file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            f"File exceeds the {container.settings.max_upload_mb} MB limit",
        )
    return container.rag.ingest(file.filename, data)


@router.get("/documents", response_model=DocumentList, tags=["documents"])
def list_documents(rag: RagDep) -> DocumentList:
    documents = rag.store.list_documents()
    return DocumentList(documents=documents, total=len(documents))


@router.delete(
    "/documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["documents"],
    responses={404: {"model": ErrorResponse}},
)
def delete_document(document_id: str, rag: RagDep) -> Response:
    if not rag.delete(document_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Document '{document_id}' not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/query", response_model=QueryResponse, tags=["query"])
def query(request: QueryRequest, rag: RagDep) -> QueryResponse:
    """Answer a question grounded in the indexed documents."""
    return rag.query(request)


@router.delete("/cache", status_code=status.HTTP_204_NO_CONTENT, tags=["query"])
def clear_cache(rag: RagDep) -> Response:
    rag.clear_cache()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
