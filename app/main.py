import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse

from app import __version__
from app.api.routes import router
from app.config import Settings, get_settings
from app.container import Container, build_container
from app.logging import configure_logging, get_logger
from app.services.parsing import DocumentParsingError, UnsupportedFileTypeError

logger = get_logger(__name__)


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    """Application factory. Tests inject a pre-built container with fakes."""
    settings = settings or (container.settings if container else get_settings())
    configure_logging(settings.log_level, settings.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.container = container or build_container(settings)
        logger.info(
            "startup",
            chroma_mode=settings.chroma_mode,
            llm_provider=app.state.container.rag.llm.name,
            cache_enabled=settings.cache_enabled,
        )
        yield
        logger.info("shutdown")

    app = FastAPI(
        title="RAG API",
        version=__version__,
        description="Upload PDFs/Markdown, index them in ChromaDB and ask grounded questions.",
        lifespan=lifespan,
    )
    app.include_router(router)

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("unhandled_error", method=request.method, path=request.url.path)
            response = JSONResponse(
                {"detail": "Internal server error"},
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        response.headers["x-request-id"] = request_id
        logger.info(
            "http_request",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return response

    @app.exception_handler(UnsupportedFileTypeError)
    async def unsupported_type(_: Request, exc: UnsupportedFileTypeError) -> JSONResponse:
        return JSONResponse(
            {"detail": str(exc)}, status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
        )

    @app.exception_handler(DocumentParsingError)
    async def parsing_error(_: Request, exc: DocumentParsingError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=status.HTTP_422_UNPROCESSABLE_CONTENT)

    return app


app = create_app()
