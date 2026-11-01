# syntax=docker/dockerfile:1

# ---- build stage: install dependencies into an isolated virtualenv ----
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build
COPY pyproject.toml README.md ./
COPY app ./app
RUN pip install .

# ---- runtime stage: slim image, non-root user ----
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    HOME=/home/app

RUN useradd --create-home --uid 1000 app
COPY --from=builder /opt/venv /opt/venv

WORKDIR /srv
COPY --chown=app:app app ./app
RUN mkdir -p /srv/data && chown app:app /srv/data
USER app

# Bake the local embedding model (~80 MB) into the image so the first upload
# does not have to download it. Disable with --build-arg PRELOAD_EMBEDDINGS=false.
ARG PRELOAD_EMBEDDINGS=true
RUN if [ "$PRELOAD_EMBEDDINGS" = "true" ]; then \
      python -c "from app.services.embeddings import MiniLMEmbedder; MiniLMEmbedder().embed(['warmup'])"; \
    fi

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(urllib.request.urlopen('http://localhost:8080/health').status != 200)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
