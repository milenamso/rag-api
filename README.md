# RAG API

API de **Retrieval-Augmented Generation** construída com **FastAPI**, **ChromaDB** e **Claude**.
Faça upload de PDFs e Markdown, indexe-os em um banco vetorial local e faça perguntas
respondidas pelo LLM com base apenas nos seus documentos, com citação das fontes.

[![CI](https://github.com/milenamso/rag-api/actions/workflows/ci.yml/badge.svg)](https://github.com/milenamso/rag-api/actions/workflows/ci.yml)

## Destaques

- **Tipagem estrita**: Pydantic v2 com `strict=True` e `extra="forbid"` nos requests, configuração via `pydantic-settings` e `mypy --strict` no CI.
- **Banco vetorial open-source local**: ChromaDB em modo embarcado (pasta local), servidor HTTP (Docker) ou em memória (testes).
- **Embeddings locais**: modelo ONNX `all-MiniLM-L6-v2` com padding dinâmico (perguntas ~6× mais rápidas que o embedder padrão do Chroma, vetores idênticos), sem chamadas externas.
- **Cache semântico**: perguntas parecidas ("Como instalo?" / "Como faço a instalação?") reaproveitam a resposta anterior por similaridade de cosseno, com TTL, escopo por parâmetros de busca e invalidação automática quando o corpus muda.
- **Fallback resiliente**: se o LLM falhar (rede, rate limit, recusa, resposta vazia), a API devolve uma resposta extrativa com os trechos mais relevantes e marca `"fallback": true`. Respostas degradadas nunca vão para o cache.
- **Logging estruturado** (JSON via `structlog`) com `request_id` propagado em todos os eventos e no header `x-request-id`.
- **Benchmarks reproduzíveis** com métricas reais de recuperação, latência e cache ([resultados](benchmarks/results.md)).
- **Docker + docker-compose**, **pytest** (66 testes, sem rede) e **GitHub Actions** com ruff, mypy, testes e build da imagem.

## Arquitetura

```
              ┌──────────── POST /documents ────────────┐
 PDF / MD ──► │ parse ─► chunk (overlap) ─► embed ─► Chroma (documents)
              └─────────────────────────────────────────┘

              ┌──────────────── POST /query ───────────────────────────────┐
 pergunta ──► │ embed ─► cache semântico ──hit──► resposta (cached=true)    │
              │              │ miss                                         │
              │              ▼                                              │
              │        busca top-k ─► Claude ──erro──► fallback extrativo   │
              │                         │                                   │
              │                         ▼                                   │
              │                 resposta + fontes ─► grava no cache         │
              └─────────────────────────────────────────────────────────────┘
```

```
benchmarks/           # experimentos reproduzíveis (BEIR/SciFact + cache)
app/
├── main.py            # app factory, middleware de request id, handlers de erro
├── config.py          # Settings (pydantic-settings, prefixo RAG_)
├── container.py       # injeção de dependências
├── schemas.py         # modelos Pydantic de request/response
├── logging.py         # configuração do structlog
├── api/routes.py      # endpoints
└── services/
    ├── parsing.py     # PDF (pypdf) e Markdown/TXT
    ├── chunking.py    # divisão em chunks com sobreposição
    ├── embeddings.py  # MiniLM ONNX com padding dinâmico e HashEmbedder (testes)
    ├── vector_store.py# wrapper do ChromaDB
    ├── cache.py       # cache semântico
    ├── llm.py         # Claude (SDK oficial) e provedor extrativo
    └── rag.py         # orquestração de ingestão e consulta
```

## Endpoints

| Método | Rota | Descrição |
|---|---|---|
| `GET` | `/health` | Status da API e do banco vetorial |
| `POST` | `/documents` | Upload (multipart) de `.pdf`, `.md`, `.markdown` ou `.txt` |
| `GET` | `/documents` | Lista documentos indexados |
| `DELETE` | `/documents/{id}` | Remove um documento e seus chunks |
| `POST` | `/query` | Pergunta com resposta fundamentada e fontes |
| `DELETE` | `/cache` | Limpa o cache semântico |

Documentação interativa em `http://localhost:8080/docs`.

## Como rodar

### Com Docker (recomendado)

```bash
cp .env.example .env   # preencha ANTHROPIC_API_KEY
docker compose up --build
```

Sobe a API em `:8080` e o ChromaDB em `:8000`, com volume persistente.

### Localmente

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env
uvicorn app.main:app --reload --port 8080
```

Sem chave da API? Use `RAG_LLM_PROVIDER=extractive` para rodar 100% offline.

## Exemplos

```bash
curl -F "file=@docs/manual.pdf" http://localhost:8080/documents
```

```bash
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Como faço o deploy da aplicação?", "top_k": 4}'
```

Resposta:

```json
{
  "answer": "Use docker compose up para subir a API junto com o ChromaDB [1].",
  "sources": [
    {"document_id": "3f2a...", "filename": "manual.pdf", "chunk_index": 2, "score": 0.83, "text": "..."}
  ],
  "model": "claude-opus-5-5",
  "cached": false,
  "fallback": false,
  "latency_ms": 1840.2
}
```

Filtre por documentos específicos com `"document_ids": ["<id>"]` e ignore o cache com `"use_cache": false`.

## Benchmarks

Métricas reais, medidas com `python -m benchmarks.run_benchmarks` em um notebook
(Intel 12ª geração, 12 threads, só CPU). Dataset: [BEIR/SciFact](https://github.com/beir-cellar/beir),
300 perguntas de teste sobre 1.500 resumos científicos (todos os relevantes + distratores com seed fixa).
Tabelas completas em [`benchmarks/results.md`](benchmarks/results.md).

**Qualidade de recuperação vs. latência** (MiniLM, top-10 documentos):

| Chunk | Chunks | Recall@5 | nDCG@10 | Query p95 | Indexação |
|---|---|---|---|---|---|
| 400 caracteres | 8.433 | 0,826 | 0,768 | 62,9 ms | 26,9 chunks/s |
| **800 caracteres** | 4.071 | **0,851** | **0,781** | **15,3 ms** | 21,5 chunks/s |
| 1.600 caracteres | 2.131 | 0,841 | 0,764 | 15,3 ms | 15,0 chunks/s |
| 800 (hashing, sem modelo) | 4.071 | 0,346 | 0,291 | 3,2 ms | 1.175 chunks/s |

**Principais achados**

- **Chunks de 800 caracteres** foram o melhor ponto: chunks menores duplicam o índice e perdem contexto; maiores
  passam do limite de 256 tokens do MiniLM e são truncados.
- **Padding dinâmico**: o embedder padrão do Chroma completa todo texto até 256 tokens. Preenchendo só até o
  maior texto do lote, o embedding de uma pergunta caiu de **54,1 ms para 8,0 ms (p50)** e a indexação
  ficou 1,37× mais rápida, com vetores idênticos (cosseno mínimo 1,0).
- **Cache semântico tem falsos positivos perigosos**: no limiar 0,92, 8 de 100 perguntas receberam a resposta
  de uma pergunta com sentido **oposto** (ex.: *"Cold exposure reduces BAT recruitment"* × *"...increases..."*,
  similaridade 0,984). Embeddings de frase quase não distinguem negação.
- **Em português o MiniLM não separa paráfrases de perguntas diferentes** (similaridade média 0,69 × 0,72):
  o cache praticamente não acerta em PT. Um modelo multilíngue é o próximo passo.
- **HNSW só compensa em escala**: com 10 mil vetores, busca exata em NumPy é 2,6× mais rápida que o HNSW
  (0,87 × 2,26 ms); o HNSW passa à frente perto de 50 mil e fica 2× mais rápido com 100 mil (3,0 × 6,2 ms). Em dados sem estrutura (pior caso), o recall do HNSW
  com parâmetros padrão cai para 0,30.

## Configuração

Todas as variáveis usam o prefixo `RAG_` (veja [`.env.example`](.env.example) e [`app/config.py`](app/config.py)).

| Variável | Padrão | Descrição |
|---|---|---|
| `ANTHROPIC_API_KEY` | | Chave da API do Claude |
| `RAG_LLM_PROVIDER` | `anthropic` | `anthropic` ou `extractive` |
| `RAG_ANTHROPIC_MODEL` | `claude-opus-5-5` | Modelo do Claude |
| `RAG_ANTHROPIC_EFFORT` | `medium` | `low` … `max`, controla profundidade/custo |
| `RAG_ANTHROPIC_SERVER_FALLBACK` | `true` | Em caso de recusa do modelo, a API do Claude tenta outro modelo automaticamente |
| `RAG_CHROMA_MODE` | `persistent` | `persistent`, `http` ou `memory` |
| `RAG_EMBEDDING_PROVIDER` | `default` | `default` (ONNX MiniLM) ou `hash` |
| `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` | `800` / `120` | Tamanho e sobreposição dos chunks (caracteres; 800 teve o melhor nDCG nos benchmarks) |
| `RAG_TOP_K` | `4` | Chunks recuperados por pergunta |
| `RAG_CACHE_SIMILARITY_THRESHOLD` | `0.92` | Similaridade mínima para um hit de cache |
| `RAG_CACHE_TTL_SECONDS` | `3600` | Validade das entradas do cache |
| `RAG_LOG_JSON` | `true` | Logs em JSON (`false` para saída legível no terminal) |

## Logs estruturados

```json
{"event": "llm_failed_using_fallback", "provider": "anthropic", "fallback_provider": "extractive", "error": "Anthropic API error 529: Overloaded", "request_id": "9b1c...", "level": "warning", "timestamp": "2026-11-01T12:00:00Z"}
{"event": "cache_hit", "similarity": 0.9631, "matched": "Como faço o deploy?", "request_id": "a77e...", "level": "info", "timestamp": "..."}
```

## Qualidade

```bash
ruff check .
ruff format --check .
mypy app
pytest --cov=app
```

Para reproduzir os benchmarks (baixa o SciFact, ~3 MB, em `data/`):

```bash
python -m benchmarks.run_benchmarks          # ~20 min em CPU
python -m benchmarks.run_benchmarks --quick  # ~5 min
```

Os testes rodam sem rede: usam ChromaDB em memória, um embedder determinístico e um LLM falso.
O pipeline de CI (`.github/workflows/ci.yml`) executa tudo isso em cada push e pull request,
em Python 3.11 e 3.12, e valida o build da imagem Docker.

## Licença

MIT
