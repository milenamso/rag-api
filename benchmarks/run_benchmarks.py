"""Reproducible benchmarks for the RAG pipeline.

Experiments:
  1. Retrieval quality vs. latency on BEIR/SciFact (300 test queries), across embedders
     (MiniLM ONNX vs. hashing) and chunk sizes. The corpus is a reproducible subset: every
     document relevant to a test query plus seeded random distractors (``--docs``).
  2. Approximate search (Chroma HNSW) vs. exact brute-force search, on the real corpus and
     on synthetic collections of up to 100k vectors.
  3. Semantic cache: hit rate vs. false hits per similarity threshold (PT and EN),
     plus end-to-end latency of a cache hit vs. a cache miss.
  4. Embedding throughput: Chroma's fixed 256-token padding vs. dynamic padding.

Usage:
    python -m benchmarks.run_benchmarks                # full run (~20 min on a laptop CPU)
    python -m benchmarks.run_benchmarks --docs 5183    # whole SciFact corpus
    python -m benchmarks.run_benchmarks --quick        # 60 queries, fewer configs

Results are written to benchmarks/results.json and benchmarks/results.md.
"""

import argparse
import io
import json
import math
import os
import platform
import random
import statistics
import time
import uuid
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import chromadb
import httpx
import numpy as np
import numpy.typing as npt

from app.logging import configure_logging
from app.schemas import QueryRequest
from app.services.cache import SemanticCache
from app.services.chunking import split_text
from app.services.embeddings import Embedder, HashEmbedder, MiniLMEmbedder
from app.services.llm import ExtractiveProvider
from app.services.rag import RagService
from app.services.vector_store import VectorStore, get_cosine_collection
from benchmarks.paraphrases import DIFFERENT, PARAPHRASES

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT.parent / "data" / "benchmarks"
SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
BATCH = 256


# --------------------------------------------------------------------------- dataset


@dataclass(frozen=True, slots=True)
class Dataset:
    corpus: dict[str, str]  # doc_id -> title + abstract
    queries: dict[str, str]  # query_id -> text
    qrels: dict[str, set[str]]  # query_id -> relevant doc ids


def load_scifact() -> Dataset:
    folder = DATA_DIR / "scifact"
    if not (folder / "corpus.jsonl").exists():
        print(f"Downloading SciFact from {SCIFACT_URL} ...")
        response = httpx.get(SCIFACT_URL, follow_redirects=True, timeout=120)
        response.raise_for_status()
        payload = response.content
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            for member in archive.namelist():
                name = member.removeprefix("scifact/")
                if name in {"corpus.jsonl", "queries.jsonl", "qrels/test.tsv"}:
                    target = folder / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(archive.read(member))

    corpus: dict[str, str] = {}
    with (folder / "corpus.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            corpus[str(row["_id"])] = f"{row['title']}. {row['text']}".strip()

    qrels: dict[str, set[str]] = {}
    with (folder / "qrels" / "test.tsv").open(encoding="utf-8") as fh:
        next(fh)  # header
        for line in fh:
            query_id, doc_id, score = line.rstrip("\n").split("\t")
            if int(score) > 0:
                qrels.setdefault(query_id, set()).add(doc_id)

    queries: dict[str, str] = {}
    with (folder / "queries.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if str(row["_id"]) in qrels:
                queries[str(row["_id"])] = str(row["text"])
    return Dataset(corpus=corpus, queries=queries, qrels=qrels)


def subsample(dataset: Dataset, n_docs: int, seed: int = 42) -> Dataset:
    """Keep every relevant document and fill up to ``n_docs`` with random distractors."""
    relevant = {doc_id for docs in dataset.qrels.values() for doc_id in docs}
    relevant &= dataset.corpus.keys()
    others = sorted(dataset.corpus.keys() - relevant)
    random.Random(seed).shuffle(others)
    keep = relevant | set(others[: max(0, n_docs - len(relevant))])
    corpus = {doc_id: text for doc_id, text in dataset.corpus.items() if doc_id in keep}
    qrels = {q: docs & keep for q, docs in dataset.qrels.items() if docs & keep}
    queries = {q: text for q, text in dataset.queries.items() if q in qrels}
    return Dataset(corpus=corpus, queries=queries, qrels=qrels)


# --------------------------------------------------------------------------- metrics


def percentile(values: Sequence[float], pct: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(pct / 100 * len(ordered)) - 1))
    return ordered[index]


def recall_at(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant)


def mrr_at(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    for position, doc_id in enumerate(ranked[:k], start=1):
        if doc_id in relevant:
            return 1.0 / position
    return 0.0


def ndcg_at(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    dcg = sum(
        1.0 / math.log2(position + 1)
        for position, doc_id in enumerate(ranked[:k], start=1)
        if doc_id in relevant
    )
    ideal = sum(1.0 / math.log2(position + 1) for position in range(1, min(k, len(relevant)) + 1))
    return dcg / ideal


def dedupe_docs(chunk_doc_ids: Sequence[str], k: int) -> list[str]:
    """Several chunks can come from one document; rank documents by their best chunk."""
    seen: list[str] = []
    for doc_id in chunk_doc_ids:
        if doc_id not in seen:
            seen.append(doc_id)
        if len(seen) == k:
            break
    return seen


def embed_batched(embedder: Embedder, texts: Sequence[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), BATCH):
        vectors.extend(embedder.embed(texts[start : start + BATCH]))
    return vectors


# --------------------------------------------------------------------------- experiment 1 + 2


@dataclass(slots=True)
class RetrievalResult:
    embedder: str
    chunk_size: int
    chunks: int
    index_seconds: float
    chunks_per_second: float
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr_at_10: float
    ndcg_at_10: float
    embed_p50_ms: float
    search_p50_ms: float
    search_p95_ms: float
    query_p50_ms: float
    query_p95_ms: float


@dataclass(slots=True)
class ExactVsAnnResult:
    embedder: str
    chunk_size: int
    chunks: int
    overlap_at_10: float
    hnsw_p50_ms: float
    hnsw_p95_ms: float
    exact_p50_ms: float
    exact_p95_ms: float
    ndcg_hnsw: float
    ndcg_exact: float


def run_retrieval(
    dataset: Dataset,
    embedder_name: str,
    embedder: Embedder,
    chunk_size: int,
    query_ids: Sequence[str],
    compare_exact: bool,
) -> tuple[RetrievalResult, ExactVsAnnResult | None]:
    overlap = chunk_size // 7
    chunk_texts: list[str] = []
    chunk_docs: list[str] = []
    for doc_id, text in dataset.corpus.items():
        for chunk in split_text(text, chunk_size, overlap):
            chunk_texts.append(chunk)
            chunk_docs.append(doc_id)

    client = chromadb.EphemeralClient()
    name = f"bench-{uuid.uuid4().hex[:8]}"
    collection = get_cosine_collection(client, name)
    store = VectorStore(client, name)

    started = time.perf_counter()
    vectors = embed_batched(embedder, chunk_texts)
    # Added in batches to stay under Chroma's max batch size.
    for start in range(0, len(chunk_texts), 5000):
        end = start + 5000
        collection.add(
            ids=[f"{i}" for i in range(start, min(end, len(chunk_texts)))],
            documents=chunk_texts[start:end],
            embeddings=vectors[start:end],
            metadatas=[{"document_id": d} for d in chunk_docs[start:end]],
        )
    index_seconds = time.perf_counter() - started

    fetch = 30  # chunks fetched per query, then collapsed to the top-10 documents
    metrics: dict[str, list[float]] = {
        k: [] for k in ("r1", "r5", "r10", "mrr", "ndcg", "embed", "search", "total")
    }
    query_vectors: list[list[float]] = []
    hnsw_rankings: list[list[str]] = []
    for query_id in query_ids:
        relevant = dataset.qrels[query_id]
        t0 = time.perf_counter()
        query_vector = embedder.embed([dataset.queries[query_id]])[0]
        t1 = time.perf_counter()
        hits = store.search(query_vector, top_k=fetch)
        t2 = time.perf_counter()
        ranked = dedupe_docs([str(hit.metadata["document_id"]) for hit in hits], 10)
        query_vectors.append(query_vector)
        hnsw_rankings.append(ranked)
        metrics["r1"].append(recall_at(ranked, relevant, 1))
        metrics["r5"].append(recall_at(ranked, relevant, 5))
        metrics["r10"].append(recall_at(ranked, relevant, 10))
        metrics["mrr"].append(mrr_at(ranked, relevant, 10))
        metrics["ndcg"].append(ndcg_at(ranked, relevant, 10))
        metrics["embed"].append((t1 - t0) * 1000)
        metrics["search"].append((t2 - t1) * 1000)
        metrics["total"].append((t2 - t0) * 1000)

    result = RetrievalResult(
        embedder=embedder_name,
        chunk_size=chunk_size,
        chunks=len(chunk_texts),
        index_seconds=round(index_seconds, 1),
        chunks_per_second=round(len(chunk_texts) / index_seconds, 1),
        recall_at_1=round(statistics.mean(metrics["r1"]), 4),
        recall_at_5=round(statistics.mean(metrics["r5"]), 4),
        recall_at_10=round(statistics.mean(metrics["r10"]), 4),
        mrr_at_10=round(statistics.mean(metrics["mrr"]), 4),
        ndcg_at_10=round(statistics.mean(metrics["ndcg"]), 4),
        embed_p50_ms=round(percentile(metrics["embed"], 50), 2),
        search_p50_ms=round(percentile(metrics["search"], 50), 2),
        search_p95_ms=round(percentile(metrics["search"], 95), 2),
        query_p50_ms=round(percentile(metrics["total"], 50), 2),
        query_p95_ms=round(percentile(metrics["total"], 95), 2),
    )

    exact: ExactVsAnnResult | None = None
    if compare_exact:
        matrix = np.asarray(vectors, dtype=np.float32)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        exact_ms: list[float] = []
        overlaps: list[float] = []
        ndcg_exact: list[float] = []
        hnsw_ms: list[float] = metrics["search"]
        for query_id, query_vector, hnsw_ranked in zip(
            query_ids, query_vectors, hnsw_rankings, strict=True
        ):
            t0 = time.perf_counter()
            q = np.asarray(query_vector, dtype=np.float32)
            q /= np.linalg.norm(q)
            scores = matrix @ q
            top = np.argpartition(-scores, fetch)[:fetch]
            top = top[np.argsort(-scores[top])]
            exact_ms.append((time.perf_counter() - t0) * 1000)
            ranked = dedupe_docs([chunk_docs[i] for i in top], 10)
            overlaps.append(len(set(ranked) & set(hnsw_ranked)) / max(1, len(ranked)))
            ndcg_exact.append(ndcg_at(ranked, dataset.qrels[query_id], 10))
        exact = ExactVsAnnResult(
            embedder=embedder_name,
            chunk_size=chunk_size,
            chunks=len(chunk_texts),
            overlap_at_10=round(statistics.mean(overlaps), 4),
            hnsw_p50_ms=round(percentile(hnsw_ms, 50), 2),
            hnsw_p95_ms=round(percentile(hnsw_ms, 95), 2),
            exact_p50_ms=round(percentile(exact_ms, 50), 2),
            exact_p95_ms=round(percentile(exact_ms, 95), 2),
            ndcg_hnsw=result.ndcg_at_10,
            ndcg_exact=round(statistics.mean(ndcg_exact), 4),
        )
    return result, exact


# --------------------------------------------------------------------------- experiment 3


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    va, vb = np.asarray(a), np.asarray(b)
    return float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))


def run_cache_thresholds(
    embedders: dict[str, Embedder], thresholds: Sequence[float]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lang in ("pt", "en"):
        for name, embedder in embedders.items():
            same = [cosine(*embedder.embed([p[lang][0], p[lang][1]])) for p in PARAPHRASES]
            diff = [cosine(*embedder.embed([p[lang][0], p[lang][1]])) for p in DIFFERENT]
            for threshold in thresholds:
                hit_rate = sum(s >= threshold for s in same) / len(same)
                false_hits = sum(s >= threshold for s in diff) / len(diff)
                rows.append(
                    {
                        "language": lang,
                        "embedder": name,
                        "threshold": threshold,
                        "hit_rate": round(hit_rate, 3),
                        "false_hit_rate": round(false_hits, 3),
                        "mean_sim_paraphrase": round(statistics.mean(same), 3),
                        "mean_sim_different": round(statistics.mean(diff), 3),
                    }
                )
    return rows


def run_cache_latency(
    dataset: Dataset, embedder: Embedder, n_docs: int, repeats: int
) -> dict[str, Any]:
    """End-to-end RagService.query latency: cold (retrieval + generation) vs. cache hit.

    Generation uses the offline extractive provider, so the "miss" figure excludes LLM
    time; a real Claude call adds the model latency on top of it.
    """
    client = chromadb.EphemeralClient()
    suffix = uuid.uuid4().hex[:8]
    store = VectorStore(client, f"lat-docs-{suffix}")
    cache = SemanticCache(client, f"lat-cache-{suffix}", 0.92, 3600)
    extractive = ExtractiveProvider()
    rag = RagService(store, embedder, extractive, extractive, cache, 800, 120, 4)
    for doc_id, text in list(dataset.corpus.items())[:n_docs]:
        rag.ingest(f"{doc_id}.md", text.encode())

    questions = list(dataset.queries.values())[:repeats]
    miss_ms: list[float] = []
    hit_ms: list[float] = []
    asked: list[tuple[str, list[float]]] = []
    false_hits: list[dict[str, Any]] = []
    for question in questions:
        t0 = time.perf_counter()
        first = rag.query(QueryRequest(question=question))
        elapsed = (time.perf_counter() - t0) * 1000
        vector = embedder.embed([question])[0]
        if first.cached:
            # A *different* question was answered from the cache: record what it matched.
            similarity, matched = max((cosine(vector, v), q) for q, v in asked)
            false_hits.append(
                {"question": question, "matched": matched, "similarity": round(similarity, 3)}
            )
        else:
            miss_ms.append(elapsed)
        asked.append((question, vector))
        t0 = time.perf_counter()
        second = rag.query(QueryRequest(question=question))
        hit_ms.append((time.perf_counter() - t0) * 1000)
        assert second.cached
    return {
        "queries": len(questions),
        "indexed_documents": n_docs,
        "miss_p50_ms": round(percentile(miss_ms, 50), 2),
        "miss_p95_ms": round(percentile(miss_ms, 95), 2),
        "hit_p50_ms": round(percentile(hit_ms, 50), 2),
        "hit_p95_ms": round(percentile(hit_ms, 95), 2),
        "false_hits": false_hits,
    }


# --------------------------------------------------------------------------- experiment 2b + 4


def synthetic_vectors(
    rng: np.random.Generator, size: int, distribution: str
) -> npt.NDArray[np.float32]:
    """``uniform``: isotropic noise, the worst case for HNSW. ``clustered``: points around
    topic centers, closer to how real text embeddings are distributed."""
    if distribution == "uniform":
        base = rng.standard_normal((size, 384)).astype(np.float32)
    else:
        centers = rng.standard_normal((max(1, size // 20), 384)).astype(np.float32)
        noise = rng.standard_normal((size, 384)).astype(np.float32)
        base = centers[rng.integers(0, len(centers), size)] + 0.4 * noise
    return base / np.linalg.norm(base, axis=1, keepdims=True)


def run_ann_scaling(sizes: Sequence[int], n_queries: int = 200) -> list[dict[str, Any]]:
    """HNSW (Chroma defaults) vs. exact search on synthetic unit vectors (dim 384)."""
    rows: list[dict[str, Any]] = []
    for distribution in ("clustered", "uniform"):
        for size in sizes:
            rng = np.random.default_rng(7)
            base = synthetic_vectors(rng, size, distribution)
            # Queries near existing points, like real questions near their answers.
            picks = rng.choice(size, n_queries, replace=False)
            noise = rng.standard_normal((n_queries, 384)).astype(np.float32)
            queries = base[picks] + 0.02 * noise
            queries /= np.linalg.norm(queries, axis=1, keepdims=True)

            collection = get_cosine_collection(
                chromadb.EphemeralClient(), f"scale-{uuid.uuid4().hex[:8]}"
            )
            started = time.perf_counter()
            for start in range(0, size, 5000):
                end = min(start + 5000, size)
                collection.add(ids=[str(i) for i in range(start, end)], embeddings=base[start:end])
            build_seconds = time.perf_counter() - started

            hnsw_ms: list[float] = []
            exact_ms: list[float] = []
            recalls: list[float] = []
            for query in queries:
                t0 = time.perf_counter()
                result = collection.query(query_embeddings=[query], n_results=10)
                hnsw_ms.append((time.perf_counter() - t0) * 1000)
                t0 = time.perf_counter()
                scores = base @ query
                top = np.argpartition(-scores, 10)[:10]
                exact_ms.append((time.perf_counter() - t0) * 1000)
                recalls.append(len({int(i) for i in result["ids"][0]} & set(top.tolist())) / 10)
            rows.append(
                {
                    "distribution": distribution,
                    "vectors": size,
                    "build_seconds": round(build_seconds, 1),
                    "recall_at_10": round(statistics.mean(recalls), 4),
                    "hnsw_p50_ms": round(percentile(hnsw_ms, 50), 2),
                    "hnsw_p95_ms": round(percentile(hnsw_ms, 95), 2),
                    "exact_p50_ms": round(percentile(exact_ms, 50), 2),
                    "exact_p95_ms": round(percentile(exact_ms, 95), 2),
                }
            )
            print(f"  {rows[-1]}")
    return rows


def run_padding(chunks: Sequence[str], questions: Sequence[str]) -> list[dict[str, Any]]:
    """Chroma's embedding function (every input padded to 256 tokens) vs. MiniLMEmbedder
    (padded to the longest text in the batch): throughput, latency and output equality."""
    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

    fixed_fn = ONNXMiniLM_L6_V2(preferred_providers=["CPUExecutionProvider"])
    dynamic = MiniLMEmbedder()
    fixed_fn(["warm-up"])
    dynamic.embed(["warm-up"])

    def fixed(texts: Sequence[str]) -> list[list[float]]:
        return [[float(x) for x in v] for v in fixed_fn(list(texts))]

    rows: list[dict[str, Any]] = []
    for workload, texts, batched in (
        ("index chunks (batch)", chunks, True),
        ("single question", questions, False),
    ):
        timings: dict[str, float] = {}
        outputs: dict[str, npt.NDArray[np.float32]] = {}
        per_call: dict[str, list[float]] = {}
        for name, fn in (("fixed-256", fixed), ("dynamic", dynamic.embed)):
            started = time.perf_counter()
            if batched:
                vectors = fn(texts)
                per_call[name] = []
            else:
                vectors, per_call[name] = [], []
                for text in texts:
                    t0 = time.perf_counter()
                    vectors.extend(fn([text]))
                    per_call[name].append((time.perf_counter() - t0) * 1000)
            timings[name] = time.perf_counter() - started
            outputs[name] = np.asarray(vectors, dtype=np.float32)
        tokenizer = dynamic._load_tokenizer()
        tokens = [len(tokenizer.encode(text).ids) for text in texts]  # unpadded lengths
        rows.append(
            {
                "workload": workload,
                "texts": len(texts),
                "mean_tokens": round(statistics.mean(tokens), 1),
                "fixed_texts_per_s": round(len(texts) / timings["fixed-256"], 1),
                "dynamic_texts_per_s": round(len(texts) / timings["dynamic"], 1),
                "fixed_p50_ms": round(percentile(per_call["fixed-256"], 50), 1)
                if per_call["fixed-256"]
                else "-",
                "dynamic_p50_ms": round(percentile(per_call["dynamic"], 50), 1)
                if per_call["dynamic"]
                else "-",
                "speedup": round(timings["fixed-256"] / timings["dynamic"], 2),
                "min_cosine": round(
                    float((outputs["fixed-256"] * outputs["dynamic"]).sum(1).min()), 6
                ),
            }
        )
        print(f"  {rows[-1]}")
    return rows


# --------------------------------------------------------------------------- report


def markdown_table(rows: Sequence[dict[str, Any]], columns: dict[str, str]) -> str:
    header = "| " + " | ".join(columns.values()) + " |"
    divider = "|" + "|".join("---" for _ in columns) + "|"
    body = ["| " + " | ".join(str(row[key]) for key in columns) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def write_report(results: dict[str, Any]) -> None:
    (ROOT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    env = results["environment"]
    parts = [
        "# Benchmark results",
        "",
        f"Generated by `python -m benchmarks.run_benchmarks` on {env['date']}  ",
        f"Machine: {env['processor']} · {env['cpu_count']} logical CPUs · "
        f"Python {env['python']} · chromadb {env['chromadb']} · CPU only",
        "",
        f"Dataset: BEIR/SciFact subset, {env['documents']} documents (all "
        f"{env['relevant_documents']} relevant ones + seeded random distractors), "
        f"{env['queries']} test queries.",
        "",
        "## 1. Retrieval quality vs. latency",
        "",
        markdown_table(
            results["retrieval"],
            {
                "embedder": "Embedder",
                "chunk_size": "Chunk (chars)",
                "chunks": "Chunks",
                "recall_at_1": "Recall@1",
                "recall_at_5": "Recall@5",
                "recall_at_10": "Recall@10",
                "mrr_at_10": "MRR@10",
                "ndcg_at_10": "nDCG@10",
                "embed_p50_ms": "Embed p50 (ms)",
                "search_p50_ms": "Search p50 (ms)",
                "query_p95_ms": "Query p95 (ms)",
                "chunks_per_second": "Indexing (chunks/s)",
            },
        ),
        "",
        "## 2. HNSW (approximate) vs. exact search",
        "",
        markdown_table(
            results["exact_vs_ann"],
            {
                "embedder": "Embedder",
                "chunk_size": "Chunk",
                "chunks": "Chunks",
                "overlap_at_10": "Top-10 overlap",
                "ndcg_hnsw": "nDCG@10 HNSW",
                "ndcg_exact": "nDCG@10 exact",
                "hnsw_p50_ms": "HNSW p50 (ms)",
                "hnsw_p95_ms": "HNSW p95 (ms)",
                "exact_p50_ms": "Exact p50 (ms)",
                "exact_p95_ms": "Exact p95 (ms)",
            },
        ),
        "",
        "Synthetic scaling (unit vectors, dim 384, 200 queries, Chroma default HNSW settings):",
        "",
        markdown_table(
            results["ann_scaling"],
            {
                "distribution": "Distribution",
                "vectors": "Vectors",
                "build_seconds": "Build (s)",
                "recall_at_10": "Recall@10 vs exact",
                "hnsw_p50_ms": "HNSW p50 (ms)",
                "hnsw_p95_ms": "HNSW p95 (ms)",
                "exact_p50_ms": "Exact p50 (ms)",
                "exact_p95_ms": "Exact p95 (ms)",
            },
        ),
        "",
        "## 3. Semantic cache",
        "",
        f"{len(PARAPHRASES)} paraphrase pairs (should hit) and {len(DIFFERENT)} pairs of "
        "similar-looking but different questions (should miss), in Portuguese and English.",
        "",
        markdown_table(
            results["cache_thresholds"],
            {
                "language": "Lang",
                "embedder": "Embedder",
                "threshold": "Threshold",
                "hit_rate": "Hit rate",
                "false_hit_rate": "False hits",
                "mean_sim_paraphrase": "Mean sim (paraphrase)",
                "mean_sim_different": "Mean sim (different)",
            },
        ),
        "",
        "End-to-end `RagService.query` latency (MiniLM, extractive generation, so LLM time "
        "is excluded):",
        "",
        markdown_table(
            [results["cache_latency"]],
            {k: k for k in results["cache_latency"] if k != "false_hits"},
        ),
        "",
        "Distinct questions answered from the cache at the default threshold (0.92):",
        "",
        markdown_table(
            results["cache_latency"]["false_hits"],
            {"question": "Question", "matched": "Served answer of", "similarity": "Similarity"},
        ),
        "",
        "## 4. Embedding throughput: fixed vs. dynamic padding",
        "",
        markdown_table(results["padding"], {k: k for k in results["padding"][0]}),
        "",
    ]
    (ROOT / "results.md").write_text("\n".join(parts), encoding="utf-8")


def timed(label: str, fn: Callable[[], Any]) -> Any:
    started = time.perf_counter()
    value = fn()
    print(f"  {label}: {time.perf_counter() - started:.1f}s")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="60 queries and fewer configs")
    parser.add_argument("--docs", type=int, default=1500, help="corpus size (max 5183)")
    args = parser.parse_args()

    configure_logging("WARNING", json_logs=False)
    full = load_scifact()
    dataset = subsample(full, 500 if args.quick else args.docs)
    relevant_docs = len({d for docs in dataset.qrels.values() for d in docs})
    query_ids = sorted(dataset.queries)[: 60 if args.quick else None]
    embedders: dict[str, Embedder] = {
        "minilm-onnx": MiniLMEmbedder(),
        "hashing-256": HashEmbedder(),
    }
    chunk_sizes = [800] if args.quick else [400, 800, 1600]

    retrieval: list[dict[str, Any]] = []
    exact_vs_ann: list[dict[str, Any]] = []
    for name, embedder in embedders.items():
        for chunk_size in chunk_sizes:
            print(f"[retrieval] {name} chunk={chunk_size}")
            result, exact = timed(
                "done",
                lambda e=embedder, n=name, c=chunk_size: run_retrieval(
                    dataset, n, e, c, query_ids, compare_exact=True
                ),
            )
            retrieval.append(asdict(result))
            if exact is not None:
                exact_vs_ann.append(asdict(exact))

    print("[cache] thresholds")
    thresholds = [0.80, 0.85, 0.90, 0.92, 0.95]
    cache_rows = run_cache_thresholds(embedders, thresholds)
    print("[cache] latency")
    cache_latency = run_cache_latency(
        dataset, embedders["minilm-onnx"], 100 if args.quick else 300, 30 if args.quick else 100
    )
    print("[ann] scaling")
    ann_scaling = run_ann_scaling([10_000] if args.quick else [10_000, 50_000, 100_000])
    print("[padding]")
    sample = [c for text in list(dataset.corpus.values())[:200] for c in split_text(text, 800, 114)]
    questions = list(dataset.queries.values())[: 20 if args.quick else 100]
    padding = run_padding(sample[: 64 if args.quick else 256], questions)

    results = {
        "environment": {
            "date": time.strftime("%Y-%m-%d"),
            "processor": platform.processor() or platform.machine(),
            "cpu_count": os.cpu_count(),
            "python": platform.python_version(),
            "chromadb": chromadb.__version__,
            "documents": len(dataset.corpus),
            "relevant_documents": relevant_docs,
            "queries": len(query_ids),
        },
        "retrieval": retrieval,
        "exact_vs_ann": exact_vs_ann,
        "ann_scaling": ann_scaling,
        "cache_thresholds": cache_rows,
        "cache_latency": cache_latency,
        "padding": padding,
    }
    write_report(results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
