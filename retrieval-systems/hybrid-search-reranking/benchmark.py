"""Comparative benchmark runner for four single-stage and two-stage retrieval pipelines."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence

import numpy as np

from config import RetrievalSettings, load_settings
from corpus import load_corpus, load_eval_queries
from retrievers import (
    BM25LexicalRetriever,
    DenseBiEncoderRetriever,
    HybridRRFRetriever,
    SentenceTransformerBiEncoder,
    SentenceTransformerCrossEncoder,
    TwoStageHybridRetriever,
)
from schemas import (
    BenchmarkReport,
    CorpusChunk,
    EvalQuery,
    PipelineName,
    QuerySlice,
    ScoredCandidate,
    SliceMetrics,
    StageMetrics,
)


def compute_recall_at_k(
    ranked_ids: Sequence[str],
    relevant_grades: Mapping[str, int],
    k: int,
) -> float:
    """Fraction of relevant chunks (grade >= 1) present within the top-k results."""
    relevant_ids = {cid for cid, grade in relevant_grades.items() if grade >= 1}
    if not relevant_ids:
        return 0.0
    hits = sum(1 for cid in ranked_ids[:k] if cid in relevant_ids)
    return hits / len(relevant_ids)


def compute_mrr_at_k(
    ranked_ids: Sequence[str],
    relevant_grades: Mapping[str, int],
    k: int,
    primary_min_grade: int = 2,
) -> float:
    """Reciprocal rank of the highest-ranked primary target chunk (grade >= 2) within top-k."""
    primary_ids = {
        cid for cid, grade in relevant_grades.items() if grade >= primary_min_grade
    }
    for idx, cid in enumerate(ranked_ids[:k], start=1):
        if cid in primary_ids:
            return 1.0 / idx
    return 0.0


def compute_ndcg_at_k(
    ranked_ids: Sequence[str],
    relevant_grades: Mapping[str, int],
    k: int,
) -> float:
    """Graded nDCG@k using exponential gain (2^rel - 1) / log2(rank + 1)."""
    dcg = 0.0
    for idx, cid in enumerate(ranked_ids[:k], start=1):
        rel = relevant_grades.get(cid, 0)
        if rel > 0:
            dcg += ((2**rel) - 1.0) / math.log2(idx + 1.0)

    ideal_rels = sorted((g for g in relevant_grades.values() if g > 0), reverse=True)[:k]
    idcg = sum(
        ((2**rel) - 1.0) / math.log2(idx + 1.0)
        for idx, rel in enumerate(ideal_rels, start=1)
    )
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def evaluate_query_subset(
    queries: Sequence[EvalQuery],
    predictions_by_query: Mapping[str, Sequence[ScoredCandidate]],
    eval_top_k: int = 10,
    stage1_top_k: int = 25,
    rerank_top_k: int = 5,
) -> SliceMetrics:
    """Aggregate Recall@5, Recall@25, MRR@10, and nDCG@10 across a query slice."""
    if not queries:
        raise ValueError("Cannot evaluate an empty query slice")

    recalls_5: list[float] = []
    recalls_25: list[float] = []
    mrrs_10: list[float] = []
    ndcgs_10: list[float] = []

    for q in queries:
        ranked_ids = [cand.chunk_id for cand in predictions_by_query[q.query_id]]
        recalls_5.append(compute_recall_at_k(ranked_ids, q.relevant_grades, k=rerank_top_k))
        recalls_25.append(compute_recall_at_k(ranked_ids, q.relevant_grades, k=stage1_top_k))
        mrrs_10.append(compute_mrr_at_k(ranked_ids, q.relevant_grades, k=eval_top_k))
        ndcgs_10.append(compute_ndcg_at_k(ranked_ids, q.relevant_grades, k=eval_top_k))

    return SliceMetrics(
        query_count=len(queries),
        recall_at_5=round(float(np.mean(recalls_5)), 4),
        recall_at_25=round(float(np.mean(recalls_25)), 4),
        mrr_at_10=round(float(np.mean(mrrs_10)), 4),
        ndcg_at_10=round(float(np.mean(ndcgs_10)), 4),
    )


def _measure_pipeline(
    pipeline_id: PipelineName,
    pipeline_label: str,
    search_fn: Callable[[str], list[ScoredCandidate]],
    queries: Sequence[EvalQuery],
    settings: RetrievalSettings,
) -> StageMetrics:
    """Warm up, time per-query execution across repeats, and compute overall + slice metrics."""
    for warm_q in queries[: settings.warmup_queries]:
        search_fn(warm_q.query_text)

    predictions: dict[str, list[ScoredCandidate]] = {}
    latencies_ms: list[float] = []

    for repeat_idx in range(settings.latency_repeats):
        for q in queries:
            t0 = time.perf_counter()
            hits = search_fn(q.query_text)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(elapsed_ms)
            if repeat_idx == 0:
                predictions[q.query_id] = hits

    overall = evaluate_query_subset(
        queries=queries,
        predictions_by_query=predictions,
        eval_top_k=settings.eval_top_k,
        stage1_top_k=settings.stage1_top_k,
        rerank_top_k=settings.rerank_top_k,
    )

    slices: dict[QuerySlice, SliceMetrics] = {}
    for slice_name in ("exact_id", "semantic_paraphrase"):
        subset = [q for q in queries if q.query_slice == slice_name]
        slices[slice_name] = evaluate_query_subset(
            queries=subset,
            predictions_by_query=predictions,
            eval_top_k=settings.eval_top_k,
            stage1_top_k=settings.stage1_top_k,
            rerank_top_k=settings.rerank_top_k,
        )

    return StageMetrics(
        pipeline_id=pipeline_id,
        pipeline_label=pipeline_label,
        overall=overall,
        by_slice=slices,
        latency_p50_ms=round(float(np.percentile(latencies_ms, 50)), 2),
        latency_p95_ms=round(float(np.percentile(latencies_ms, 95)), 2),
        latency_mean_ms=round(float(np.mean(latencies_ms)), 2),
    )


def run_benchmark(
    settings: RetrievalSettings | None = None,
    chunks: Sequence[CorpusChunk] | None = None,
    queries: Sequence[EvalQuery] | None = None,
) -> BenchmarkReport:
    """Execute all four retrieval pipelines and assemble a validated BenchmarkReport."""
    cfg = settings or load_settings()
    np.random.seed(cfg.seed)

    corpus_chunks = tuple(chunks) if chunks is not None else load_corpus()
    eval_queries = tuple(queries) if queries is not None else load_eval_queries()

    bi_encoder = SentenceTransformerBiEncoder(cfg.bi_encoder_model, device=cfg.device)
    cross_encoder = SentenceTransformerCrossEncoder(
        cfg.cross_encoder_model,
        device=cfg.device,
        batch_size=cfg.rerank_batch_size,
    )

    bm25 = BM25LexicalRetriever(corpus_chunks)
    dense = DenseBiEncoderRetriever(corpus_chunks, encoder=bi_encoder)
    hybrid_rrf = HybridRRFRetriever(
        lexical=bm25,
        dense=dense,
        rrf_k=cfg.rrf_k,
        stage1_top_k=cfg.stage1_top_k,
    )
    two_stage = TwoStageHybridRetriever(
        chunks=corpus_chunks,
        stage1=hybrid_rrf,
        reranker=cross_encoder,
        stage1_top_k=cfg.stage1_top_k,
        rerank_top_k=cfg.rerank_top_k,
    )

    pipelines: tuple[tuple[PipelineName, str, Callable[[str], list[ScoredCandidate]]], ...] = (
        ("bm25_only", "1. BM25 Only", lambda q: bm25.search(q, top_k=cfg.stage1_top_k)),
        (
            "dense_only",
            "2. Dense Bi-Encoder Only",
            lambda q: dense.search(q, top_k=cfg.stage1_top_k),
        ),
        (
            "hybrid_rrf",
            f"3. Hybrid BM25 + Dense RRF (k={cfg.rrf_k})",
            lambda q: hybrid_rrf.search(q, top_k=cfg.stage1_top_k),
        ),
        (
            "hybrid_rrf_rerank",
            f"4. Two-Stage Hybrid RRF + Cross-Encoder (top-{cfg.stage1_top_k})",
            lambda q: two_stage.search(q, top_k=cfg.stage1_top_k),
        ),
    )

    stage_results = tuple(
        _measure_pipeline(
            pipeline_id=pid,
            pipeline_label=label,
            search_fn=fn,
            queries=eval_queries,
            settings=cfg,
        )
        for pid, label, fn in pipelines
    )

    return BenchmarkReport(
        corpus_size=len(corpus_chunks),
        query_count=len(eval_queries),
        bi_encoder_model=cfg.bi_encoder_model,
        cross_encoder_model=cfg.cross_encoder_model,
        device=cfg.device,
        rrf_k=cfg.rrf_k,
        stage1_top_k=cfg.stage1_top_k,
        rerank_top_k=cfg.rerank_top_k,
        eval_top_k=cfg.eval_top_k,
        stages=stage_results,
    )


def format_report_table(report: BenchmarkReport) -> str:
    """Format human-readable overall and per-slice ASCII scorecards."""
    lines: list[str] = []
    lines.append(
        f"=== Two-Stage Hybrid Retrieval Benchmark "
        f"({report.corpus_size} chunks, {report.query_count} queries, device={report.device}) ==="
    )
    lines.append(
        f"Models: bi_encoder={report.bi_encoder_model} | "
        f"cross_encoder={report.cross_encoder_model} | rrf_k={report.rrf_k}"
    )
    lines.append("")
    header = (
        f"{'Pipeline':<48} | {'Recall@5':>8} | {'Recall@25':>9} | "
        f"{'MRR@10':>6} | {'nDCG@10':>7} | {'p50 ms':>7} | {'p95 ms':>7}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for st in report.stages:
        ov = st.overall
        lines.append(
            f"{st.pipeline_label:<48} | {ov.recall_at_5:>8.3f} | {ov.recall_at_25:>9.3f} | "
            f"{ov.mrr_at_10:>6.3f} | {ov.ndcg_at_10:>7.3f} | "
            f"{st.latency_p50_ms:>7.2f} | {st.latency_p95_ms:>7.2f}"
        )

    lines.append("")
    slice_header = (
        f"{'Pipeline':<48} | {'Exact R@5':>9} | {'Exact nDCG':>10} | "
        f"{'Sem R@5':>7} | {'Sem nDCG':>8}"
    )
    lines.append("=== Diagnostic Slice Breakdown (exact_id vs semantic_paraphrase) ===")
    lines.append(slice_header)
    lines.append("-" * len(slice_header))
    for st in report.stages:
        ex = st.by_slice["exact_id"]
        sem = st.by_slice["semantic_paraphrase"]
        lines.append(
            f"{st.pipeline_label:<48} | {ex.recall_at_5:>9.3f} | {ex.ndcg_at_10:>10.3f} | "
            f"{sem.recall_at_5:>7.3f} | {sem.ndcg_at_10:>8.3f}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the 4-pipeline hybrid retrieval and cross-encoder reranking benchmark."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the full BenchmarkReport as formatted JSON after the summary table.",
    )
    args = parser.parse_args()
    report = run_benchmark()
    print(format_report_table(report))
    if args.json:
        print("\n" + json.dumps(report.model_dump(), indent=2))


if __name__ == "__main__":
    main()
