"""Bi-encoder retrieval benchmark across symmetric and asymmetric tasks plus batched timing."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence

import numpy as np

from config import BiEncoderSettings, load_settings
from index import DenseVectorIndex, SentenceTransformerEncoder, TextEncoder
from schemas import (
    BenchmarkQuery,
    BiEncoderBenchmarkReport,
    DocumentRecord,
    QueryTask,
    RetrievalMetrics,
    SearchHit,
    SearchModeTiming,
)


def load_corpus() -> tuple[DocumentRecord, ...]:
    """Return 24 multi-domain documents covering mechanical repair, ML, Python, and systems."""
    return (
        DocumentRecord(
            doc_id="auto-flat-tire-highway",
            domain="mechanical-repair",
            title="How to Change a Flat Tire Safely on the Highway",
            body=(
                "Pull onto a level shoulder, engage the parking brake, place wheel chocks, "
                "loosen lug nuts slightly before jacking the frame, mount the spare wheel, "
                "and torque the lug nuts in a star pattern."
            ),
        ),
        DocumentRecord(
            doc_id="bike-inner-tube-swap",
            domain="mechanical-repair",
            title="Guide to Replacing Punctured Bicycle Inner Tubes",
            body=(
                "Unseat one tire bead with nylon levers, inspect the casing interior for "
                "glass or thorns, pre-inflate the replacement butyl tube slightly to avoid "
                "pinch flats, and seat the bead evenly around the rim."
            ),
        ),
        DocumentRecord(
            doc_id="auto-engine-overheat",
            domain="mechanical-repair",
            title="Troubleshooting Car Engine Cooling and Thermostat Failures",
            body=(
                "Monitor coolant temperature gauges, inspect radiator hoses for pressure leaks, "
                "verify electric fan relay activation, and never open a pressurized radiator cap "
                "while the engine block remains hot."
            ),
        ),
        DocumentRecord(
            doc_id="bike-disc-brake-bleed",
            domain="mechanical-repair",
            title="Bleeding Hydraulic Disc Brakes on Road and Trail Bikes",
            body=(
                "Remove contaminated brake pads, attach a syringe filled with degassed mineral "
                "oil at the caliper bleed port, and purge trapped air bubbles upward to the lever."
            ),
        ),
        DocumentRecord(
            doc_id="py-code-review-standards",
            domain="python-engineering",
            title="Engineering Checklist for Python Code Reviews",
            body=(
                "Verify explicit boundary type validation, single-responsibility functions, "
                "deterministic unit tests with injected fakes, and specific exception handling "
                "without silent bare except blocks."
            ),
        ),
        DocumentRecord(
            doc_id="py-asyncio-taskgroups",
            domain="python-engineering",
            title="Structured Concurrency with Python Asyncio TaskGroup",
            body=(
                "Use asyncio.TaskGroup to scope concurrent coroutines so an unhandled exception "
                "in one child cancels sibling tasks deterministically and propagates ExceptionGroup."
            ),
        ),
        DocumentRecord(
            doc_id="py-memory-profiling",
            domain="python-engineering",
            title="Diagnosing Python Heap Growth with Tracemalloc",
            body=(
                "Capture tracemalloc snapshots across request batches to isolate retained "
                "reference cycles, unbounded module caches, and oversized NumPy buffer copies."
            ),
        ),
        DocumentRecord(
            doc_id="py-packaging-pyproject",
            domain="python-engineering",
            title="Reproducible Python Packaging with Pyproject and Virtualenvs",
            body=(
                "Declare runtime version bounds and dev tool dependencies inside pyproject.toml "
                "and isolate environments per project instead of mutating global site-packages."
            ),
        ),
        DocumentRecord(
            doc_id="dl-backprop-gradients",
            domain="deep-learning",
            title="Understanding Neural Network Backpropagation and Vanishing Gradients",
            body=(
                "Reverse-mode automatic differentiation applies the chain rule from scalar loss "
                "back to layer weights, where residual skip connections and layer normalization "
                "preserve gradient magnitude across deep stacks."
            ),
        ),
        DocumentRecord(
            doc_id="dl-adamw-cosine-schedule",
            domain="deep-learning",
            title="Deep Learning Optimization with Decoupled Weight Decay and Warmup",
            body=(
                "AdamW separates L2 weight decay from adaptive moment rescaling and pairs linear "
                "learning-rate warmup with cosine annealing to stabilize transformer training."
            ),
        ),
        DocumentRecord(
            doc_id="dl-contrastive-hard-negatives",
            domain="deep-learning",
            title="Training Bi-Encoders with In-Batch and Hard Negative Triplets",
            body=(
                "Contrastive InfoNCE and triplet losses pull positive query-passage pairs "
                "together while BM25-mined hard negatives teach fine semantic boundaries."
            ),
        ),
        DocumentRecord(
            doc_id="dl-mixed-precision-bf16",
            domain="deep-learning",
            title="Mixed-Precision Training and Inference with BFloat16 Tensor Cores",
            body=(
                "BFloat16 preserves the eight-bit exponent dynamic range of FP32 while halving "
                "activation memory bandwidth and accelerating matrix multiplications."
            ),
        ),
        DocumentRecord(
            doc_id="mlops-canary-rollouts",
            domain="production-ml",
            title="Machine Learning Model Deployment with Shadow and Canary Traffic",
            body=(
                "Mirror live production requests to candidate model versions asynchronously "
                "before shifting five percent canary traffic gated on p95 latency and calibration."
            ),
        ),
        DocumentRecord(
            doc_id="mlops-feature-drift",
            domain="production-ml",
            title="Detecting Covariate Shift in Production Feature Pipelines",
            body=(
                "Compare rolling inference feature distributions against training baselines "
                "using Population Stability Index and Kolmogorov-Smirnov two-sample statistics."
            ),
        ),
        DocumentRecord(
            doc_id="mlops-vector-ann-hnsw",
            domain="production-ml",
            title="Scaling Approximate Nearest Neighbor Search with HNSW Graphs",
            body=(
                "Hierarchical Navigable Small World graphs traverse multi-layer proximity links "
                "to deliver sub-millisecond vector recall across tens of millions of embeddings."
            ),
        ),
        DocumentRecord(
            doc_id="mlops-batch-gpu-encoding",
            domain="production-ml",
            title="High-Throughput Corpus Encoding with Length-Sorted GPU Batches",
            body=(
                "Sorting documents by token count before batched bi-encoder forward passes "
                "minimizes padding waste and increases sustained embedding throughput."
            ),
        ),
        DocumentRecord(
            doc_id="db-postgres-btree-partial",
            domain="data-systems",
            title="PostgreSQL Partial B-Tree Indexes for Skewed Status Columns",
            body=(
                "Create a filtered B-tree index over active or pending rows only so background "
                "workers scan a compact index structure without indexing archived history."
            ),
        ),
        DocumentRecord(
            doc_id="db-wal-checkpoint-tuning",
            domain="data-systems",
            title="Write-Ahead Log Checkpoint Smoothing Under Heavy Write Bursts",
            body=(
                "Increase max_wal_size and checkpoint_completion_target so dirty shared buffers "
                "flush gradually instead of stalling foreground commit latency."
            ),
        ),
        DocumentRecord(
            doc_id="db-isolation-serializable",
            domain="data-systems",
            title="Preventing Write Skew Anomalies with Serializable Snapshot Isolation",
            body=(
                "Repeatable Read permits concurrent transactions to violate cross-row invariants, "
                "whereas Serializable Snapshot Isolation detects rw-antidependency cycles."
            ),
        ),
        DocumentRecord(
            doc_id="db-connection-pooling",
            domain="data-systems",
            title="Transaction-Level Connection Multiplexing with PgBouncer",
            body=(
                "Multiplex thousands of stateless application workers onto a bounded pool of "
                "PostgreSQL backend processes to cap per-connection memory and lock contention."
            ),
        ),
        DocumentRecord(
            doc_id="kitchen-sourdough-ferment",
            domain="culinary-craft",
            title="How to Bake Naturally Leavened Sourdough Bread from Scratch",
            body=(
                "Autolyse high-protein bread flour with water, fold in an active levain starter, "
                "develop gluten through coil folds during bulk fermentation, and bake in a Dutch oven."
            ),
        ),
        DocumentRecord(
            doc_id="kitchen-brown-butter-cookies",
            domain="culinary-craft",
            title="Recipe for Brown-Butter Chocolate Chip Cookies with Sea Salt",
            body=(
                "Toast milk solids in unsalted butter until nutty, chill the hydrated dough "
                "twenty-four hours to concentrate caramel notes, and bake until edges set."
            ),
        ),
        DocumentRecord(
            doc_id="net-tcp-keepalive-probes",
            domain="linux-networking",
            title="Configuring Linux TCP Keepalive Probes Behind Idle NAT Gateways",
            body=(
                "Lower net.ipv4.tcp_keepalive_time below intermediate firewall idle timeouts so "
                "long-lived RPC streams send periodic heartbeat ACKs before state eviction."
            ),
        ),
        DocumentRecord(
            doc_id="net-dns-ndots-kubernetes",
            domain="linux-networking",
            title="Reducing Kubernetes CoreDNS Tail Latency Caused by Search Path Expansion",
            body=(
                "Default ndots:5 settings expand external hostnames across four cluster suffix "
                "lookups before querying upstream resolvers; lowering ndots cuts DNS amplification."
            ),
        ),
    )


def load_eval_queries() -> tuple[BenchmarkQuery, ...]:
    """Return 16 evaluation queries: 8 symmetric paraphrases and 8 asymmetric short lookups."""
    return (
        BenchmarkQuery(
            query_id="sym-highway-tire",
            task="symmetric",
            query_text="Step-by-step instructions for safely swapping a punctured car wheel beside a busy road",
            relevant_doc_ids=("auto-flat-tire-highway",),
        ),
        BenchmarkQuery(
            query_id="sym-bike-tube",
            task="symmetric",
            query_text="How do I remove a bike tire bead and install a fresh inner tube without pinching it?",
            relevant_doc_ids=("bike-inner-tube-swap",),
        ),
        BenchmarkQuery(
            query_id="sym-py-review",
            task="symmetric",
            query_text="What quality standards and test checks should engineers verify when reviewing Python pull requests?",
            relevant_doc_ids=("py-code-review-standards",),
        ),
        BenchmarkQuery(
            query_id="sym-backprop-chain-rule",
            task="symmetric",
            query_text="How reverse-mode autodiff propagates loss gradients through deep neural layers without vanishing",
            relevant_doc_ids=("dl-backprop-gradients",),
        ),
        BenchmarkQuery(
            query_id="sym-hard-negative-mining",
            task="symmetric",
            query_text="Why mining difficult negative passages improves contrastive sentence embedding models",
            relevant_doc_ids=("dl-contrastive-hard-negatives",),
        ),
        BenchmarkQuery(
            query_id="sym-canary-shadow-deploy",
            task="symmetric",
            query_text="Rolling out a new machine learning model safely by mirroring live traffic before a small canary",
            relevant_doc_ids=("mlops-canary-rollouts",),
        ),
        BenchmarkQuery(
            query_id="sym-partial-btree-index",
            task="symmetric",
            query_text="Indexing only unarchived pending rows in PostgreSQL so queries stay small and fast",
            relevant_doc_ids=("db-postgres-btree-partial",),
        ),
        BenchmarkQuery(
            query_id="sym-sourdough-levain",
            task="symmetric",
            query_text="Baking crusty artisan bread at home using wild yeast starter, bulk folds, and a Dutch oven",
            relevant_doc_ids=("kitchen-sourdough-ferment",),
        ),
        BenchmarkQuery(
            query_id="asym-punctured-wheel",
            task="asymmetric",
            query_text="fixing a punctured wheel",
            relevant_doc_ids=("auto-flat-tire-highway", "bike-inner-tube-swap"),
        ),
        BenchmarkQuery(
            query_id="asym-python-memory-leak",
            task="asymmetric",
            query_text="python memory leak heap snapshots",
            relevant_doc_ids=("py-memory-profiling",),
        ),
        BenchmarkQuery(
            query_id="asym-adamw-warmup",
            task="asymmetric",
            query_text="decoupled weight decay cosine lr schedule",
            relevant_doc_ids=("dl-adamw-cosine-schedule",),
        ),
        BenchmarkQuery(
            query_id="asym-hnsw-ann",
            task="asymmetric",
            query_text="graph approximate nearest neighbor vector search",
            relevant_doc_ids=("mlops-vector-ann-hnsw",),
        ),
        BenchmarkQuery(
            query_id="asym-padding-batch-encode",
            task="asymmetric",
            query_text="sort token length batch embedding throughput",
            relevant_doc_ids=("mlops-batch-gpu-encoding",),
        ),
        BenchmarkQuery(
            query_id="asym-write-skew",
            task="asymmetric",
            query_text="prevent write skew concurrent transactions",
            relevant_doc_ids=("db-isolation-serializable",),
        ),
        BenchmarkQuery(
            query_id="asym-nat-idle-drop",
            task="asymmetric",
            query_text="idle firewall dropping long tcp connections",
            relevant_doc_ids=("net-tcp-keepalive-probes",),
        ),
        BenchmarkQuery(
            query_id="asym-k8s-dns-latency",
            task="asymmetric",
            query_text="kubernetes coredns ndots search suffix lookups",
            relevant_doc_ids=("net-dns-ndots-kubernetes",),
        ),
    )


def compute_recall_at_k(
    ranked_doc_ids: Sequence[str],
    relevant_doc_ids: Sequence[str],
    k: int,
) -> float:
    """Fraction of ground-truth relevant documents retrieved within top-k."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    relevant_set = set(relevant_doc_ids)
    if not relevant_set:
        return 0.0
    hits = sum(1 for doc_id in ranked_doc_ids[:k] if doc_id in relevant_set)
    return hits / len(relevant_set)


def compute_mrr_at_k(
    ranked_doc_ids: Sequence[str],
    relevant_doc_ids: Sequence[str],
    k: int,
) -> float:
    """Reciprocal rank of the highest-ranked relevant document within top-k."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    relevant_set = set(relevant_doc_ids)
    for rank, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def evaluate_predictions(
    queries: Sequence[BenchmarkQuery],
    hits_by_query: Mapping[str, Sequence[SearchHit]],
    k: int,
) -> RetrievalMetrics:
    """Aggregate Recall@k and MRR@k over a non-empty slice of benchmark queries."""
    if not queries:
        raise ValueError("Cannot evaluate an empty query slice")

    recalls: list[float] = []
    mrrs: list[float] = []
    for query in queries:
        ranked_ids = [hit.doc_id for hit in hits_by_query[query.query_id]]
        recalls.append(compute_recall_at_k(ranked_ids, query.relevant_doc_ids, k=k))
        mrrs.append(compute_mrr_at_k(ranked_ids, query.relevant_doc_ids, k=k))

    return RetrievalMetrics(
        query_count=len(queries),
        recall_at_k=round(float(np.mean(recalls)), 4),
        mrr_at_k=round(float(np.mean(mrrs)), 4),
    )


def run_benchmark(
    settings: BiEncoderSettings | None = None,
    encoder: TextEncoder | None = None,
    documents: Sequence[DocumentRecord] | None = None,
    queries: Sequence[BenchmarkQuery] | None = None,
) -> BiEncoderBenchmarkReport:
    """Build the normalized dense index, evaluate symmetric vs asymmetric recall, and time batching."""
    cfg = settings or load_settings()
    np.random.seed(cfg.seed)

    corpus = tuple(documents) if documents is not None else load_corpus()
    eval_queries = tuple(queries) if queries is not None else load_eval_queries()
    active_encoder = encoder or SentenceTransformerEncoder(
        model_name=cfg.model_name,
        device=cfg.device,
    )

    index = DenseVectorIndex(encoder=active_encoder)
    index.index_documents(corpus, batch_size=cfg.encode_batch_size)
    query_texts = [q.query_text for q in eval_queries]

    for _ in range(cfg.warmup_runs):
        index.search_exact(query_texts[0], top_k=cfg.eval_top_k)
        index.search_batched(query_texts, top_k=cfg.eval_top_k, batch_size=cfg.encode_batch_size)

    single_totals_ms: list[float] = []
    single_per_query_ms: list[float] = []
    predictions_single: dict[str, list[SearchHit]] = {}

    for repeat_idx in range(cfg.timing_repeats):
        t_batch_start = time.perf_counter()
        for q in eval_queries:
            t0 = time.perf_counter()
            hits = index.search_exact(q.query_text, top_k=cfg.eval_top_k)
            single_per_query_ms.append((time.perf_counter() - t0) * 1000.0)
            if repeat_idx == 0:
                predictions_single[q.query_id] = hits
        single_totals_ms.append((time.perf_counter() - t_batch_start) * 1000.0)

    batched_totals_ms: list[float] = []
    for _ in range(cfg.timing_repeats):
        t0 = time.perf_counter()
        index.search_batched(
            query_texts,
            top_k=cfg.eval_top_k,
            batch_size=cfg.encode_batch_size,
        )
        batched_totals_ms.append((time.perf_counter() - t0) * 1000.0)

    overall = evaluate_predictions(eval_queries, predictions_single, k=cfg.eval_top_k)
    by_task: dict[QueryTask, RetrievalMetrics] = {}
    for task_name in ("symmetric", "asymmetric"):
        subset = [q for q in eval_queries if q.task == task_name]
        by_task[task_name] = evaluate_predictions(subset, predictions_single, k=cfg.eval_top_k)

    mean_single_total = max(float(np.mean(single_totals_ms)), 1e-6)
    mean_batched_total = max(float(np.mean(batched_totals_ms)), 1e-6)
    q_count = len(eval_queries)

    timings = (
        SearchModeTiming(
            mode="single_query_loop",
            query_count=q_count,
            total_ms=round(mean_single_total, 2),
            per_query_p50_ms=round(float(np.percentile(single_per_query_ms, 50)), 2),
            queries_per_second=round((q_count * 1000.0) / mean_single_total, 1),
        ),
        SearchModeTiming(
            mode="batched_matrix",
            query_count=q_count,
            total_ms=round(mean_batched_total, 2),
            per_query_p50_ms=round(float(np.percentile(batched_totals_ms, 50)) / q_count, 2),
            queries_per_second=round((q_count * 1000.0) / mean_batched_total, 1),
        ),
    )

    return BiEncoderBenchmarkReport(
        model_name=cfg.model_name,
        device=cfg.device,
        embedding_dim=index.embedding_dim,
        corpus_size=index.size,
        eval_k=cfg.eval_top_k,
        overall=overall,
        by_task=by_task,
        timings=timings,
    )


def format_report_table(report: BiEncoderBenchmarkReport) -> str:
    """Format an ASCII summary table for terminal output."""
    lines: list[str] = []
    lines.append(
        f"=== Bi-Encoder Semantic Search Benchmark "
        f"(model={report.model_name}, dim={report.embedding_dim}, "
        f"corpus={report.corpus_size}, device={report.device}) ==="
    )
    lines.append("")
    recall_col = "@".join(("Recall", str(report.eval_k)))
    mrr_col = "@".join(("MRR", str(report.eval_k)))
    q_header = (
        f"{'Query Slice':<26} | {'Queries':>7} | "
        f"{recall_col:>10} | {mrr_col:>8}"
    )
    lines.append(q_header)
    lines.append("-" * len(q_header))
    rows = [
        ("Overall", report.overall),
        ("Symmetric Paraphrase", report.by_task["symmetric"]),
        ("Asymmetric Short Query", report.by_task["asymmetric"]),
    ]
    for label, metrics in rows:
        lines.append(
            f"{label:<26} | {metrics.query_count:>7d} | "
            f"{metrics.recall_at_k:>10.3f} | {metrics.mrr_at_k:>8.3f}"
        )

    lines.append("")
    t_header = (
        f"{'Execution Mode':<26} | {'Batch Total (ms)':>16} | "
        f"{'p50 / Query (ms)':>16} | {'Throughput (QPS)':>16}"
    )
    lines.append(t_header)
    lines.append("-" * len(t_header))
    for timing in report.timings:
        lines.append(
            f"{timing.mode:<26} | {timing.total_ms:>16.2f} | "
            f"{timing.per_query_p50_ms:>16.2f} | {timing.queries_per_second:>16.1f}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run bi-encoder semantic search quality and batch-throughput benchmark."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the validated BiEncoderBenchmarkReport as formatted JSON.",
    )
    args = parser.parse_args()
    report = run_benchmark()
    print(format_report_table(report))
    if args.json:
        print("\n" + json.dumps(report.model_dump(), indent=2))


if __name__ == "__main__":
    main()
