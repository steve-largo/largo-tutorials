"""Benchmark runner measuring MRR@10, Precision@1, and candidate-pool latency scaling."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

import numpy as np

from config import CrossEncoderSettings, load_settings
from reranker import (
    BiEncoderBackend,
    CrossEncoderScorer,
    MsMarcoCrossEncoder,
    SentenceTransformerBiEncoder,
    TwoStageReranker,
)
from schemas import (
    CrossEncoderBenchmarkReport,
    PassageRecord,
    PoolScaleResult,
    RerankEvalQuery,
)

_CURATED_ANCHOR_PASSAGES: tuple[PassageRecord, ...] = (
    PassageRecord(
        passage_id="py-pytorch-training-loop",
        topic="python-ml",
        title="Training PyTorch Neural Networks in Python with Mixed Precision",
        text=(
            "Write a Python training loop with torch.optim.AdamW, zero_grad(set_to_none=True), "
            "and GradScaler to train transformer and vision models on GPU hardware."
        ),
    ),
    PassageRecord(
        passage_id="zoology-reticulated-python",
        topic="python-ml",
        title="Tropical Habitat and Ambush Hunting Behavior of Python Snakes",
        text=(
            "The reticulated python inhabits humid Southeast Asian rainforests and riverbanks, "
            "relying on heat-sensing labial pits to track nocturnal mammal prey."
        ),
    ),
    PassageRecord(
        passage_id="py-beginner-syntax-guide",
        topic="python-ml",
        title="Python Variables, Loops, and String Formatting for First-Time Coders",
        text=(
            "Learn basic Python integers, list comprehensions, and f-strings before writing "
            "command-line scripts and automated file organizers."
        ),
    ),
    PassageRecord(
        passage_id="deploy-systemd-no-docker",
        topic="linux-deploy",
        title="Deploying Python FastAPI Services Directly with Systemd Without Containers",
        text=(
            "Run uvicorn workers natively on Debian hosts using systemd unit files, DynamicUser "
            "sandboxing, and UNIX domain sockets when Docker daemons are prohibited."
        ),
    ),
    PassageRecord(
        passage_id="deploy-docker-multistage",
        topic="linux-deploy",
        title="Building Multi-Stage Docker Images for FastAPI Production Releases",
        text=(
            "Compile wheels inside a builder container stage and copy the virtual environment "
            "into a slim Docker runtime image for Kubernetes pod rollouts."
        ),
    ),
    PassageRecord(
        passage_id="gpu-budget-l4-24gb",
        topic="gpu-sizing",
        title="Serving Quantized 8B Models Under a 24 GB Single-GPU VRAM Budget",
        text=(
            "An NVIDIA L4 with 24 GB VRAM hosts INT8 or FP8 8B instruction models alongside "
            "16K paged KV-cache blocks at sub-dollar hourly cloud instance rates."
        ),
    ),
    PassageRecord(
        passage_id="gpu-cluster-h100-80gb",
        topic="gpu-sizing",
        title="Multi-Node 8x H100 80 GB Tensor-Parallel Serving for 405B Frontier Models",
        text=(
            "Shard dense 405B parameter weights across eight 80 GB H100 SXM GPUs interconnected "
            "with 900 GB/s NVLink fabrics for high-concurrency enterprise traffic."
        ),
    ),
    PassageRecord(
        passage_id="lockfree-ring-buffer-queue",
        topic="concurrency",
        title="Wait-Free Single-Producer Ring Buffers Without Mutex Locks",
        text=(
            "Use acquire-release atomic head and tail indices over a cache-line-padded ring "
            "buffer to pass telemetry events between threads without OS mutex contention."
        ),
    ),
    PassageRecord(
        passage_id="mutex-condition-variable-pool",
        topic="concurrency",
        title="Coordinating Worker Threads with POSIX Mutex Locks and Condition Variables",
        text=(
            "Acquire a pthread mutex lock before inspecting shared queue predicates and sleep "
            "on a condition variable until producer threads broadcast new work."
        ),
    ),
    PassageRecord(
        passage_id="pg-lock-timeout-55p03",
        topic="database-ops",
        title="Mitigating PostgreSQL Lock Not Available Errors (SQLSTATE 55P03) During DDL",
        text=(
            "Set a short lock_timeout before ALTER TABLE statements and retry with backoff so "
            "schema migrations fail fast with 55P03 instead of queuing behind long reads."
        ),
    ),
    PassageRecord(
        passage_id="pg-deadlock-40p01",
        topic="database-ops",
        title="Resolving PostgreSQL Deadlock Detected Aborts (SQLSTATE 40P01)",
        text=(
            "Enforce a canonical primary-key ordering across multi-row UPDATE transactions so "
            "concurrent workers never form circular row-lock wait graphs."
        ),
    ),
    PassageRecord(
        passage_id="tls-mtls-client-cert-rotation",
        topic="security-net",
        title="Zero-Downtime mTLS Client Certificate Rotation with Dual Trust Bundles",
        text=(
            "Stage both the expiring and successor CA roots in the server trust bundle before "
            "rolling workload client certificates so active mutual TLS handshakes never drop."
        ),
    ),
    PassageRecord(
        passage_id="tls-public-acme-http01",
        topic="security-net",
        title="Automating Public Edge TLS Certificates with ACME HTTP-01 Challenges",
        text=(
            "Respond to Let's Encrypt HTTP-01 token verification requests on port 80 to renew "
            "public browser-facing server certificates every sixty days."
        ),
    ),
    PassageRecord(
        passage_id="k8s-oomkill-exit-137",
        topic="k8s-runtime",
        title="Diagnosing Kubernetes Pod Cgroup Memory Evictions (Exit Code 137)",
        text=(
            "When container RSS plus page cache exceeds resources.limits.memory, the Linux "
            "kernel OOM killer sends SIGKILL resulting in termination exit code 137."
        ),
    ),
    PassageRecord(
        passage_id="k8s-segfault-exit-139",
        topic="k8s-runtime",
        title="Investigating Native C++ Extension Crashes in Pods (Exit Code 139)",
        text=(
            "Exit code 139 indicates fatal signal 11 SIGSEGV from invalid pointer dereferences "
            "or ABI mismatches inside compiled shared libraries."
        ),
    ),
    PassageRecord(
        passage_id="rag-cross-encoder-truncation",
        topic="search-ir",
        title="Handling 512-Token Sequence Truncation in Cross-Encoder Rerankers",
        text=(
            "BERT-based MS MARCO cross-encoders cap combined query and passage length at 512 "
            "tokens, requiring passage windowing or sliding max-score aggregation on long docs."
        ),
    ),
    PassageRecord(
        passage_id="rag-bi-encoder-precompute",
        topic="search-ir",
        title="Offline Document Embedding Precomputation for Bi-Encoder Indexes",
        text=(
            "Encode millions of static corpus chunks asynchronously on batch GPUs and load the "
            "resulting float32 vectors into an approximate nearest neighbor graph."
        ),
    ),
    PassageRecord(
        passage_id="http-idempotency-keys-post",
        topic="api-design",
        title="Preventing Duplicate Payment Charges with Idempotency-Key Request Headers",
        text=(
            "Persist client-supplied UUID Idempotency-Key records atomically alongside ledger "
            "mutations so network client retries replay the original response payload."
        ),
    ),
    PassageRecord(
        passage_id="http-rate-limit-token-bucket",
        topic="api-design",
        title="Enforcing Per-Tenant API Burst Quotas with Redis Token Buckets",
        text=(
            "Evaluate refill timestamps and remaining token balances inside an atomic Lua "
            "script to reject excess tenant bursts with HTTP 429 Retry-After headers."
        ),
    ),
    PassageRecord(
        passage_id="spark-skewed-join-salting",
        topic="data-pipelines",
        title="Eliminating Straggler Tasks in Skewed Shuffle Joins via Key Salting",
        text=(
            "Append uniform random integer suffixes to hot partition keys on the fact table "
            "and replicate dimension rows across matching salt buckets."
        ),
    ),
    PassageRecord(
        passage_id="spark-parquet-predicate-pushdown",
        topic="data-pipelines",
        title="Pruning Cloud Object Reads with Parquet Row-Group Min-Max Statistics",
        text=(
            "Sort analytical tables on filter columns before writing Parquet files so query "
            "engines skip irrelevant row groups using footer metadata alone."
        ),
    ),
    PassageRecord(
        passage_id="sqlite-wal-single-host",
        topic="embedded-storage",
        title="Embedded Single-Node Read Concurrency with SQLite Write-Ahead Logging",
        text=(
            "Enable PRAGMA journal_mode=WAL and busy_timeout=5000 so local readers proceed "
            "concurrently without deploying an external networked database server."
        ),
    ),
    PassageRecord(
        passage_id="distributed-raft-quorum-writes",
        topic="embedded-storage",
        title="Cross-Zone Replicated State Machines with Raft Majority Quorums",
        text=(
            "Replicate log entries across three availability zones and wait for majority "
            "follower disk fsync acknowledgments before committing state transitions."
        ),
    ),
    PassageRecord(
        passage_id="onnx-cpu-int8-quantization",
        topic="edge-inference",
        title="Sub-20ms CPU Transformer Inference with ONNX Runtime Dynamic INT8 Quantization",
        text=(
            "Quantize linear projection weights to signed 8-bit integers in ONNX Runtime so "
            "8-core Xeon and ARM servers score batches without discrete GPU accelerators."
        ),
    ),
)

_BACKGROUND_TOPICS: tuple[tuple[str, str, str], ...] = (
    (
        "python-ml",
        "Python Data Processing Note",
        "Vectorized array broadcasting, generator pipelines, and typed dataclasses reduce interpreter overhead across batch analytics workloads.",
    ),
    (
        "linux-deploy",
        "Linux Service Operations Note",
        "Configure logrotate retention limits, journald rate thresholds, and graceful SIGTERM drain timeouts for stateless HTTP worker fleets.",
    ),
    (
        "gpu-sizing",
        "Accelerator Capacity Planning Note",
        "Profile activation tensor footprints, PCIe host-to-device transfer bandwidth, and CUDA graph capture warmup across batch sizes.",
    ),
    (
        "concurrency",
        "Systems Threading Architecture Note",
        "Pin latency-sensitive event loops to dedicated NUMA cores and size bounded backpressure queues to prevent memory exhaustion.",
    ),
    (
        "database-ops",
        "Relational Storage Maintenance Note",
        "Schedule autovacuum scale factors, monitor transaction ID wraparound horizons, and verify B-tree index bloat ratios weekly.",
    ),
    (
        "security-net",
        "Zero-Trust Network Transport Note",
        "Pin TLS 1.3 cipher suites, enforce short-lived workload identities, and audit ingress proxy access logs for anomalous TLS alerts.",
    ),
    (
        "k8s-runtime",
        "Cluster Pod Scheduling Note",
        "Define PodDisruptionBudgets, topologySpreadConstraints across failure domains, and startup probes for slow-initializing containers.",
    ),
    (
        "search-ir",
        "Information Retrieval Indexing Note",
        "Track recall at candidate cutoff alongside reciprocal rank and calibrate chunk overlap boundaries on technical documentation.",
    ),
    (
        "api-design",
        "HTTP Service Contract Note",
        "Validate JSON request payloads at the transport boundary, return RFC 9457 problem details, and propagate W3C traceparent headers.",
    ),
    (
        "data-pipelines",
        "Columnar Lakehouse Execution Note",
        "Compact small streaming files into 256 MB Parquet partitions and validate schema evolution rules in continuous integration.",
    ),
    (
        "embedded-storage",
        "Local Storage Engine Note",
        "Verify checksum integrity on page reads, bound memory-mapped cache sizes, and benchmark fsync durability latencies on NVMe media.",
    ),
    (
        "edge-inference",
        "Low-Latency Model Runtime Note",
        "Pre-allocate static input tensor buffers, pin host memory pages, and warm thread pools prior to accepting live inference traffic.",
    ),
)


def load_corpus() -> tuple[PassageRecord, ...]:
    """Assemble 120 unique passages (24 curated disambiguation targets + 96 domain passages)."""
    records: list[PassageRecord] = list(_CURATED_ANCHOR_PASSAGES)
    for topic_slug, title_prefix, base_text in _BACKGROUND_TOPICS:
        for variant_idx in range(1, 9):
            records.append(
                PassageRecord(
                    passage_id=f"bg-{topic_slug}-{variant_idx:02d}",
                    topic=topic_slug,
                    title=f"{title_prefix} #{variant_idx}: {topic_slug.replace('-', ' ').title()}",
                    text=(
                        f"{base_text} Reference profile {variant_idx} documents operational "
                        f"thresholds for batch window {variant_idx * 16} in {topic_slug} systems."
                    ),
                )
            )
    return tuple(records)


def load_eval_queries() -> tuple[RerankEvalQuery, ...]:
    """Return 12 evaluation queries spanning polysemy, negation, constraints, and error codes."""
    return (
        RerankEvalQuery(
            query_id="q-polysemy-python-pytorch",
            phenomenon="polysemy",
            query_text="python neural network training loop with mixed precision on GPU",
            primary_passage_id="py-pytorch-training-loop",
            relevant_passage_ids=("py-pytorch-training-loop",),
        ),
        RerankEvalQuery(
            query_id="q-polysemy-python-reptile",
            phenomenon="polysemy",
            query_text="how python snakes hunt nocturnal prey in tropical rainforests",
            primary_passage_id="zoology-reticulated-python",
            relevant_passage_ids=("zoology-reticulated-python",),
        ),
        RerankEvalQuery(
            query_id="q-negation-no-docker",
            phenomenon="negation",
            query_text="deploying FastAPI on Linux hosts without Docker containers",
            primary_passage_id="deploy-systemd-no-docker",
            relevant_passage_ids=("deploy-systemd-no-docker",),
        ),
        RerankEvalQuery(
            query_id="q-negation-without-mutex",
            phenomenon="negation",
            query_text="passing events between threads without mutex locks",
            primary_passage_id="lockfree-ring-buffer-queue",
            relevant_passage_ids=("lockfree-ring-buffer-queue",),
        ),
        RerankEvalQuery(
            query_id="q-negation-no-external-db",
            phenomenon="negation",
            query_text="concurrent local database reads without running a networked server",
            primary_passage_id="sqlite-wal-single-host",
            relevant_passage_ids=("sqlite-wal-single-host",),
        ),
        RerankEvalQuery(
            query_id="q-constraint-24gb-vram",
            phenomenon="constraint",
            query_text="serving quantized 8B LLM under a single 24 GB GPU memory budget",
            primary_passage_id="gpu-budget-l4-24gb",
            relevant_passage_ids=("gpu-budget-l4-24gb",),
        ),
        RerankEvalQuery(
            query_id="q-constraint-cpu-only-int8",
            phenomenon="constraint",
            query_text="sub-20ms transformer inference on CPU servers without a GPU",
            primary_passage_id="onnx-cpu-int8-quantization",
            relevant_passage_ids=("onnx-cpu-int8-quantization",),
        ),
        RerankEvalQuery(
            query_id="q-constraint-512-token-cap",
            phenomenon="constraint",
            query_text="scoring long documents when cross-encoder models truncate at 512 tokens",
            primary_passage_id="rag-cross-encoder-truncation",
            relevant_passage_ids=("rag-cross-encoder-truncation",),
        ),
        RerankEvalQuery(
            query_id="q-spec-pg-55p03",
            phenomenon="technical_specificity",
            query_text="postgres ALTER TABLE failing fast with SQLSTATE 55P03 lock_timeout",
            primary_passage_id="pg-lock-timeout-55p03",
            relevant_passage_ids=("pg-lock-timeout-55p03",),
        ),
        RerankEvalQuery(
            query_id="q-spec-k8s-exit-137",
            phenomenon="technical_specificity",
            query_text="kubernetes container killed with exit code 137 cgroup memory limit",
            primary_passage_id="k8s-oomkill-exit-137",
            relevant_passage_ids=("k8s-oomkill-exit-137",),
        ),
        RerankEvalQuery(
            query_id="q-spec-mtls-cert-rotation",
            phenomenon="technical_specificity",
            query_text="rotating mutual TLS workload client certificates with dual CA roots",
            primary_passage_id="tls-mtls-client-cert-rotation",
            relevant_passage_ids=("tls-mtls-client-cert-rotation",),
        ),
        RerankEvalQuery(
            query_id="q-spec-idempotency-key",
            phenomenon="technical_specificity",
            query_text="preventing duplicate POST payment charges on client network retries",
            primary_passage_id="http-idempotency-keys-post",
            relevant_passage_ids=("http-idempotency-keys-post",),
        ),
    )


def compute_mrr_at_k(ranked_ids: Sequence[str], primary_id: str, k: int = 10) -> float:
    """Reciprocal rank of the primary relevant passage within top-k."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    for rank, pid in enumerate(ranked_ids[:k], start=1):
        if pid == primary_id:
            return 1.0 / rank
    return 0.0


def compute_precision_at_1(ranked_ids: Sequence[str], relevant_ids: Sequence[str]) -> float:
    """Return 1.0 if the rank-1 result belongs to relevant_ids, else 0.0."""
    if not ranked_ids:
        return 0.0
    return 1.0 if ranked_ids[0] in set(relevant_ids) else 0.0


def compute_pool_recall(candidate_ids: Sequence[str], relevant_ids: Sequence[str]) -> float:
    """Fraction of ground-truth relevant passages captured inside the stage-1 candidate pool."""
    rel = set(relevant_ids)
    if not rel:
        return 0.0
    hits = sum(1 for pid in candidate_ids if pid in rel)
    return hits / len(rel)


def run_benchmark(
    settings: CrossEncoderSettings | None = None,
    bi_encoder: BiEncoderBackend | None = None,
    cross_encoder: CrossEncoderScorer | None = None,
    passages: Sequence[PassageRecord] | None = None,
    queries: Sequence[RerankEvalQuery] | None = None,
) -> CrossEncoderBenchmarkReport:
    """Evaluate stage-1 bi-encoder baseline vs two-stage cross-encoder across candidate pools."""
    cfg = settings or load_settings()
    np.random.seed(cfg.seed)

    corpus = tuple(passages) if passages is not None else load_corpus()
    eval_queries = tuple(queries) if queries is not None else load_eval_queries()

    active_bi = bi_encoder or SentenceTransformerBiEncoder(
        model_name=cfg.bi_encoder_model,
        device=cfg.device,
    )
    active_ce = cross_encoder or MsMarcoCrossEncoder(
        model_name=cfg.cross_encoder_model,
        device=cfg.device,
    )

    engine = TwoStageReranker(bi_encoder=active_bi, cross_encoder=active_ce)
    engine.index_passages(corpus, batch_size=cfg.rerank_batch_size)

    for warm_q in eval_queries[: cfg.warmup_queries]:
        engine.search(
            warm_q.query_text,
            top_k=cfg.eval_top_k,
            candidate_pool_size=cfg.candidate_pools[0],
            batch_size=cfg.rerank_batch_size,
            fusion_alpha=cfg.fusion_alpha,
        )

    results: list[PoolScaleResult] = []

    bi_mrrs: list[float] = []
    bi_p1s: list[float] = []
    bi_recalls: list[float] = []
    bi_latencies: list[float] = []

    for repeat_idx in range(cfg.latency_repeats):
        for q in eval_queries:
            diag = engine.search_bi_encoder_only(q.query_text, top_k=cfg.eval_top_k)
            bi_latencies.append(diag.retrieval_ms)
            if repeat_idx == 0:
                ranked_ids = [r.passage_id for r in diag.results]
                bi_mrrs.append(compute_mrr_at_k(ranked_ids, q.primary_passage_id, k=cfg.eval_top_k))
                bi_p1s.append(compute_precision_at_1(ranked_ids, q.relevant_passage_ids))
                bi_recalls.append(compute_pool_recall(ranked_ids, q.relevant_passage_ids))

    bi_p50 = round(float(np.percentile(bi_latencies, 50)), 2)
    results.append(
        PoolScaleResult(
            pipeline_label=f"Stage-1 Bi-Encoder Only (top-{cfg.eval_top_k})",
            candidate_pool_size=0,
            mrr_at_10=round(float(np.mean(bi_mrrs)), 4),
            precision_at_1=round(float(np.mean(bi_p1s)), 4),
            recall_at_pool=round(float(np.mean(bi_recalls)), 4),
            retrieval_p50_ms=bi_p50,
            rerank_p50_ms=0.0,
            total_p50_ms=bi_p50,
            pairs_per_second=0.0,
        )
    )

    for pool_size in cfg.candidate_pools:
        mrrs: list[float] = []
        p1s: list[float] = []
        pool_recalls: list[float] = []
        ret_ms_list: list[float] = []
        rerank_ms_list: list[float] = []
        total_ms_list: list[float] = []
        total_pairs_scored = 0
        total_rerank_sec = 0.0

        for repeat_idx in range(cfg.latency_repeats):
            for q in eval_queries:
                full_diag = engine.search(
                    q.query_text,
                    top_k=max(pool_size, cfg.eval_top_k),
                    candidate_pool_size=pool_size,
                    batch_size=cfg.rerank_batch_size,
                    fusion_alpha=cfg.fusion_alpha,
                )
                ret_ms_list.append(full_diag.retrieval_ms)
                rerank_ms_list.append(full_diag.rerank_ms)
                total_ms_list.append(full_diag.total_ms)
                total_pairs_scored += full_diag.candidates_reranked
                total_rerank_sec += full_diag.rerank_ms / 1000.0

                if repeat_idx == 0:
                    all_pool_ids = [r.passage_id for r in full_diag.results]
                    top_ids = all_pool_ids[: cfg.eval_top_k]
                    mrrs.append(compute_mrr_at_k(top_ids, q.primary_passage_id, k=cfg.eval_top_k))
                    p1s.append(compute_precision_at_1(top_ids, q.relevant_passage_ids))
                    pool_recalls.append(compute_pool_recall(all_pool_ids, q.relevant_passage_ids))

        pairs_per_sec = (
            round(total_pairs_scored / total_rerank_sec, 1)
            if total_rerank_sec > 1e-9
            else 0.0
        )
        results.append(
            PoolScaleResult(
                pipeline_label=f"Two-Stage Cross-Encoder (top-{pool_size})",
                candidate_pool_size=pool_size,
                mrr_at_10=round(float(np.mean(mrrs)), 4),
                precision_at_1=round(float(np.mean(p1s)), 4),
                recall_at_pool=round(float(np.mean(pool_recalls)), 4),
                retrieval_p50_ms=round(float(np.percentile(ret_ms_list, 50)), 2),
                rerank_p50_ms=round(float(np.percentile(rerank_ms_list, 50)), 2),
                total_p50_ms=round(float(np.percentile(total_ms_list, 50)), 2),
                pairs_per_second=pairs_per_sec,
            )
        )

    return CrossEncoderBenchmarkReport(
        bi_encoder_model=cfg.bi_encoder_model,
        cross_encoder_model=cfg.cross_encoder_model,
        device=cfg.device,
        corpus_size=len(corpus),
        query_count=len(eval_queries),
        rerank_batch_size=cfg.rerank_batch_size,
        results=tuple(results),
    )


def format_report_table(report: CrossEncoderBenchmarkReport) -> str:
    """Format an ASCII comparison table across candidate pool sizes."""
    lines: list[str] = []
    lines.append(
        f"=== Cross-Encoder Reranking & Pool Scaling Benchmark "
        f"({report.corpus_size} passages, {report.query_count} queries, device={report.device}) ==="
    )
    lines.append(
        f"Models: bi_encoder={report.bi_encoder_model} | "
        f"cross_encoder={report.cross_encoder_model} | batch_size={report.rerank_batch_size}"
    )
    lines.append("")
    header = (
        f"{'Pipeline':<38} | {'MRR@10':>6} | {'P@1':>5} | {'Pool Rec':>8} | "
        f"{'Ret p50':>7} | {'Rerank p50':>10} | {'Total p50':>9} | {'Pairs/s':>8}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for row in report.results:
        lines.append(
            f"{row.pipeline_label:<38} | {row.mrr_at_10:>6.3f} | {row.precision_at_1:>5.3f} | "
            f"{row.recall_at_pool:>8.3f} | {row.retrieval_p50_ms:>7.2f} | "
            f"{row.rerank_p50_ms:>10.2f} | {row.total_p50_ms:>9.2f} | {row.pairs_per_second:>8.1f}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run two-stage cross-encoder reranking quality and candidate-pool scaling benchmark."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the full CrossEncoderBenchmarkReport as formatted JSON.",
    )
    args = parser.parse_args()
    report = run_benchmark()
    print(format_report_table(report))
    if args.json:
        print("\n" + json.dumps(report.model_dump(), indent=2))


if __name__ == "__main__":
    main()
