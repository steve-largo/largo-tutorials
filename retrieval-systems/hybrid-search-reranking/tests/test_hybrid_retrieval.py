"""Unit, mathematical invariant, and end-to-end tests for two-stage hybrid retrieval."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
from pydantic import ValidationError

from benchmark import (
    compute_mrr_at_k,
    compute_ndcg_at_k,
    compute_recall_at_k,
    run_benchmark,
)
from config import RetrievalSettings
from corpus import load_corpus, load_eval_queries
from retrievers import (
    BM25LexicalRetriever,
    DenseBiEncoderRetriever,
    HybridRRFRetriever,
    TwoStageHybridRetriever,
    reciprocal_rank_fusion,
    tokenize_lexical,
)
from schemas import CorpusChunk, EvalQuery, ScoredCandidate


class _KeywordAxisEncoder:
    """Deterministic unit-test encoder that embeds texts onto orthogonal concept axes."""

    _AXES: tuple[tuple[str, ...], ...] = (
        ("retry", "backoff", "jitter", "canary", "storms", "recovering"),
        ("prefill", "streaming", "decode", "slices", "prompts"),
        ("ssl", "tls", "err_ssl_protocol_error", "443"),
    )

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        rows: list[np.ndarray] = []
        for text in texts:
            lowered = text.lower()
            vec = np.array(
                [
                    sum(1.0 for kw in axis if kw in lowered) + 1e-3
                    for axis in self._AXES
                ],
                dtype=np.float32,
            )
            vec /= np.linalg.norm(vec)
            rows.append(vec)
        return np.vstack(rows)


class _ExactPhraseReranker:
    """Deterministic unit-test cross-encoder rewarding joint query-passage cues."""

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        scores: list[float] = []
        for query, passage in pairs:
            q_low = query.lower()
            p_low = passage.lower()
            bonus = 0.0
            if "retry storms" in q_low and "decorrelated" in p_low:
                bonus += 8.5
            if "err_ssl_protocol_error" in q_low and "err_ssl_protocol_error" in p_low:
                bonus += 9.0
            overlap = len(set(tokenize_lexical(q_low)) & set(tokenize_lexical(p_low)))
            scores.append(bonus + 0.25 * overlap)
        return np.asarray(scores, dtype=np.float32)


def test_settings_and_schema_validation() -> None:
    with pytest.raises(ValidationError):
        RetrievalSettings(bi_encoder_model="../etc/passwd")

    with pytest.raises(ValidationError):
        RetrievalSettings(device="tpu-v99")

    with pytest.raises(ValidationError):
        RetrievalSettings(stage1_top_k=10, rerank_top_k=15)

    with pytest.raises(ValidationError):
        EvalQuery(
            query_id="bad_grades",
            query_slice="exact_id",
            query_text="missing primary grade target",
            relevant_grades={"tls-alpn-record-mismatch": 1},
        )

    cfg = RetrievalSettings(rrf_k=60, stage1_top_k=25, rerank_top_k=5)
    with pytest.raises(ValidationError):
        cfg.rrf_k = 10  # type: ignore[misc]


def test_corpus_and_query_referential_integrity() -> None:
    chunks = load_corpus()
    queries = load_eval_queries()

    assert len(chunks) == 48
    chunk_ids = [c.chunk_id for c in chunks]
    assert len(chunk_ids) == len(set(chunk_ids))

    assert len(queries) == 28
    query_ids = [q.query_id for q in queries]
    assert len(query_ids) == len(set(query_ids))

    exact_queries = [q for q in queries if q.query_slice == "exact_id"]
    sem_queries = [q for q in queries if q.query_slice == "semantic_paraphrase"]
    assert len(exact_queries) == 14
    assert len(sem_queries) == 14

    valid_chunk_set = set(chunk_ids)
    for q in queries:
        for target_id in q.relevant_grades:
            assert target_id in valid_chunk_set, f"Unknown chunk {target_id} in {q.query_id}"


def test_lexical_tokenizer_and_bm25_exact_identifiers() -> None:
    tokens = tokenize_lexical(
        "Tune --max-num-batched-tokens on g5.12xlarge for net.ipv4.tcp_tw_reuse"
    )
    assert "max-num-batched-tokens" in tokens
    assert "g5.12xlarge" in tokens
    assert "net.ipv4.tcp_tw_reuse" in tokens

    bm25 = BM25LexicalRetriever(load_corpus())
    ssl_hits = bm25.search("ERR_SSL_PROTOCOL_ERROR port 443", top_k=5)
    assert ssl_hits[0].chunk_id == "tls-alpn-record-mismatch"

    sku_hits = bm25.search("g5.12xlarge PCIe interconnect", top_k=5)
    assert sku_hits[0].chunk_id == "gpu-sku-g5-12xlarge"


def test_reciprocal_rank_fusion_scale_invariance() -> None:
    lex_normal = [
        ScoredCandidate(chunk_id="doc-only-lex", rank=1, score=18.4, bm25_rank=1),
        ScoredCandidate(chunk_id="doc-consensus", rank=2, score=11.2, bm25_rank=2),
    ]
    dense_normal = [
        ScoredCandidate(chunk_id="doc-only-dense", rank=1, score=0.81, dense_rank=1),
        ScoredCandidate(chunk_id="doc-consensus", rank=2, score=0.79, dense_rank=2),
    ]

    lex_rescaled = [
        ScoredCandidate(chunk_id="doc-only-lex", rank=1, score=18400.0, bm25_rank=1),
        ScoredCandidate(chunk_id="doc-consensus", rank=2, score=11200.0, bm25_rank=2),
    ]
    dense_shifted = [
        ScoredCandidate(chunk_id="doc-only-dense", rank=1, score=-0.19, dense_rank=1),
        ScoredCandidate(chunk_id="doc-consensus", rank=2, score=-0.21, dense_rank=2),
    ]

    fused_a = reciprocal_rank_fusion(lex_normal, dense_normal, rrf_k=60, top_k=3)
    fused_b = reciprocal_rank_fusion(lex_rescaled, dense_shifted, rrf_k=60, top_k=3)

    assert [c.chunk_id for c in fused_a] == [c.chunk_id for c in fused_b]
    assert [c.score for c in fused_a] == pytest.approx([c.score for c in fused_b])
    assert fused_a[0].chunk_id == "doc-consensus"
    assert fused_a[0].score == pytest.approx(2.0 / 62.0)


def test_information_retrieval_metrics_exact_math() -> None:
    grades = {"primary-doc": 2, "secondary-doc": 1}
    ranked_suboptimal = ["distractor-a", "secondary-doc", "primary-doc", "distractor-b"]
    ranked_ideal = ["primary-doc", "secondary-doc", "distractor-a", "distractor-b"]

    assert compute_recall_at_k(ranked_suboptimal, grades, k=2) == pytest.approx(0.5)
    assert compute_recall_at_k(ranked_suboptimal, grades, k=3) == pytest.approx(1.0)
    assert compute_mrr_at_k(ranked_suboptimal, grades, k=10) == pytest.approx(1.0 / 3.0)
    assert compute_ndcg_at_k(ranked_ideal, grades, k=10) == pytest.approx(1.0)
    assert 0.5 < compute_ndcg_at_k(ranked_suboptimal, grades, k=10) < 0.75


def test_two_stage_pipeline_wiring_with_injected_fakes() -> None:
    mini_chunks = (
        CorpusChunk(
            chunk_id="tls-alpn-record-mismatch",
            domain="tls-ingress",
            title="Diagnosing ERR_SSL_PROTOCOL_ERROR on Port 443",
            text="Clients surface ERR_SSL_PROTOCOL_ERROR when plaintext bytes hit a TLS port.",
        ),
        CorpusChunk(
            chunk_id="http-jitter-circuit-breaker",
            domain="http-resilience",
            title="Decorrelated Backoff Jitter and Half-Open Canary Probing",
            text="Decorrelated uniform backoff intervals disperse traffic across convalescing tiers.",
        ),
        CorpusChunk(
            chunk_id="vllm-chunked-prefill-sched",
            domain="llm-serving",
            title="Iteration-Level Chunked Prefill Scheduling",
            text="Fixed-size prefill slices interleave with decode steps to protect streaming.",
        ),
    )
    bm25 = BM25LexicalRetriever(mini_chunks)
    dense = DenseBiEncoderRetriever(mini_chunks, encoder=_KeywordAxisEncoder())
    hybrid = HybridRRFRetriever(bm25, dense, rrf_k=60, stage1_top_k=3)
    two_stage = TwoStageHybridRetriever(
        chunks=mini_chunks,
        stage1=hybrid,
        reranker=_ExactPhraseReranker(),
        stage1_top_k=3,
        rerank_top_k=2,
    )

    hits = two_stage.search("stopping client retry storms on a recovering backend")
    assert len(hits) == 2
    assert hits[0].chunk_id == "http-jitter-circuit-breaker"
    assert hits[0].rerank_score is not None


def test_end_to_end_four_stage_benchmark_invariants() -> None:
    settings = RetrievalSettings(warmup_queries=1, latency_repeats=1)
    report = run_benchmark(settings=settings)

    by_id = {stage.pipeline_id: stage for stage in report.stages}
    bm25 = by_id["bm25_only"]
    dense = by_id["dense_only"]
    rrf = by_id["hybrid_rrf"]
    reranked = by_id["hybrid_rrf_rerank"]

    assert bm25.by_slice["exact_id"].ndcg_at_10 > bm25.by_slice["semantic_paraphrase"].ndcg_at_10
    assert dense.by_slice["semantic_paraphrase"].recall_at_5 > bm25.by_slice["semantic_paraphrase"].recall_at_5
    assert rrf.overall.recall_at_25 >= max(bm25.overall.recall_at_25, dense.overall.recall_at_25)
    assert reranked.overall.ndcg_at_10 >= rrf.overall.ndcg_at_10
    assert reranked.overall.mrr_at_10 >= rrf.overall.mrr_at_10
