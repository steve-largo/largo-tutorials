"""Deterministic offline unit and invariant tests for two-stage cross-encoder reranking."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
from pydantic import ValidationError

from benchmark import (
    compute_mrr_at_k,
    compute_pool_recall,
    compute_precision_at_1,
    load_corpus,
    load_eval_queries,
    run_benchmark,
)
from config import CrossEncoderSettings
from reranker import TwoStageReranker, blend_relevance_scores, sigmoid_probability
from schemas import PassageRecord, RerankEvalQuery


class _CoarseBagEncoder:
    """Coarse stage-1 fake encoder conflating shared topic nouns (simulating bi-encoder limits)."""

    _AXES: tuple[tuple[str, ...], ...] = (
        ("python", "pytorch", "snakes", "rainforests", "coders"),
        ("fastapi", "docker", "containers", "systemd", "linux"),
        ("gpu", "vram", "24 gb", "80 gb", "h100", "8b", "int8", "cpu"),
        ("mutex", "locks", "threads", "ring", "atomic"),
        ("postgres", "postgresql", "55p03", "40p01", "lock_timeout", "deadlock"),
        ("tls", "mtls", "certificates", "acme", "ca"),
        ("kubernetes", "exit", "137", "139", "cgroup"),
        ("cross-encoder", "512", "tokens", "bi-encoder"),
        ("idempotency", "payment", "post", "retries", "429"),
        ("sqlite", "wal", "local", "networked", "raft", "quorums"),
    )

    def encode_texts(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        assert batch_size >= 1
        rows: list[np.ndarray] = []
        for idx, text in enumerate(texts):
            low = text.lower()
            vec = np.array(
                [sum(1.0 for kw in axis if kw in low) + 1e-3 for axis in self._AXES],
                dtype=np.float32,
            )
            if "tropical" in low or "docker images" in low or "mutex locks and" in low:
                vec *= 1.15
            vec[idx % len(self._AXES)] += 1e-4
            rows.append(vec / np.linalg.norm(vec))
        return np.vstack(rows)


class _JointTokenInteractionScorer:
    """Deterministic cross-encoder fake modeling joint query-passage token interactions."""

    def __init__(self) -> None:
        self.call_pair_counts: list[int] = []

    def score_pairs(
        self,
        pairs: Sequence[tuple[str, str]],
        batch_size: int = 32,
    ) -> np.ndarray:
        assert batch_size >= 1
        self.call_pair_counts.append(len(pairs))
        logits: list[float] = []
        for query, passage in pairs:
            q = query.lower()
            p = passage.lower()
            score = -2.0
            if "neural network" in q and "torch.optim" in p:
                score += 10.5
            if "neural network" in q and "rainforests" in p:
                score -= 6.0
            if "snakes" in q and "rainforests" in p:
                score += 10.5
            if "without docker" in q and "systemd" in p:
                score += 9.8
            if "without docker" in q and "multi-stage docker" in p:
                score -= 5.0
            if "without mutex" in q and "wait-free" in p:
                score += 9.5
            if "without running a networked" in q and "sqlite" in p:
                score += 9.5
            if "24 gb" in q and "24 gb" in p:
                score += 9.2
            if "without a gpu" in q and "onnx runtime" in p:
                score += 9.4
            if "512 tokens" in q and "512" in p:
                score += 9.1
            if "55p03" in q and "55p03" in p:
                score += 10.0
            if "137" in q and "137" in p:
                score += 10.0
            if "dual ca" in q and "dual trust" in p:
                score += 9.6
            if "payment" in q and "idempotency-key" in p:
                score += 9.7
            logits.append(score)
        return np.asarray(logits, dtype=np.float32)


def test_schema_and_settings_validation() -> None:
    with pytest.raises(ValidationError):
        CrossEncoderSettings(cross_encoder_model="../local/traversal")

    with pytest.raises(ValidationError):
        CrossEncoderSettings(candidate_pools=(50, 10))

    with pytest.raises(ValidationError):
        CrossEncoderSettings(fusion_alpha=1.5)

    with pytest.raises(ValidationError):
        RerankEvalQuery(
            query_id="mismatch-primary",
            phenomenon="polysemy",
            query_text="valid query text here",
            primary_passage_id="py-pytorch-training-loop",
            relevant_passage_ids=("zoology-reticulated-python",),
        )


def test_sigmoid_and_score_fusion_numerical_stability() -> None:
    assert sigmoid_probability(0.0) == pytest.approx(0.5)
    assert sigmoid_probability(500.0) == pytest.approx(1.0)
    assert sigmoid_probability(-500.0) == pytest.approx(0.0)

    assert blend_relevance_scores(0.65, 8.2, alpha=0.0) == pytest.approx(8.2)
    assert blend_relevance_scores(0.65, 8.2, alpha=1.0) == pytest.approx(0.65)
    blended = blend_relevance_scores(0.50, 0.0, alpha=0.2)
    assert blended == pytest.approx(0.2 * 0.50 + 0.8 * 0.50)


def test_cross_encoder_promotes_disambiguated_target_over_bi_encoder_trap() -> None:
    mini_corpus = (
        PassageRecord(
            passage_id="deploy-docker-multistage",
            topic="linux-deploy",
            title="Building Multi-Stage Docker Images for FastAPI Production Releases",
            text="Compile wheels in a container stage and copy into a slim Docker runtime image.",
        ),
        PassageRecord(
            passage_id="deploy-systemd-no-docker",
            topic="linux-deploy",
            title="Deploying Python FastAPI Services Directly with Systemd Without Containers",
            text="Run uvicorn natively on Linux hosts with systemd when Docker is prohibited.",
        ),
    )
    engine = TwoStageReranker(
        bi_encoder=_CoarseBagEncoder(),
        cross_encoder=_JointTokenInteractionScorer(),
    )
    engine.index_passages(mini_corpus)

    query = "deploying FastAPI on Linux hosts without Docker containers"
    bi_only = engine.search_bi_encoder_only(query, top_k=2)
    reranked = engine.search(query, top_k=2, candidate_pool_size=2)

    assert bi_only.results[0].passage_id == "deploy-docker-multistage"
    assert reranked.results[0].passage_id == "deploy-systemd-no-docker"
    assert reranked.results[0].bi_encoder_rank == 2
    assert reranked.results[0].cross_encoder_logit is not None
    assert reranked.results[0].cross_encoder_logit > reranked.results[1].cross_encoder_logit  # type: ignore[operator]


def test_candidate_pool_scaling_and_end_to_end_benchmark() -> None:
    corpus = load_corpus()
    queries = load_eval_queries()
    assert len(corpus) == 120
    assert len({p.passage_id for p in corpus}) == 120
    assert len(queries) == 12

    corpus_ids = {p.passage_id for p in corpus}
    for q in queries:
        assert q.primary_passage_id in corpus_ids

    assert compute_mrr_at_k(["a", "b", "c"], "b", k=10) == pytest.approx(0.5)
    assert compute_precision_at_1(["b", "a"], ("b",)) == pytest.approx(1.0)
    assert compute_precision_at_1(["a", "b"], ("b",)) == pytest.approx(0.0)
    assert compute_pool_recall(["x", "b", "y"], ("b", "z")) == pytest.approx(0.5)

    scorer = _JointTokenInteractionScorer()
    settings = CrossEncoderSettings(
        candidate_pools=(10, 50, 100),
        eval_top_k=10,
        warmup_queries=0,
        latency_repeats=1,
    )
    report = run_benchmark(
        settings=settings,
        bi_encoder=_CoarseBagEncoder(),
        cross_encoder=scorer,
        passages=corpus,
        queries=queries,
    )

    assert len(report.results) == 4
    bi_row, top10_row, top50_row, top100_row = report.results
    assert bi_row.candidate_pool_size == 0
    assert (top10_row.candidate_pool_size, top50_row.candidate_pool_size, top100_row.candidate_pool_size) == (
        10,
        50,
        100,
    )

    expected_counts = ([10] * 12) + ([50] * 12) + ([100] * 12)
    assert scorer.call_pair_counts == expected_counts

    assert top50_row.mrr_at_10 > bi_row.mrr_at_10
    assert top50_row.precision_at_1 > bi_row.precision_at_1
    assert top100_row.recall_at_pool >= top50_row.recall_at_pool >= top10_row.recall_at_pool
