"""Deterministic unit and invariant tests for bi-encoder semantic search."""

from __future__ import annotations

import zlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from benchmark import (
    compute_mrr_at_k,
    compute_recall_at_k,
    load_corpus,
    load_eval_queries,
    run_benchmark,
)
from config import BiEncoderSettings
from index import DenseVectorIndex, cosine_similarity_matrix, l2_normalize_rows
from schemas import BenchmarkQuery, DocumentRecord


class _ConceptSubspaceEncoder:
    """Deterministic offline test encoder projecting text onto 16 concept dimensions."""

    _CONCEPTS: tuple[tuple[str, ...], ...] = (
        ("tire", "lug", "highway", "wheel", "spare", "jacking"),
        ("bike", "bicycle", "tube", "bead", "punctured", "pinch"),
        ("review", "pull requests", "checklist", "except", "boundary"),
        ("tracemalloc", "heap", "leak", "snapshots", "memory"),
        ("backpropagation", "autodiff", "gradients", "chain rule", "vanishing"),
        ("adamw", "weight decay", "warmup", "cosine"),
        ("negative", "triplet", "infonce", "contrastive"),
        ("canary", "shadow", "mirror", "rollout"),
        ("hnsw", "nearest neighbor", "proximity", "vector search"),
        ("padding", "token length", "sort", "throughput", "batches"),
        ("partial", "b-tree", "pending", "unarchived"),
        ("serializable", "write skew", "transactions"),
        ("sourdough", "levain", "dutch oven", "artisan", "bread"),
        ("keepalive", "firewall", "idle", "nat", "tcp"),
        ("coredns", "ndots", "suffix", "kubernetes", "dns"),
        ("coolant", "thermostat", "bleed", "cookies", "pgbouncer", "wal"),
    )

    def __init__(self, scale: float = 1.0) -> None:
        self._scale = scale

    def encode_texts(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            raise ValueError("texts must not be empty")
        assert batch_size >= 1
        rows: list[np.ndarray] = []
        for text in texts:
            lowered = text.lower()
            fingerprint = zlib.crc32(lowered.encode("utf-8"))
            vec = np.array(
                [
                    sum(1.5 for kw in axis if kw in lowered)
                    + 1e-3 * (1.0 + (((fingerprint >> (dim_idx % 16)) & 0xF) / 16.0))
                    for dim_idx, axis in enumerate(self._CONCEPTS)
                ],
                dtype=np.float32,
            )
            rows.append(vec * self._scale)
        return np.vstack(rows)


def test_schema_and_settings_validation() -> None:
    with pytest.raises(ValidationError):
        BiEncoderSettings(model_name="../unsafe/path")

    with pytest.raises(ValidationError):
        BiEncoderSettings(device="quantum-core")

    with pytest.raises(ValidationError):
        BenchmarkQuery(
            query_id="duplicate-targets",
            task="symmetric",
            query_text="duplicate doc ids should fail validation",
            relevant_doc_ids=("auto-flat-tire-highway", "auto-flat-tire-highway"),
        )

    cfg = BiEncoderSettings(eval_top_k=5)
    with pytest.raises(ValidationError):
        cfg.eval_top_k = 10  # type: ignore[misc]


def test_l2_normalization_scale_invariance_and_zero_vector_guard() -> None:
    raw = np.array([[3.0, 4.0, 0.0], [0.0, 5.0, 12.0]], dtype=np.float32)
    normed = l2_normalize_rows(raw)
    norms = np.linalg.norm(normed, axis=1)
    assert norms == pytest.approx([1.0, 1.0], abs=1e-6)

    sim_unscaled = cosine_similarity_matrix(raw[:1], raw)
    sim_scaled = cosine_similarity_matrix(raw[:1] * 250.0, raw * 0.01)
    assert sim_unscaled == pytest.approx(sim_scaled, abs=1e-6)

    with pytest.raises(ValueError, match="zero-norm"):
        l2_normalize_rows(np.zeros((1, 4), dtype=np.float32))


def test_exact_vs_batched_search_equivalence() -> None:
    corpus = load_corpus()
    queries = load_eval_queries()

    idx_unit = DenseVectorIndex(encoder=_ConceptSubspaceEncoder(scale=1.0))
    idx_scaled = DenseVectorIndex(encoder=_ConceptSubspaceEncoder(scale=42.0))
    assert idx_unit.index_documents(corpus, batch_size=8) == 24
    assert idx_scaled.index_documents(corpus, batch_size=8) == 24

    batched_hits = idx_unit.search_batched(
        [q.query_text for q in queries],
        top_k=5,
        batch_size=4,
    )
    assert len(batched_hits) == len(queries)

    for query, batch_row in zip(queries, batched_hits, strict=True):
        single_row = idx_unit.search_exact(query.query_text, top_k=5)
        scaled_row = idx_scaled.search_exact(query.query_text, top_k=5)

        assert [h.doc_id for h in single_row] == [h.doc_id for h in batch_row]
        assert [h.doc_id for h in single_row] == [h.doc_id for h in scaled_row]
        assert [h.cosine_score for h in single_row] == pytest.approx(
            [h.cosine_score for h in batch_row], abs=1e-5
        )


def test_npz_persistence_roundtrip_without_pickle(tmp_path: Path) -> None:
    mini_docs = load_corpus()[:6]
    encoder = _ConceptSubspaceEncoder()
    original = DenseVectorIndex(encoder=encoder)
    original.index_documents(mini_docs)
    original.save(tmp_path / "saved_index")

    restored = DenseVectorIndex.load(tmp_path / "saved_index", encoder=encoder)
    assert restored.size == 6
    assert restored.embedding_dim == 16

    hits_before = original.search_exact("fixing a punctured wheel", top_k=3)
    hits_after = restored.search_exact("fixing a punctured wheel", top_k=3)
    assert hits_before == hits_after


def test_recall_and_mrr_metric_math() -> None:
    ranked = ["doc-x", "doc-b", "doc-a", "doc-y"]
    relevant = ("doc-a", "doc-b")

    assert compute_recall_at_k(ranked, relevant, k=1) == pytest.approx(0.0)
    assert compute_recall_at_k(ranked, relevant, k=2) == pytest.approx(0.5)
    assert compute_recall_at_k(ranked, relevant, k=3) == pytest.approx(1.0)
    assert compute_mrr_at_k(ranked, relevant, k=4) == pytest.approx(0.5)
    assert compute_mrr_at_k(["doc-x", "doc-y"], relevant, k=2) == pytest.approx(0.0)


def test_corpus_integrity_and_symmetric_asymmetric_benchmark() -> None:
    corpus = load_corpus()
    queries = load_eval_queries()

    doc_ids = {d.doc_id for d in corpus}
    assert len(doc_ids) == 24

    sym_queries = [q for q in queries if q.task == "symmetric"]
    asym_queries = [q for q in queries if q.task == "asymmetric"]
    assert len(sym_queries) == 8
    assert len(asym_queries) == 8

    for q in queries:
        for target_id in q.relevant_doc_ids:
            assert target_id in doc_ids

    with pytest.raises(ValueError, match="Duplicate doc_id"):
        DenseVectorIndex(encoder=_ConceptSubspaceEncoder()).index_documents(
            (DocumentRecord.model_validate(corpus[0].model_dump()), corpus[0])
        )

    settings = BiEncoderSettings(eval_top_k=5, warmup_runs=1, timing_repeats=2)
    report = run_benchmark(
        settings=settings,
        encoder=_ConceptSubspaceEncoder(),
        documents=corpus,
        queries=queries,
    )
    assert report.overall.recall_at_k >= 0.9
    assert report.by_task["symmetric"].mrr_at_k >= 0.9
    assert report.by_task["asymmetric"].recall_at_k >= 0.9
    assert len(report.timings) == 2
