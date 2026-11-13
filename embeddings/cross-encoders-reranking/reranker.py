"""Two-stage dense retrieval and cross-encoder reranking with injectable model backends."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from schemas import PassageRecord, RankedPassage, SearchDiagnostics


class BiEncoderBackend(Protocol):
    """Protocol for stage-1 text embedding encoders."""

    def encode_texts(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts into an (N, D) float32 embedding matrix."""


class CrossEncoderScorer(Protocol):
    """Protocol for stage-2 joint query-passage relevance scorers."""

    def score_pairs(
        self,
        pairs: Sequence[tuple[str, str]],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Return raw relevance logits for concatenated (query, passage) pairs."""


def format_passage_text(record: PassageRecord) -> str:
    """Format passage title and body into a canonical scoring string."""
    return f"{record.title}. {record.text}"


def normalize_embeddings_l2(matrix: np.ndarray) -> np.ndarray:
    """Normalize rows of a 2D embedding matrix to unit L2 length."""
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"Expected non-empty 2D embedding matrix, got {arr.shape}")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("Cannot L2-normalize a zero-norm embedding vector")
    return (arr / norms).astype(np.float32)


def sigmoid_probability(logit: float) -> float:
    """Numerically stable logistic sigmoid mapping raw cross-encoder logits onto (0, 1)."""
    if logit >= 0.0:
        z = math.exp(-logit)
        return 1.0 / (1.0 + z)
    z = math.exp(logit)
    return z / (1.0 + z)


def blend_relevance_scores(
    bi_cosine: float,
    cross_logit: float,
    alpha: float = 0.0,
) -> float:
    """Rank purely by cross-encoder logit when alpha=0, or blend cosine + sigmoid(logit)."""
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError(f"alpha must be in [0.0, 1.0], got {alpha}")
    if alpha == 0.0:
        return float(cross_logit)
    if alpha == 1.0:
        return float(bi_cosine)
    bi_unit = max(0.0, min(1.0, (bi_cosine + 1.0) * 0.5 if bi_cosine < 0.0 else bi_cosine))
    return float(alpha * bi_unit + (1.0 - alpha) * sigmoid_probability(cross_logit))


class SentenceTransformerBiEncoder:
    """SentenceTransformer stage-1 bi-encoder adapter."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name, device=device)

    def encode_texts(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            raise ValueError("encode_texts requires at least one string")
        vectors = self._model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return normalize_embeddings_l2(np.asarray(vectors, dtype=np.float32))


class MsMarcoCrossEncoder:
    """SentenceTransformers CrossEncoder adapter (e.g. cross-encoder/ms-marco-MiniLM-L-6-v2)."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self._model = CrossEncoder(model_name, device=device)

    def score_pairs(
        self,
        pairs: Sequence[tuple[str, str]],
        batch_size: int = 32,
    ) -> np.ndarray:
        if not pairs:
            return np.empty(0, dtype=np.float32)
        logits = self._model.predict(
            [[query, passage] for query, passage in pairs],
            batch_size=batch_size,
            show_progress_bar=False,
        )
        return np.asarray(logits, dtype=np.float32)


class TwoStageReranker:
    """Two-stage search engine pairing fast dense candidate recall with cross-encoder reranking."""

    def __init__(
        self,
        bi_encoder: BiEncoderBackend,
        cross_encoder: CrossEncoderScorer,
    ) -> None:
        self._bi_encoder = bi_encoder
        self._cross_encoder = cross_encoder
        self._passages: tuple[PassageRecord, ...] = ()
        self._formatted_texts: tuple[str, ...] = ()
        self._embeddings: np.ndarray | None = None

    @property
    def corpus_size(self) -> int:
        return len(self._passages)

    def index_passages(
        self,
        passages: Sequence[PassageRecord],
        batch_size: int = 32,
    ) -> int:
        """Pre-compute normalized stage-1 embeddings for all corpus passages."""
        if not passages:
            raise ValueError("index_passages requires a non-empty passage sequence")
        records = tuple(passages)
        seen_ids: set[str] = set()
        for rec in records:
            if rec.passage_id in seen_ids:
                raise ValueError(f"Duplicate passage_id in corpus: {rec.passage_id!r}")
            seen_ids.add(rec.passage_id)

        texts = tuple(format_passage_text(rec) for rec in records)
        matrix = self._bi_encoder.encode_texts(texts, batch_size=batch_size)
        if matrix.shape[0] != len(records):
            raise ValueError(
                f"Bi-encoder returned {matrix.shape[0]} vectors for {len(records)} passages"
            )
        self._embeddings = normalize_embeddings_l2(matrix)
        self._passages = records
        self._formatted_texts = texts
        return len(self._passages)

    def retrieve_candidates(
        self,
        query: str,
        pool_size: int = 100,
    ) -> tuple[list[tuple[int, float]], float]:
        """Execute stage-1 cosine search and return ((corpus_index, bi_score), retrieval_ms)."""
        if self._embeddings is None:
            raise RuntimeError("Index is empty; call index_passages before searching")
        if pool_size < 1:
            raise ValueError(f"pool_size must be >= 1, got {pool_size}")
        if not query.strip():
            raise ValueError("query must not be empty or whitespace")

        t0 = time.perf_counter()
        q_vec = normalize_embeddings_l2(self._bi_encoder.encode_texts([query], batch_size=1))[0]
        similarities = self._embeddings @ q_vec
        limit = min(pool_size, len(self._passages))
        ordered_indices = sorted(
            range(len(self._passages)),
            key=lambda idx: (-float(similarities[idx]), self._passages[idx].passage_id),
        )[:limit]
        retrieval_ms = (time.perf_counter() - t0) * 1000.0
        candidates = [(idx, float(similarities[idx])) for idx in ordered_indices]
        return candidates, retrieval_ms

    def search_bi_encoder_only(
        self,
        query: str,
        top_k: int = 10,
    ) -> SearchDiagnostics:
        """Run stage-1 dense retrieval without cross-encoder reranking."""
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        candidates, retrieval_ms = self.retrieve_candidates(query, pool_size=top_k)
        results = tuple(
            RankedPassage(
                passage_id=self._passages[idx].passage_id,
                rank=rank,
                title=self._passages[idx].title,
                bi_encoder_rank=rank,
                bi_encoder_score=bi_score,
                cross_encoder_logit=None,
                final_score=bi_score,
            )
            for rank, (idx, bi_score) in enumerate(candidates, start=1)
        )
        return SearchDiagnostics(
            query=query,
            candidates_reranked=0,
            retrieval_ms=round(retrieval_ms, 3),
            rerank_ms=0.0,
            total_ms=round(retrieval_ms, 3),
            results=results,
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        candidate_pool_size: int = 100,
        batch_size: int = 32,
        fusion_alpha: float = 0.0,
    ) -> SearchDiagnostics:
        """Retrieve candidate_pool_size passages via bi-encoder and rerank via cross-encoder."""
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if candidate_pool_size < 1:
            raise ValueError(f"candidate_pool_size must be >= 1, got {candidate_pool_size}")

        pool_k = max(candidate_pool_size, top_k)
        candidates, retrieval_ms = self.retrieve_candidates(query, pool_size=pool_k)

        t_rerank = time.perf_counter()
        pairs = [(query, self._formatted_texts[idx]) for idx, _ in candidates]
        cross_logits = self._cross_encoder.score_pairs(pairs, batch_size=batch_size)
        if len(cross_logits) != len(candidates):
            raise ValueError(
                f"Cross-encoder returned {len(cross_logits)} scores for {len(candidates)} pairs"
            )
        rerank_ms = (time.perf_counter() - t_rerank) * 1000.0

        enriched: list[tuple[int, int, float, float, float]] = []
        for bi_rank, ((corpus_idx, bi_score), raw_logit) in enumerate(
            zip(candidates, cross_logits, strict=True),
            start=1,
        ):
            logit_val = float(raw_logit)
            final_val = blend_relevance_scores(
                bi_cosine=bi_score,
                cross_logit=logit_val,
                alpha=fusion_alpha,
            )
            enriched.append((corpus_idx, bi_rank, bi_score, logit_val, final_val))

        enriched.sort(
            key=lambda item: (-item[4], item[1], self._passages[item[0]].passage_id)
        )

        ranked_results = tuple(
            RankedPassage(
                passage_id=self._passages[corpus_idx].passage_id,
                rank=new_rank,
                title=self._passages[corpus_idx].title,
                bi_encoder_rank=bi_rank,
                bi_encoder_score=bi_score,
                cross_encoder_logit=logit_val,
                final_score=final_val,
            )
            for new_rank, (corpus_idx, bi_rank, bi_score, logit_val, final_val) in enumerate(
                enriched[: min(top_k, len(enriched))],
                start=1,
            )
        )

        total_ms = retrieval_ms + rerank_ms
        return SearchDiagnostics(
            query=query,
            candidates_reranked=len(candidates),
            retrieval_ms=round(retrieval_ms, 3),
            rerank_ms=round(rerank_ms, 3),
            total_ms=round(total_ms, 3),
            results=ranked_results,
        )
