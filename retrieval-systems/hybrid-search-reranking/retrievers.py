"""Lexical BM25, dense bi-encoder, Reciprocal Rank Fusion, and cross-encoder reranking."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Protocol

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from schemas import CorpusChunk, ScoredCandidate

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[._=-][a-z0-9]+)*")

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "into",
        "is",
        "it",
        "of",
        "on",
        "or",
        "our",
        "that",
        "the",
        "to",
        "via",
        "when",
        "with",
    }
)


class DenseEncoder(Protocol):
    """Protocol for unit-L2-normalized text embedding backends."""

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Encode texts into an (N, D) float32 L2-normalized matrix."""


class PairReranker(Protocol):
    """Protocol for joint query-passage relevance scoring backends."""

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        """Return raw relevance logits for (query, passage) pairs."""


def format_chunk_passage(chunk: CorpusChunk) -> str:
    """Combine chunk title and body into the canonical passage string."""
    return f"{chunk.title}. {chunk.text}"


def tokenize_lexical(text: str) -> list[str]:
    """Tokenize prose and technical identifiers (flags, SKUs, sysctls, error codes)."""
    tokens = _TOKEN_RE.findall(text.lower())
    filtered = [tok for tok in tokens if tok not in _STOPWORDS]
    return filtered if filtered else tokens


class SentenceTransformerBiEncoder:
    """Production SentenceTransformer adapter producing L2-normalized float32 vectors."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.model_name = model_name
        self._model = SentenceTransformer(model_name, device=device)

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        embeddings = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)


class SentenceTransformerCrossEncoder:
    """Production CrossEncoder adapter scoring concatenated query-passage pairs."""

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = CrossEncoder(model_name, device=device)

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        if not pairs:
            return np.empty(0, dtype=np.float32)
        scores = self._model.predict(
            [[query, passage] for query, passage in pairs],
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        return np.asarray(scores, dtype=np.float32)


class BM25LexicalRetriever:
    """Exact lexical retriever backed by Okapi BM25 over identifier-aware tokens."""

    def __init__(self, chunks: Sequence[CorpusChunk]) -> None:
        if not chunks:
            raise ValueError("BM25LexicalRetriever requires a non-empty corpus")
        self._chunks = tuple(chunks)
        tokenized_corpus = [
            tokenize_lexical(format_chunk_passage(chunk)) for chunk in self._chunks
        ]
        self._bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = 25) -> list[ScoredCandidate]:
        query_tokens = tokenize_lexical(query)
        if not query_tokens:
            return []
        raw_scores = np.asarray(self._bm25.get_scores(query_tokens), dtype=np.float64)
        matched_indices = [
            idx for idx in range(len(self._chunks)) if float(raw_scores[idx]) > 0.0
        ]
        ordered_indices = sorted(
            matched_indices,
            key=lambda idx: (-float(raw_scores[idx]), self._chunks[idx].chunk_id),
        )[: min(top_k, len(matched_indices))]
        return [
            ScoredCandidate(
                chunk_id=self._chunks[idx].chunk_id,
                rank=rank,
                score=float(raw_scores[idx]),
                bm25_rank=rank,
            )
            for rank, idx in enumerate(ordered_indices, start=1)
        ]


class DenseBiEncoderRetriever:
    """Cosine-similarity vector retriever over pre-encoded L2-normalized embeddings."""

    def __init__(self, chunks: Sequence[CorpusChunk], encoder: DenseEncoder) -> None:
        if not chunks:
            raise ValueError("DenseBiEncoderRetriever requires a non-empty corpus")
        self._chunks = tuple(chunks)
        self._encoder = encoder
        passages = [format_chunk_passage(chunk) for chunk in self._chunks]
        matrix = self._encoder.encode_texts(passages)
        if matrix.ndim != 2 or matrix.shape[0] != len(self._chunks):
            raise ValueError(
                f"Expected ({len(self._chunks)}, D) embedding matrix, got {matrix.shape}"
            )
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        self._embeddings = matrix / np.clip(norms, 1e-12, None)

    def search(self, query: str, top_k: int = 25) -> list[ScoredCandidate]:
        query_vec = self._encoder.encode_texts([query])[0]
        query_norm = float(np.linalg.norm(query_vec))
        if query_norm > 0.0:
            query_vec = query_vec / query_norm
        similarities = self._embeddings @ query_vec
        limit = min(top_k, len(self._chunks))
        ordered_indices = sorted(
            range(len(self._chunks)),
            key=lambda idx: (-float(similarities[idx]), self._chunks[idx].chunk_id),
        )[:limit]
        return [
            ScoredCandidate(
                chunk_id=self._chunks[idx].chunk_id,
                rank=rank,
                score=float(similarities[idx]),
                dense_rank=rank,
            )
            for rank, idx in enumerate(ordered_indices, start=1)
        ]


def reciprocal_rank_fusion(
    lexical_results: Sequence[ScoredCandidate],
    dense_results: Sequence[ScoredCandidate],
    rrf_k: int = 60,
    top_k: int = 25,
) -> list[ScoredCandidate]:
    """Fuse lexical and dense ranked lists via scale-free RRF: sum(1 / (k + rank))."""
    if rrf_k < 1:
        raise ValueError(f"rrf_k must be >= 1, got {rrf_k}")
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")

    rrf_scores: dict[str, float] = {}
    bm25_ranks: dict[str, int] = {}
    dense_ranks: dict[str, int] = {}

    for item in lexical_results:
        rrf_scores[item.chunk_id] = rrf_scores.get(item.chunk_id, 0.0) + (
            1.0 / (rrf_k + item.rank)
        )
        bm25_ranks[item.chunk_id] = item.rank

    for item in dense_results:
        rrf_scores[item.chunk_id] = rrf_scores.get(item.chunk_id, 0.0) + (
            1.0 / (rrf_k + item.rank)
        )
        dense_ranks[item.chunk_id] = item.rank

    def _sort_key(chunk_id: str) -> tuple[float, int, str]:
        best_rank = min(
            bm25_ranks.get(chunk_id, 10**9),
            dense_ranks.get(chunk_id, 10**9),
        )
        return (-rrf_scores[chunk_id], best_rank, chunk_id)

    ranked_ids = sorted(rrf_scores, key=_sort_key)[:top_k]
    return [
        ScoredCandidate(
            chunk_id=chunk_id,
            rank=rank,
            score=rrf_scores[chunk_id],
            bm25_rank=bm25_ranks.get(chunk_id),
            dense_rank=dense_ranks.get(chunk_id),
            rrf_score=rrf_scores[chunk_id],
        )
        for rank, chunk_id in enumerate(ranked_ids, start=1)
    ]


class HybridRRFRetriever:
    """Stage-1 hybrid retriever combining BM25Okapi and dense cosine search with RRF."""

    def __init__(
        self,
        lexical: BM25LexicalRetriever,
        dense: DenseBiEncoderRetriever,
        rrf_k: int = 60,
        stage1_top_k: int = 25,
    ) -> None:
        self._lexical = lexical
        self._dense = dense
        self.rrf_k = rrf_k
        self.stage1_top_k = stage1_top_k

    def search(self, query: str, top_k: int | None = None) -> list[ScoredCandidate]:
        target_k = top_k if top_k is not None else self.stage1_top_k
        fetch_k = max(self.stage1_top_k, target_k)
        lex_hits = self._lexical.search(query, top_k=fetch_k)
        dense_hits = self._dense.search(query, top_k=fetch_k)
        return reciprocal_rank_fusion(
            lexical_results=lex_hits,
            dense_results=dense_hits,
            rrf_k=self.rrf_k,
            top_k=target_k,
        )


class TwoStageHybridRetriever:
    """Two-stage hybrid pipeline: Stage-1 BM25+Dense RRF followed by Stage-2 Cross-Encoder."""

    def __init__(
        self,
        chunks: Sequence[CorpusChunk],
        stage1: HybridRRFRetriever,
        reranker: PairReranker,
        stage1_top_k: int = 25,
        rerank_top_k: int = 5,
    ) -> None:
        self._passage_by_id: Mapping[str, str] = {
            chunk.chunk_id: format_chunk_passage(chunk) for chunk in chunks
        }
        self._stage1 = stage1
        self._reranker = reranker
        self.stage1_top_k = stage1_top_k
        self.rerank_top_k = rerank_top_k

    def search(self, query: str, top_k: int | None = None) -> list[ScoredCandidate]:
        output_k = top_k if top_k is not None else self.rerank_top_k
        stage1_pool = self._stage1.search(
            query, top_k=max(self.stage1_top_k, output_k)
        )
        if not stage1_pool:
            return []

        pairs = [(query, self._passage_by_id[cand.chunk_id]) for cand in stage1_pool]
        cross_scores = self._reranker.score_pairs(pairs)

        rescored = sorted(
            zip(stage1_pool, cross_scores, strict=True),
            key=lambda item: (-float(item[1]), item[0].rank, item[0].chunk_id),
        )[:output_k]

        return [
            ScoredCandidate(
                chunk_id=cand.chunk_id,
                rank=new_rank,
                score=float(cross_logit),
                bm25_rank=cand.bm25_rank,
                dense_rank=cand.dense_rank,
                rrf_score=cand.rrf_score,
                rerank_score=float(cross_logit),
            )
            for new_rank, (cand, cross_logit) in enumerate(rescored, start=1)
        ]
