"""Normalized dense vector index supporting exact single-query and batched cosine search."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from schemas import DocumentRecord, SearchHit


class TextEncoder(Protocol):
    """Protocol for dense text embedding backends."""

    def encode_texts(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts into an (N, D) float32 embedding matrix."""


def format_document_passage(doc: DocumentRecord) -> str:
    """Combine document title and body into the indexed passage representation."""
    return f"{doc.title}. {doc.body}"


def l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Normalize each row vector in a 2D matrix to unit L2 length."""
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"Expected non-empty 2D embedding matrix, got shape {arr.shape}")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("Cannot L2-normalize a zero-norm embedding vector")
    return (arr / norms).astype(np.float32)


def cosine_similarity_matrix(
    query_embeddings: np.ndarray,
    corpus_embeddings: np.ndarray,
) -> np.ndarray:
    """Compute exact cosine similarity via inner product over unit-normalized matrices."""
    q_norm = l2_normalize_rows(query_embeddings)
    c_norm = l2_normalize_rows(corpus_embeddings)
    if q_norm.shape[1] != c_norm.shape[1]:
        raise ValueError(
            f"Embedding dimension mismatch: queries={q_norm.shape[1]}, corpus={c_norm.shape[1]}"
        )
    return (q_norm @ c_norm.T).astype(np.float32)


class SentenceTransformerEncoder:
    """SentenceTransformer adapter producing unit-normalized float32 embeddings."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name, device=device)

    def encode_texts(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            raise ValueError("encode_texts requires at least one input string")
        vectors = self._model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return l2_normalize_rows(np.asarray(vectors, dtype=np.float32))


class DenseVectorIndex:
    """In-memory L2-normalized dense index with single-query and batched BLAS search."""

    def __init__(self, encoder: TextEncoder) -> None:
        self._encoder = encoder
        self._documents: tuple[DocumentRecord, ...] = ()
        self._embeddings: np.ndarray | None = None

    @property
    def size(self) -> int:
        return len(self._documents)

    @property
    def embedding_dim(self) -> int:
        if self._embeddings is None:
            return 0
        return int(self._embeddings.shape[1])

    @property
    def embeddings(self) -> np.ndarray:
        if self._embeddings is None:
            raise RuntimeError("Index contains no embeddings; call index_documents first")
        return self._embeddings

    def index_documents(
        self,
        documents: Sequence[DocumentRecord],
        batch_size: int = 32,
    ) -> int:
        """Encode and L2-normalize all corpus passages once at index build time."""
        if not documents:
            raise ValueError("index_documents requires a non-empty document sequence")
        doc_tuple = tuple(documents)
        seen_ids: set[str] = set()
        for doc in doc_tuple:
            if doc.doc_id in seen_ids:
                raise ValueError(f"Duplicate doc_id in corpus: {doc.doc_id!r}")
            seen_ids.add(doc.doc_id)

        passages = [format_document_passage(doc) for doc in doc_tuple]
        raw_matrix = self._encoder.encode_texts(passages, batch_size=batch_size)
        if raw_matrix.shape[0] != len(doc_tuple):
            raise ValueError(
                f"Encoder returned {raw_matrix.shape[0]} rows for {len(doc_tuple)} documents"
            )
        self._embeddings = l2_normalize_rows(raw_matrix)
        self._documents = doc_tuple
        return len(self._documents)

    def search_exact(self, query: str, top_k: int = 5) -> list[SearchHit]:
        """Encode one query and rank corpus vectors via exact dot-product cosine similarity."""
        if self._embeddings is None:
            raise RuntimeError("Index is empty; call index_documents before searching")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if not query.strip():
            raise ValueError("query must not be empty or whitespace")

        query_vec = l2_normalize_rows(self._encoder.encode_texts([query], batch_size=1))[0]
        scores = self._embeddings @ query_vec
        return self._rank_scores(scores, top_k=top_k)

    def search_batched(
        self,
        queries: Sequence[str],
        top_k: int = 5,
        batch_size: int = 32,
    ) -> list[list[SearchHit]]:
        """Encode a batch of queries and rank all queries in one matrix multiplication."""
        if self._embeddings is None:
            raise RuntimeError("Index is empty; call index_documents before searching")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if not queries:
            return []
        for q in queries:
            if not q.strip():
                raise ValueError("queries must not contain empty or whitespace strings")

        query_matrix = l2_normalize_rows(
            self._encoder.encode_texts(list(queries), batch_size=batch_size)
        )
        score_matrix = query_matrix @ self._embeddings.T
        return [
            self._rank_scores(score_matrix[row_idx], top_k=top_k)
            for row_idx in range(score_matrix.shape[0])
        ]

    def _rank_scores(self, scores: np.ndarray, top_k: int) -> list[SearchHit]:
        limit = min(top_k, len(self._documents))
        ordered_indices = sorted(
            range(len(self._documents)),
            key=lambda idx: (-float(scores[idx]), self._documents[idx].doc_id),
        )[:limit]
        return [
            SearchHit(
                doc_id=self._documents[idx].doc_id,
                rank=rank,
                cosine_score=float(scores[idx]),
                title=self._documents[idx].title,
            )
            for rank, idx in enumerate(ordered_indices, start=1)
        ]

    def save(self, directory: Path) -> None:
        """Persist embeddings (.npz) and validated document records (.json) without pickle."""
        if self._embeddings is None:
            raise RuntimeError("Cannot save an uninitialized DenseVectorIndex")
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / "embeddings.npz", embeddings=self._embeddings)
        payload = [doc.model_dump() for doc in self._documents]
        (directory / "documents.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path, encoder: TextEncoder) -> DenseVectorIndex:
        """Restore a persisted DenseVectorIndex from embeddings.npz and documents.json."""
        emb_path = directory / "embeddings.npz"
        doc_path = directory / "documents.json"
        if not emb_path.exists() or not doc_path.exists():
            raise FileNotFoundError(f"Missing index artifacts in {directory}")

        with np.load(emb_path, allow_pickle=False) as archive:
            raw_matrix = np.asarray(archive["embeddings"], dtype=np.float32)
        if raw_matrix.ndim != 2 or raw_matrix.shape[0] == 0 or raw_matrix.shape[1] == 0:
            raise ValueError(f"Invalid persisted embedding matrix shape: {raw_matrix.shape}")
        norms = np.linalg.norm(raw_matrix, axis=1)
        matrix = raw_matrix if np.allclose(norms, 1.0, atol=1e-4) else l2_normalize_rows(raw_matrix)
        raw_docs = json.loads(doc_path.read_text(encoding="utf-8"))
        docs = tuple(DocumentRecord.model_validate(item) for item in raw_docs)
        if matrix.shape[0] != len(docs):
            raise ValueError(
                f"Corrupt index: {matrix.shape[0]} vectors vs {len(docs)} document records"
            )

        instance = cls(encoder=encoder)
        instance._documents = docs
        instance._embeddings = matrix
        return instance
