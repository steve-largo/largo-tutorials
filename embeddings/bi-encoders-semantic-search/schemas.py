"""Frozen Pydantic v2 schemas for bi-encoder documents, queries, hits, and benchmark metrics."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SAFE_ID_PATTERN = r"^[a-z0-9][a-z0-9_-]{2,63}$"
_SAFE_ID_RE = re.compile(SAFE_ID_PATTERN)

QueryTask = Literal["symmetric", "asymmetric"]
SearchModeName = Literal["single_query_loop", "batched_matrix"]


class DocumentRecord(BaseModel):
    """An indexed document with domain metadata for semantic vector retrieval."""

    model_config = ConfigDict(frozen=True)

    doc_id: str = Field(pattern=SAFE_ID_PATTERN)
    domain: str = Field(pattern=SAFE_ID_PATTERN)
    title: str = Field(min_length=5, max_length=160)
    body: str = Field(min_length=30, max_length=2000)


class BenchmarkQuery(BaseModel):
    """An evaluation query labeled as symmetric paraphrase or asymmetric short lookup."""

    model_config = ConfigDict(frozen=True)

    query_id: str = Field(pattern=SAFE_ID_PATTERN)
    task: QueryTask
    query_text: str = Field(min_length=3, max_length=400)
    relevant_doc_ids: tuple[str, ...] = Field(min_length=1)

    @field_validator("relevant_doc_ids")
    @classmethod
    def validate_relevant_ids(cls, doc_ids: tuple[str, ...]) -> tuple[str, ...]:
        if len(doc_ids) != len(set(doc_ids)):
            raise ValueError("relevant_doc_ids must not contain duplicates")
        for doc_id in doc_ids:
            if not _SAFE_ID_RE.match(doc_id):
                raise ValueError(f"Invalid doc_id {doc_id!r} in relevant_doc_ids")
        return doc_ids


class SearchHit(BaseModel):
    """A single nearest-neighbor result scored by cosine similarity."""

    model_config = ConfigDict(frozen=True)

    doc_id: str = Field(pattern=SAFE_ID_PATTERN)
    rank: int = Field(ge=1)
    cosine_score: float = Field(ge=-1.0001, le=1.0001)
    title: str = Field(min_length=5, max_length=160)


class RetrievalMetrics(BaseModel):
    """Aggregated Recall@k and MRR@k over a query collection."""

    model_config = ConfigDict(frozen=True)

    query_count: int = Field(ge=1)
    recall_at_k: float = Field(ge=0.0, le=1.0)
    mrr_at_k: float = Field(ge=0.0, le=1.0)


class SearchModeTiming(BaseModel):
    """Latency and throughput measurements for single-query vs batched search."""

    model_config = ConfigDict(frozen=True)

    mode: SearchModeName
    query_count: int = Field(ge=1)
    total_ms: float = Field(ge=0.0)
    per_query_p50_ms: float = Field(ge=0.0)
    queries_per_second: float = Field(ge=0.0)


class BiEncoderBenchmarkReport(BaseModel):
    """Reproducible evaluation scorecard for a bi-encoder vector index."""

    model_config = ConfigDict(frozen=True)

    model_name: str
    device: str
    embedding_dim: int = Field(ge=1)
    corpus_size: int = Field(ge=1)
    eval_k: int = Field(ge=1)
    overall: RetrievalMetrics
    by_task: dict[QueryTask, RetrievalMetrics]
    timings: tuple[SearchModeTiming, ...]
