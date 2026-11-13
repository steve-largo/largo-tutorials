"""Frozen Pydantic v2 schemas for corpus chunks, diagnostic queries, and retrieval metrics."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SAFE_ID_PATTERN = r"^[a-z0-9][a-z0-9_-]{2,63}$"

QuerySlice = Literal["exact_id", "semantic_paraphrase"]
PipelineName = Literal["bm25_only", "dense_only", "hybrid_rrf", "hybrid_rrf_rerank"]


class CorpusChunk(BaseModel):
    """A single indexed engineering runbook or architecture reference chunk."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(pattern=SAFE_ID_PATTERN)
    domain: str = Field(pattern=SAFE_ID_PATTERN)
    title: str = Field(min_length=5, max_length=160)
    text: str = Field(min_length=40, max_length=2000)


class EvalQuery(BaseModel):
    """A ground-truth benchmark query tagged with its diagnostic slice and graded relevance."""

    model_config = ConfigDict(frozen=True)

    query_id: str = Field(pattern=SAFE_ID_PATTERN)
    query_slice: QuerySlice
    query_text: str = Field(min_length=4, max_length=400)
    relevant_grades: dict[str, int] = Field(min_length=1)

    @field_validator("relevant_grades")
    @classmethod
    def validate_grades(cls, grades: dict[str, int]) -> dict[str, int]:
        for chunk_id, grade in grades.items():
            if grade < 1 or grade > 3:
                raise ValueError(
                    f"Relevance grade for {chunk_id!r} must be in 1..3, got {grade}"
                )
        if not any(grade >= 2 for grade in grades.values()):
            raise ValueError("Each EvalQuery must designate at least one primary chunk (grade >= 2)")
        return grades


class ScoredCandidate(BaseModel):
    """A ranked candidate chunk emitted by a retrieval or reranking stage."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(pattern=SAFE_ID_PATTERN)
    rank: int = Field(ge=1)
    score: float
    bm25_rank: int | None = Field(default=None, ge=1)
    dense_rank: int | None = Field(default=None, ge=1)
    rrf_score: float | None = None
    rerank_score: float | None = None


class SliceMetrics(BaseModel):
    """Aggregated retrieval quality metrics across a query subset."""

    model_config = ConfigDict(frozen=True)

    query_count: int = Field(ge=1)
    recall_at_5: float = Field(ge=0.0, le=1.0)
    recall_at_25: float = Field(ge=0.0, le=1.0)
    mrr_at_10: float = Field(ge=0.0, le=1.0)
    ndcg_at_10: float = Field(ge=0.0, le=1.0)


class StageMetrics(BaseModel):
    """Quality and latency scorecard for one retrieval pipeline configuration."""

    model_config = ConfigDict(frozen=True)

    pipeline_id: PipelineName
    pipeline_label: str = Field(min_length=3, max_length=80)
    overall: SliceMetrics
    by_slice: dict[QuerySlice, SliceMetrics]
    latency_p50_ms: float = Field(ge=0.0)
    latency_p95_ms: float = Field(ge=0.0)
    latency_mean_ms: float = Field(ge=0.0)


class BenchmarkReport(BaseModel):
    """Complete reproducible benchmark report across all four retrieval stages."""

    model_config = ConfigDict(frozen=True)

    corpus_size: int = Field(ge=1)
    query_count: int = Field(ge=1)
    bi_encoder_model: str
    cross_encoder_model: str
    device: str
    rrf_k: int = Field(ge=1)
    stage1_top_k: int = Field(ge=1)
    rerank_top_k: int = Field(ge=1)
    eval_top_k: int = Field(ge=1)
    stages: tuple[StageMetrics, ...]
