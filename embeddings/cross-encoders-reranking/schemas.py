"""Frozen Pydantic v2 schemas for two-stage candidate retrieval and cross-encoder reranking."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SAFE_ID_PATTERN = r"^[a-z0-9][a-z0-9_-]{2,63}$"
_SAFE_ID_RE = re.compile(SAFE_ID_PATTERN)

QueryPhenomenon = Literal["polysemy", "negation", "constraint", "technical_specificity"]


class PassageRecord(BaseModel):
    """A single corpus passage indexed for stage-1 retrieval and stage-2 reranking."""

    model_config = ConfigDict(frozen=True)

    passage_id: str = Field(pattern=SAFE_ID_PATTERN)
    topic: str = Field(pattern=SAFE_ID_PATTERN)
    title: str = Field(min_length=5, max_length=160)
    text: str = Field(min_length=30, max_length=2000)


class RerankEvalQuery(BaseModel):
    """Ground-truth query testing fine-grained token interaction during reranking."""

    model_config = ConfigDict(frozen=True)

    query_id: str = Field(pattern=SAFE_ID_PATTERN)
    phenomenon: QueryPhenomenon
    query_text: str = Field(min_length=4, max_length=400)
    primary_passage_id: str = Field(pattern=SAFE_ID_PATTERN)
    relevant_passage_ids: tuple[str, ...] = Field(min_length=1)

    @field_validator("relevant_passage_ids")
    @classmethod
    def validate_relevant_ids(cls, ids: tuple[str, ...]) -> tuple[str, ...]:
        if len(ids) != len(set(ids)):
            raise ValueError("relevant_passage_ids must not contain duplicates")
        for pid in ids:
            if not _SAFE_ID_RE.match(pid):
                raise ValueError(f"Invalid passage_id {pid!r} in relevant_passage_ids")
        return ids

    @model_validator(mode="after")
    def validate_primary_in_relevant(self) -> RerankEvalQuery:
        if self.primary_passage_id not in self.relevant_passage_ids:
            raise ValueError(
                f"primary_passage_id {self.primary_passage_id!r} must belong to relevant_passage_ids"
            )
        return self


class RankedPassage(BaseModel):
    """A ranked passage carrying both stage-1 bi-encoder and stage-2 cross-encoder scores."""

    model_config = ConfigDict(frozen=True)

    passage_id: str = Field(pattern=SAFE_ID_PATTERN)
    rank: int = Field(ge=1)
    title: str = Field(min_length=5, max_length=160)
    bi_encoder_rank: int = Field(ge=1)
    bi_encoder_score: float
    cross_encoder_logit: float | None = None
    final_score: float


class SearchDiagnostics(BaseModel):
    """Per-query search response bundling ranked results with stage latency breakdown."""

    model_config = ConfigDict(frozen=True)

    query: str
    candidates_reranked: int = Field(ge=0)
    retrieval_ms: float = Field(ge=0.0)
    rerank_ms: float = Field(ge=0.0)
    total_ms: float = Field(ge=0.0)
    results: tuple[RankedPassage, ...]


class PoolScaleResult(BaseModel):
    """Quality and latency metrics for one candidate-pool depth configuration."""

    model_config = ConfigDict(frozen=True)

    pipeline_label: str = Field(min_length=3, max_length=80)
    candidate_pool_size: int = Field(ge=0)
    mrr_at_10: float = Field(ge=0.0, le=1.0)
    precision_at_1: float = Field(ge=0.0, le=1.0)
    recall_at_pool: float = Field(ge=0.0, le=1.0)
    retrieval_p50_ms: float = Field(ge=0.0)
    rerank_p50_ms: float = Field(ge=0.0)
    total_p50_ms: float = Field(ge=0.0)
    pairs_per_second: float = Field(ge=0.0)


class CrossEncoderBenchmarkReport(BaseModel):
    """Complete benchmark report across stage-1 baseline and candidate-pool depths."""

    model_config = ConfigDict(frozen=True)

    bi_encoder_model: str
    cross_encoder_model: str
    device: str
    corpus_size: int = Field(ge=1)
    query_count: int = Field(ge=1)
    rerank_batch_size: int = Field(ge=1)
    results: tuple[PoolScaleResult, ...]
