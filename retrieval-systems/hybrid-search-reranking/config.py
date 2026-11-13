"""Immutable runtime configuration for the two-stage hybrid retrieval benchmark."""

from __future__ import annotations

import re
from functools import lru_cache

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)?$")
_ALLOWED_DEVICES = frozenset({"cpu", "cuda", "mps"})


class RetrievalSettings(BaseSettings):
    """Process-startup configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        frozen=True,
        env_prefix="HYBRID_RETRIEVAL_",
        extra="ignore",
    )

    bi_encoder_model: str = Field(default="BAAI/bge-small-en-v1.5")
    cross_encoder_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    rrf_k: int = Field(default=60, ge=1, le=500)
    stage1_top_k: int = Field(default=25, ge=5, le=200)
    rerank_top_k: int = Field(default=5, ge=1, le=100)
    eval_top_k: int = Field(default=10, ge=5, le=100)
    rerank_batch_size: int = Field(default=32, ge=1, le=256)
    warmup_queries: int = Field(default=3, ge=0, le=20)
    latency_repeats: int = Field(default=3, ge=1, le=20)
    seed: int = Field(default=42, ge=0)
    device: str = Field(default="cpu")

    @field_validator("bi_encoder_model", "cross_encoder_model")
    @classmethod
    def validate_model_identifier(cls, value: str) -> str:
        if ".." in value or not _MODEL_ID_PATTERN.match(value):
            raise ValueError(
                f"Model identifier must be a clean Hugging Face repo ID without '..', got {value!r}"
            )
        return value

    @field_validator("device")
    @classmethod
    def validate_device(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in _ALLOWED_DEVICES:
            raise ValueError(
                f"device must be one of {sorted(_ALLOWED_DEVICES)}, got {value!r}"
            )
        return normalized

    @model_validator(mode="after")
    def validate_cutoff_ordering(self) -> RetrievalSettings:
        if self.rerank_top_k > self.stage1_top_k:
            raise ValueError(
                f"rerank_top_k ({self.rerank_top_k}) cannot exceed stage1_top_k ({self.stage1_top_k})"
            )
        if self.eval_top_k > self.stage1_top_k:
            raise ValueError(
                f"eval_top_k ({self.eval_top_k}) cannot exceed stage1_top_k ({self.stage1_top_k})"
            )
        return self


@lru_cache(maxsize=1)
def load_settings() -> RetrievalSettings:
    """Return cached immutable settings for the current process."""
    return RetrievalSettings()
