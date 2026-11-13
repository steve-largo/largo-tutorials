"""Immutable runtime configuration for two-stage cross-encoder reranking."""

from __future__ import annotations

import re
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)?$")
_ALLOWED_DEVICES = frozenset({"cpu", "cuda", "mps"})


class CrossEncoderSettings(BaseSettings):
    """Process-startup configuration loaded from CROSS_ENCODER_* environment variables."""

    model_config = SettingsConfigDict(
        frozen=True,
        env_prefix="CROSS_ENCODER_",
        extra="ignore",
    )

    bi_encoder_model: str = Field(default="BAAI/bge-small-en-v1.5")
    cross_encoder_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    candidate_pools: tuple[int, ...] = Field(default=(10, 50, 100))
    eval_top_k: int = Field(default=10, ge=1, le=100)
    rerank_batch_size: int = Field(default=32, ge=1, le=256)
    fusion_alpha: float = Field(default=0.0, ge=0.0, le=1.0)
    warmup_queries: int = Field(default=1, ge=0, le=20)
    latency_repeats: int = Field(default=1, ge=1, le=20)
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

    @field_validator("candidate_pools")
    @classmethod
    def validate_candidate_pools(cls, pools: tuple[int, ...]) -> tuple[int, ...]:
        if not pools:
            raise ValueError("candidate_pools must specify at least one positive pool size")
        for size in pools:
            if size < 1 or size > 1000:
                raise ValueError(f"Each candidate pool size must be in 1..1000, got {size}")
        if list(pools) != sorted(set(pools)):
            raise ValueError("candidate_pools must be strictly increasing without duplicates")
        return pools

    @field_validator("device")
    @classmethod
    def validate_device(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in _ALLOWED_DEVICES:
            raise ValueError(
                f"device must be one of {sorted(_ALLOWED_DEVICES)}, got {value!r}"
            )
        return normalized


@lru_cache(maxsize=1)
def load_settings() -> CrossEncoderSettings:
    """Return cached immutable settings for the current process."""
    return CrossEncoderSettings()
