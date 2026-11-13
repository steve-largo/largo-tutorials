"""Immutable runtime configuration for the bi-encoder semantic search package."""

from __future__ import annotations

import re
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)?$")
_ALLOWED_DEVICES = frozenset({"cpu", "cuda", "mps"})


class BiEncoderSettings(BaseSettings):
    """Process-startup configuration loaded from BI_ENCODER_* environment variables."""

    model_config = SettingsConfigDict(
        frozen=True,
        env_prefix="BI_ENCODER_",
        extra="ignore",
    )

    model_name: str = Field(default="BAAI/bge-small-en-v1.5")
    encode_batch_size: int = Field(default=32, ge=1, le=512)
    eval_top_k: int = Field(default=5, ge=1, le=50)
    warmup_runs: int = Field(default=2, ge=0, le=20)
    timing_repeats: int = Field(default=5, ge=1, le=50)
    seed: int = Field(default=42, ge=0)
    device: str = Field(default="cpu")

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, value: str) -> str:
        if ".." in value or not _MODEL_ID_PATTERN.match(value):
            raise ValueError(
                f"model_name must be a clean Hugging Face repo ID without '..', got {value!r}"
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


@lru_cache(maxsize=1)
def load_settings() -> BiEncoderSettings:
    """Return cached immutable settings for the active process."""
    return BiEncoderSettings()
