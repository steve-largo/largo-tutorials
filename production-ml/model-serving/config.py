"""Immutable runtime configuration for the FastAPI serving app and PyTorch benchmark."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from schemas import ExecutionRuntime

_ALLOWED_DEVICES = frozenset({"cpu", "cuda", "mps"})
CompileMode = Literal["default", "reduce-overhead", "max-autotune"]


class ServingSettings(BaseSettings):
    """Process-startup configuration loaded from MODEL_SERVING_* environment variables."""

    model_config = SettingsConfigDict(
        frozen=True,
        env_prefix="MODEL_SERVING_",
        extra="ignore",
    )

    service_name: str = Field(default="sentiment-inference-api", min_length=3, max_length=80)
    device: str = Field(default="cpu")
    runtime: ExecutionRuntime = Field(default="inference_mode_fp32")
    compile_mode: CompileMode = Field(default="reduce-overhead")
    vocab_size: int = Field(default=4096, ge=256, le=65536)
    embed_dim: int = Field(default=256, ge=64, le=1024)
    num_heads: int = Field(default=4, ge=1, le=16)
    num_layers: int = Field(default=4, ge=1, le=12)
    max_seq_length: int = Field(default=64, ge=16, le=512)
    max_batch_size: int = Field(default=32, ge=1, le=256)
    warmup_runs: int = Field(default=5, ge=0, le=50)
    benchmark_runs: int = Field(default=20, ge=1, le=500)
    benchmark_batch_sizes: tuple[int, ...] = Field(default=(1, 8, 32))
    include_torch_compile: bool = Field(default=True)
    seed: int = Field(default=42, ge=0)

    @field_validator("device")
    @classmethod
    def validate_device(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in _ALLOWED_DEVICES:
            raise ValueError(
                f"device must be one of {sorted(_ALLOWED_DEVICES)}, got {value!r}"
            )
        return normalized

    @field_validator("benchmark_batch_sizes")
    @classmethod
    def validate_batch_sizes(cls, sizes: tuple[int, ...]) -> tuple[int, ...]:
        if not sizes:
            raise ValueError("benchmark_batch_sizes must contain at least one positive integer")
        for size in sizes:
            if size < 1 or size > 512:
                raise ValueError(f"Batch size must be in 1..512, got {size}")
        if list(sizes) != sorted(set(sizes)):
            raise ValueError("benchmark_batch_sizes must be strictly increasing without duplicates")
        return sizes

    @model_validator(mode="after")
    def validate_transformer_geometry(self) -> ServingSettings:
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})"
            )
        return self


@lru_cache(maxsize=1)
def load_settings() -> ServingSettings:
    """Return cached immutable serving settings for the active process."""
    return ServingSettings()
