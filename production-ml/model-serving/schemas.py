"""Frozen Pydantic v2 request, response, and benchmark schemas for model serving."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SentimentLabel = Literal["negative", "positive"]
ExecutionRuntime = Literal[
    "eager_fp32",
    "inference_mode_fp32",
    "autocast_low_precision",
    "torch_compile",
]


class PredictRequest(BaseModel):
    """Inference payload accepting either a single `text` or a `texts` batch."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    texts: tuple[str, ...] = Field(min_length=1, max_length=64)

    @model_validator(mode="before")
    @classmethod
    def coerce_single_or_batch_payload(cls, raw: Any) -> Any:
        if not isinstance(raw, dict):
            return raw
        has_text = "text" in raw and raw["text"] is not None
        has_texts = "texts" in raw and raw["texts"] is not None
        if has_text and has_texts:
            raise ValueError("Provide either 'text' or 'texts', not both")
        if has_text:
            extra_keys = {k: v for k, v in raw.items() if k != "text"}
            return {"texts": (raw["text"],), **extra_keys}
        return raw

    @field_validator("texts")
    @classmethod
    def validate_each_text(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        cleaned: list[str] = []
        for item in values:
            stripped = item.strip()
            if not stripped:
                raise ValueError("Input text items must not be empty or whitespace")
            if len(stripped) > 512:
                raise ValueError(
                    f"Input text length ({len(stripped)}) exceeds 512-character boundary"
                )
            cleaned.append(stripped)
        return tuple(cleaned)


class SentimentPrediction(BaseModel):
    """Per-item classification prediction with calibrated softmax confidence."""

    model_config = ConfigDict(frozen=True)

    text_preview: str = Field(min_length=1, max_length=104)
    sentiment: SentimentLabel
    confidence: float = Field(ge=0.0, le=1.0)
    item_latency_ms: float = Field(ge=0.0)


class PredictResponse(BaseModel):
    """Typed inference response returned by POST /predict."""

    model_config = ConfigDict(frozen=True)

    runtime: ExecutionRuntime
    batch_size: int = Field(ge=1)
    total_latency_ms: float = Field(ge=0.0)
    predictions: tuple[SentimentPrediction, ...]


class HealthResponse(BaseModel):
    """Liveness and telemetry payload returned by GET /health."""

    model_config = ConfigDict(frozen=True)

    status: Literal["healthy"]
    service_name: str
    device: str
    runtime: ExecutionRuntime
    requests_processed: int = Field(ge=0)
    items_processed: int = Field(ge=0)
    gpu_memory_mb: float | None = None


class ReadyResponse(BaseModel):
    """Readiness probe payload returned by GET /ready."""

    model_config = ConfigDict(frozen=True)

    ready: bool
    model_loaded: bool
    runtime: ExecutionRuntime
    max_batch_size: int = Field(ge=1)


class RuntimeLatencyRow(BaseModel):
    """Latency, throughput, and numerical parity metrics for one runtime and batch size."""

    model_config = ConfigDict(frozen=True)

    runtime: ExecutionRuntime
    precision_dtype: str
    batch_size: int = Field(ge=1)
    mean_ms: float = Field(ge=0.0)
    p50_ms: float = Field(ge=0.0)
    p95_ms: float = Field(ge=0.0)
    requests_per_second: float = Field(ge=0.0)
    items_per_second: float = Field(ge=0.0)
    max_logit_abs_diff: float = Field(ge=0.0)


class ServingBenchmarkReport(BaseModel):
    """Complete PyTorch inference optimization benchmark report."""

    model_config = ConfigDict(frozen=True)

    device: str
    parameter_count: int = Field(ge=1)
    seq_length: int = Field(ge=8)
    warmup_runs: int = Field(ge=0)
    benchmark_runs: int = Field(ge=1)
    rows: tuple[RuntimeLatencyRow, ...]
