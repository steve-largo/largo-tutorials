"""Typed contract models for MoE VRAM sizing, SFT curation, LoRA auditing, and ADK serving."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class PrecisionFormat(StrEnum):
    """Weight and KV-cache numerical storage formats supported by vLLM."""

    BF16 = "bfloat16"
    FP8 = "fp8"
    AWQ_INT4 = "awq_int4"


class ModelArchSpec(BaseModel):
    """Published architectural parameters governing VRAM residency and KV-cache growth."""

    model_config = ConfigDict(frozen=True)

    model_id: str
    total_params_billions: float = Field(gt=0)
    active_params_billions: float = Field(gt=0)
    num_layers: int = Field(gt=0)
    sliding_window_layers: int = Field(ge=0)
    global_attention_layers: int = Field(ge=0)
    sliding_window_tokens: int = Field(gt=0)
    num_kv_heads: int = Field(gt=0)
    head_dim: int = Field(gt=0)
    max_context_tokens: int = Field(gt=0)


class GpuSkuSpec(BaseModel):
    """Public Vertex AI dedicated prediction machine SKU and hourly rate."""

    model_config = ConfigDict(frozen=True)

    machine_type: str
    accelerator_type: str
    accelerator_count: int = Field(ge=1)
    vram_per_gpu_gb: float = Field(gt=0)
    vertex_hourly_usd: float = Field(gt=0)

    @property
    def total_vram_gb(self) -> float:
        return self.accelerator_count * self.vram_per_gpu_gb


class VramBreakdown(BaseModel):
    """Analytical GPU memory budget for a specific model, precision, and SKU."""

    model_config = ConfigDict(frozen=True)

    sku: GpuSkuSpec
    precision: PrecisionFormat
    kv_cache_dtype: Literal["bfloat16", "fp8"]
    sequence_length: int
    weights_gb: float
    lora_slots_gb: float
    runtime_overhead_gb: float
    usable_vram_gb: float
    kv_per_seq_mb: float
    max_concurrent_sequences: int
    fits_in_vram: bool


class BreakEvenScenario(BaseModel):
    """Analytical monthly cost comparison between dedicated vLLM and managed token APIs."""

    model_config = ConfigDict(frozen=True)

    sku_label: str
    dedicated_monthly_usd: float
    avg_prompt_tokens: int
    avg_output_tokens: int
    managed_cost_per_1k_calls_usd: float
    break_even_calls_per_day: int
    effective_cost_at_50k_calls_day_usd: float
    managed_cost_at_50k_calls_day_usd: float
    utilization_duty_cycle: float


class SftTurnMessage(BaseModel):
    """Single message in a native Gemma 4 chat-template training trajectory."""

    model_config = ConfigDict(frozen=True)

    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class SftConversationRecord(BaseModel):
    """Validated supervised fine-tuning example for tool dispatch or memory extraction."""

    model_config = ConfigDict(frozen=True)

    record_id: str
    task_family: Literal["marine_tool_dispatch", "memory_slot_reconcile"]
    chronological_index: int = Field(ge=0)
    messages: list[SftTurnMessage]
    tools: list[dict[str, Any]]


class DatasetCurationReport(BaseModel):
    """Measured curation and schema-validation summary across raw candidate turns."""

    model_config = ConfigDict(frozen=True)

    raw_candidates: int
    accepted_records: int
    rejected_malformed_args: int
    stripped_thought_blocks: int
    train_count: int
    val_count: int


class LoraTargetAudit(BaseModel):
    """Measured module-matching audit across a Gemma 4 multimodal parameter tree."""

    model_config = ConfigDict(frozen=True)

    total_modules_scanned: int
    matched_language_modules: int
    excluded_vision_clippable_modules: int
    excluded_router_gate_modules: int
    trainable_params_millions: float
    trainable_fraction_pct: float
