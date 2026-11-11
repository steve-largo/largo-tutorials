"""First-principles MoE VRAM, hybrid KV-cache, and token-cost break-even calculations."""

from __future__ import annotations

from typing import Literal

from gemma4_serving.schemas import (
    BreakEvenScenario,
    GpuSkuSpec,
    ModelArchSpec,
    PrecisionFormat,
    VramBreakdown,
)

# Published Gemma 4 26B-A4B-it architectural dimensions (Apache 2.0, Hugging Face config)
GEMMA4_26B_A4B_SPEC = ModelArchSpec(
    model_id="google/gemma-4-26B-A4B-it",
    total_params_billions=25.2,
    active_params_billions=3.8,
    num_layers=30,
    sliding_window_layers=25,
    global_attention_layers=5,
    sliding_window_tokens=1024,
    num_kv_heads=8,
    head_dim=256,
    max_context_tokens=262144,
)

# Published Vertex AI dedicated prediction GPU SKUs (us-central1 on-demand reference rates)
VERTEX_GPU_SKUS: tuple[GpuSkuSpec, ...] = (
    GpuSkuSpec(
        machine_type="g2-standard-12",
        accelerator_type="NVIDIA_L4",
        accelerator_count=1,
        vram_per_gpu_gb=24.0,
        vertex_hourly_usd=1.28,
    ),
    GpuSkuSpec(
        machine_type="g2-standard-24",
        accelerator_type="NVIDIA_L4",
        accelerator_count=2,
        vram_per_gpu_gb=24.0,
        vertex_hourly_usd=2.55,
    ),
    GpuSkuSpec(
        machine_type="g2-standard-48",
        accelerator_type="NVIDIA_L4",
        accelerator_count=4,
        vram_per_gpu_gb=24.0,
        vertex_hourly_usd=5.10,
    ),
    GpuSkuSpec(
        machine_type="a2-highgpu-1g",
        accelerator_type="NVIDIA_TESLA_A100",
        accelerator_count=1,
        vram_per_gpu_gb=40.0,
        vertex_hourly_usd=4.12,
    ),
    GpuSkuSpec(
        machine_type="a2-ultragpu-1g",
        accelerator_type="NVIDIA_A100_80GB",
        accelerator_count=1,
        vram_per_gpu_gb=80.0,
        vertex_hourly_usd=5.85,
    ),
)

_BYTES_PER_PARAM: dict[PrecisionFormat, float] = {
    PrecisionFormat.BF16: 2.0,
    PrecisionFormat.FP8: 1.05,  # 1 byte + per-channel FP16 scale metadata
    PrecisionFormat.AWQ_INT4: 0.56,  # 0.5 byte + group-128 scales and zero-points
}


def compute_hybrid_kv_cache_mb_per_seq(
    arch: ModelArchSpec,
    sequence_length: int,
    kv_cache_dtype: Literal["bfloat16", "fp8"] = "fp8",
) -> float:
    """Compute per-sequence KV-cache memory in MB accounting for hybrid sliding-window layers.

    Local sliding-window layers cap cached tokens at `min(sequence_length, sliding_window_tokens)`,
    whereas global p-RoPE layers retain KV vectors across the full `sequence_length`.
    """
    bytes_per_elem = 2.0 if kv_cache_dtype == "bfloat16" else 1.0
    # 2 tensors (K and V) * num_kv_heads * head_dim * bytes_per_elem
    bytes_per_layer_token = 2.0 * arch.num_kv_heads * arch.head_dim * bytes_per_elem

    local_tokens = min(sequence_length, arch.sliding_window_tokens)
    local_bytes = arch.sliding_window_layers * local_tokens * bytes_per_layer_token
    global_bytes = arch.global_attention_layers * sequence_length * bytes_per_layer_token
    return round((local_bytes + global_bytes) / (1024.0 * 1024.0), 2)


def compute_dense_equivalent_kv_mb(
    arch: ModelArchSpec,
    sequence_length: int,
    kv_cache_dtype: Literal["bfloat16", "fp8"] = "fp8",
) -> float:
    """Compute what KV cache would cost if all 30 layers cached the full sequence."""
    bytes_per_elem = 2.0 if kv_cache_dtype == "bfloat16" else 1.0
    bytes_per_layer_token = 2.0 * arch.num_kv_heads * arch.head_dim * bytes_per_elem
    total_bytes = arch.num_layers * sequence_length * bytes_per_layer_token
    return round(total_bytes / (1024.0 * 1024.0), 2)


def calculate_vram_budget(
    arch: ModelArchSpec,
    sku: GpuSkuSpec,
    precision: PrecisionFormat,
    sequence_length: int = 8192,
    kv_cache_dtype: Literal["bfloat16", "fp8"] = "fp8",
    gpu_memory_utilization: float = 0.92,
    lora_rank: int = 16,
    max_loras: int = 2,
) -> VramBreakdown:
    """Calculate resident MoE weight footprint, LoRA workspace, and max concurrent sequences."""
    weights_gb = round(arch.total_params_billions * _BYTES_PER_PARAM[precision], 2)
    # Pre-allocated vLLM Punica / multi-LoRA GPU slots (BF16 rank-r projections + scratch)
    lora_slots_gb = round(0.18 * (lora_rank / 16.0) * max_loras * sku.accelerator_count, 2)
    # CUDA context, NCCL collectives (when TP > 1), and vLLM CUDAGraph capture workspace
    nccl_gb = 0.85 * sku.accelerator_count if sku.accelerator_count > 1 else 0.0
    runtime_overhead_gb = round(1.6 * sku.accelerator_count + nccl_gb, 2)

    usable_vram_gb = round(sku.total_vram_gb * gpu_memory_utilization, 2)
    kv_per_seq_mb = compute_hybrid_kv_cache_mb_per_seq(
        arch=arch,
        sequence_length=sequence_length,
        kv_cache_dtype=kv_cache_dtype,
    )

    headroom_gb = usable_vram_gb - (weights_gb + lora_slots_gb + runtime_overhead_gb)
    if headroom_gb <= 0.5:
        return VramBreakdown(
            sku=sku,
            precision=precision,
            kv_cache_dtype=kv_cache_dtype,
            sequence_length=sequence_length,
            weights_gb=weights_gb,
            lora_slots_gb=lora_slots_gb,
            runtime_overhead_gb=runtime_overhead_gb,
            usable_vram_gb=usable_vram_gb,
            kv_per_seq_mb=kv_per_seq_mb,
            max_concurrent_sequences=0,
            fits_in_vram=False,
        )

    max_seqs = int((headroom_gb * 1024.0) // kv_per_seq_mb)
    return VramBreakdown(
        sku=sku,
        precision=precision,
        kv_cache_dtype=kv_cache_dtype,
        sequence_length=sequence_length,
        weights_gb=weights_gb,
        lora_slots_gb=lora_slots_gb,
        runtime_overhead_gb=runtime_overhead_gb,
        usable_vram_gb=usable_vram_gb,
        kv_per_seq_mb=kv_per_seq_mb,
        max_concurrent_sequences=max_seqs,
        fits_in_vram=max_seqs >= 4,
    )


def compute_cost_break_even(
    sku: GpuSkuSpec,
    avg_prompt_tokens: int = 1800,
    avg_output_tokens: int = 320,
    managed_input_per_1m_usd: float = 0.30,
    managed_output_per_1m_usd: float = 2.50,
    utilization_duty_cycle: float = 1.0,
) -> BreakEvenScenario:
    """Compare continuous Vertex AI dedicated node-hours against managed Gemini 2.5 Flash."""
    hours_per_month = 730.0
    # When autoscaling to zero on Cloud Run GPU or scheduled batch windows, duty_cycle < 1.0;
    # on a 24/7 Vertex AI Dedicated Endpoint, billing runs at 100% of wall-clock hours (730h/mo).
    billed_monthly_usd = round(sku.vertex_hourly_usd * hours_per_month * utilization_duty_cycle, 2)

    cost_per_call = (
        (avg_prompt_tokens / 1_000_000.0) * managed_input_per_1m_usd
        + (avg_output_tokens / 1_000_000.0) * managed_output_per_1m_usd
    )
    cost_per_1k = round(cost_per_call * 1000.0, 4)

    daily_fixed_usd = billed_monthly_usd / 30.4167
    break_even_day = round(daily_fixed_usd / cost_per_call)

    calls_per_month_at_50k = 50_000 * 30.4167
    managed_monthly_50k = round(calls_per_month_at_50k * cost_per_call, 2)

    return BreakEvenScenario(
        sku_label=f"{sku.machine_type} ({sku.accelerator_count}x {sku.accelerator_type})",
        dedicated_monthly_usd=billed_monthly_usd,
        avg_prompt_tokens=avg_prompt_tokens,
        avg_output_tokens=avg_output_tokens,
        managed_cost_per_1k_calls_usd=cost_per_1k,
        break_even_calls_per_day=break_even_day,
        effective_cost_at_50k_calls_day_usd=billed_monthly_usd,
        managed_cost_at_50k_calls_day_usd=managed_monthly_50k,
        utilization_duty_cycle=utilization_duty_cycle,
    )
