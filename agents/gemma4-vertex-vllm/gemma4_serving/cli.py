"""CLI runner printing MoE VRAM budgets, LoRA module audits, break-even math, and ADK loopback timing."""

from __future__ import annotations

import argparse
import asyncio
import json
import time

import httpx
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions.sqlite_session_service import SqliteSessionService
from google.genai import types

from gemma4_serving.config import ServingRuntimePolicy, ServingSettings
from gemma4_serving.schemas import PrecisionFormat
from gemma4_serving.sft_data import build_raw_coastal_candidates, curate_and_split_sft_dataset
from gemma4_serving.sizing import (
    GEMMA4_26B_A4B_SPEC,
    VERTEX_GPU_SKUS,
    calculate_vram_budget,
    compute_cost_break_even,
    compute_dense_equivalent_kv_mb,
    compute_hybrid_kv_cache_mb_per_seq,
)
from gemma4_serving.train_lora import (
    GEMMA4_ATTN_SHARED_ONLY_REGEX,
    GEMMA4_LANGUAGE_LORA_REGEX,
    audit_lora_target_regex,
)
from gemma4_serving.vllm_adk_adapter import VllmOpenAiLlm


def fetch_noaa_tides(station_id: str, hours_ahead: int) -> dict[str, str | int]:
    """Return high/low tide predictions for a 7-digit NOAA coastal station."""
    return {
        "station_id": station_id,
        "hours_ahead": hours_ahead,
        "summary": "High tide 06:42 EDT (+1.9 ft), Low tide 12:58 EDT (+0.2 ft)",
    }


async def run_loopback_adk_vllm_probe(iterations: int = 10) -> tuple[float, int]:
    """Measure end-to-end ADK Runner + SqliteSessionService + VllmOpenAiLlm tool round-trips."""
    call_counter = 0

    def mock_vllm_handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_counter
        call_counter += 1
        payload = json.loads(request.content.decode("utf-8"))
        messages = payload["messages"]
        has_tool_reply = any(m.get("role") == "tool" for m in messages)
        if not has_tool_reply:
            body = {
                "id": f"chatcmpl-{call_counter}",
                "object": "chat.completion",
                "model": payload["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_tide_01",
                                    "type": "function",
                                    "function": {
                                        "name": "fetch_noaa_tides",
                                        "arguments": json.dumps(
                                            {"station_id": "8723970", "hours_ahead": 12}
                                        ),
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 312, "completion_tokens": 28, "total_tokens": 340},
            }
        else:
            body = {
                "id": f"chatcmpl-{call_counter}",
                "object": "chat.completion",
                "model": payload["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": (
                                "Station 8723970 (Vaca Key) peaks at 06:42 EDT (+1.9 ft) "
                                "with low water at 12:58 EDT (+0.2 ft)."
                            ),
                        },
                    }
                ],
                "usage": {"prompt_tokens": 368, "completion_tokens": 31, "total_tokens": 399},
            }
        return httpx.Response(200, json=body)

    settings = ServingSettings(
        SERVED_MODEL_NAME="marine-triage-gemma4-lora",
        VLLM_API_BASE="http://127.0.0.1:8000/v1",
    )
    policy = ServingRuntimePolicy()
    transport = httpx.MockTransport(mock_vllm_handler)

    async with httpx.AsyncClient(transport=transport) as client:
        llm = VllmOpenAiLlm(settings=settings, policy=policy, http_client=client)
        agent = LlmAgent(
            name="coastal_gemma4_agent",
            model=llm,
            instruction="Use fetch_noaa_tides for station tide lookups.",
            tools=[fetch_noaa_tides],
        )
        session_service = SqliteSessionService(db_path=":memory:")
        try:
            runner = Runner(
                app_name="largo_part3_bench",
                agent=agent,
                session_service=session_service,
            )

            latencies_ms: list[float] = []
            for idx in range(iterations):
                sid = f"sess-{idx}"
                await session_service.create_session(
                    app_name="largo_part3_bench",
                    user_id="captain_steve",
                    session_id=sid,
                )
                t0 = time.perf_counter()
                user_msg = types.Content(
                    role="user",
                    parts=[types.Part.from_text(text="Check 12h tides at Vaca Key 8723970.")],
                )
                async for _event in runner.run_async(
                    user_id="captain_steve",
                    session_id=sid,
                    new_message=user_msg,
                ):
                    pass
                latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        finally:
            await session_service.close()

    mean_ms = round(sum(latencies_ms) / len(latencies_ms), 2)
    return mean_ms, call_counter


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemma 4 26B-A4B MoE + vLLM sizing & audit CLI")
    parser.add_argument(
        "--seq-len",
        type=int,
        default=16384,
        help="Target max sequence length per concurrent agent session (default: 16384)",
    )
    args = parser.parse_args()

    print(f"=== 1. Analytical VRAM & Concurrency Matrix (seq_len={args.seq_len} tokens) ===")
    hybrid_kv = compute_hybrid_kv_cache_mb_per_seq(GEMMA4_26B_A4B_SPEC, args.seq_len, "fp8")
    dense_kv = compute_dense_equivalent_kv_mb(GEMMA4_26B_A4B_SPEC, args.seq_len, "fp8")
    print(
        f"Hybrid KV cache / seq (FP8): {hybrid_kv:.2f} MB "
        f"(vs {dense_kv:.2f} MB if all 30 layers cached full global context -> "
        f"{round((1 - hybrid_kv / dense_kv) * 100, 1)}% KV reduction)\n"
    )

    header = (
        f"{'Vertex AI SKU':<20} | {'GPUs':<14} | {'Precision':<9} | "
        f"{'Weights':<8} | {'Usable':<8} | {'Max Seqs':<8} | {'Fits'}"
    )
    print(header)
    print("-" * len(header))
    combos = [
        (VERTEX_GPU_SKUS[0], PrecisionFormat.FP8),
        (VERTEX_GPU_SKUS[0], PrecisionFormat.AWQ_INT4),
        (VERTEX_GPU_SKUS[1], PrecisionFormat.BF16),
        (VERTEX_GPU_SKUS[1], PrecisionFormat.FP8),
        (VERTEX_GPU_SKUS[3], PrecisionFormat.FP8),
        (VERTEX_GPU_SKUS[4], PrecisionFormat.BF16),
    ]
    for sku, prec in combos:
        b = calculate_vram_budget(
            arch=GEMMA4_26B_A4B_SPEC,
            sku=sku,
            precision=prec,
            sequence_length=args.seq_len,
            kv_cache_dtype="fp8",
        )
        gpus = f"{sku.accelerator_count}x {sku.accelerator_type.replace('NVIDIA_', '')}"
        print(
            f"{sku.machine_type:<20} | {gpus:<14} | {prec.value:<9} | "
            f"{b.weights_gb:>5.1f} GB | {b.usable_vram_gb:>5.1f} GB | "
            f"{b.max_concurrent_sequences:>8d} | {'YES' if b.fits_in_vram else 'NO'}"
        )

    print("\n=== 2. Measured Gemma 4 Parameter-Tree LoRA Scoping Audit ===")
    audit_compact = audit_lora_target_regex(GEMMA4_ATTN_SHARED_ONLY_REGEX, lora_rank=16)
    audit_full = audit_lora_target_regex(GEMMA4_LANGUAGE_LORA_REGEX, lora_rank=16)
    print(
        f"Attention + Shared Expert (r=16): matched={audit_compact.matched_language_modules} "
        f"Linear layers, trainable={audit_compact.trainable_params_millions}M params "
        f"({audit_compact.trainable_fraction_pct}%), vision_excluded="
        f"{audit_compact.excluded_vision_clippable_modules}, router_excluded="
        f"{audit_compact.excluded_router_gate_modules}"
    )
    print(
        f"All 128 Routed Experts + Attn (r=16): matched={audit_full.matched_language_modules} "
        f"Linear layers, trainable={audit_full.trainable_params_millions}M params "
        f"({audit_full.trainable_fraction_pct}%), vision_excluded="
        f"{audit_full.excluded_vision_clippable_modules}"
    )

    print("\n=== 3. Measured SFT Trajectory Curation & Schema Audit ===")
    _, _, cur_report = curate_and_split_sft_dataset(build_raw_coastal_candidates())
    print(cur_report.model_dump())

    print("\n=== 4. Analytical Dedicated Endpoint vs Managed Gemini 2.5 Flash Break-Even ===")
    for sku in (VERTEX_GPU_SKUS[0], VERTEX_GPU_SKUS[1], VERTEX_GPU_SKUS[4]):
        be = compute_cost_break_even(sku=sku, avg_prompt_tokens=1800, avg_output_tokens=320)
        print(
            f"{be.sku_label:<36} | ${be.dedicated_monthly_usd:>7.2f}/mo (24/7) | "
            f"Break-even: {be.break_even_calls_per_day:>6,d} calls/day | "
            f"At 50k/day: ${be.effective_cost_at_50k_calls_day_usd:,.0f} vs "
            f"${be.managed_cost_at_50k_calls_day_usd:,.0f} managed"
        )

    print("\n=== 5. Measured ADK 2.14 Runner + VllmOpenAiLlm Loopback Wire Overhead ===")
    mean_ms, total_http = asyncio.run(run_loopback_adk_vllm_probe(iterations=10))
    print(
        f"10 two-turn tool-calling sessions ({total_http} OpenAI /v1/chat/completions calls): "
        f"mean ADK + SQLite + JSON schema translation time = {mean_ms:.2f} ms/session"
    )


if __name__ == "__main__":
    main()
