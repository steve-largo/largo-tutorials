"""Hermetic unit and integration tests for Gemma 4 26B-A4B MoE sizing, LoRA scoping, and ADK vLLM serving."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from google.adk.agents import LlmAgent
from google.adk.models.gemma_llm import Gemma
from google.adk.models.llm_request import LlmRequest
from google.adk.runners import Runner
from google.adk.sessions.sqlite_session_service import SqliteSessionService
from google.genai import types
from pydantic import ValidationError

from gemma4_serving.config import (
    ServingRuntimePolicy,
    ServingSettings,
    validate_artifact_jsonl_path,
)
from gemma4_serving.schemas import PrecisionFormat
from gemma4_serving.sft_data import (
    build_raw_coastal_candidates,
    curate_and_split_sft_dataset,
    export_sft_jsonl,
)
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
from gemma4_serving.vertex_deploy import build_vllm_serving_container_spec
from gemma4_serving.vllm_adk_adapter import (
    VllmOpenAiLlm,
    translate_adk_request_to_openai_payload,
)


def test_security_and_lora_alias_boundaries(tmp_path: Path) -> None:
    # 1. Reject cleartext HTTP to non-loopback hosts (SSRF / token-leak prevention)
    with pytest.raises(ValidationError, match="Cleartext 'http://' is restricted"):
        ServingSettings(VLLM_API_BASE="http://10.0.0.42:8000/v1")

    # 2. Permit HTTPS on remote endpoints
    remote = ServingSettings(
        VLLM_API_BASE="https://gemma-ep.us-central1.run.app/v1",
        GCS_ADAPTER_URI="gs://coastal-ml-artifacts/adapters/marine-v1",
    )
    assert remote.vllm_api_base == "https://gemma-ep.us-central1.run.app/v1"

    # 3. Catch the silent vLLM base-model bypass trap
    with pytest.raises(ValidationError, match="bypasses the loaded LoRA weights"):
        ServingSettings(
            BASE_MODEL_ID="google/gemma-4-26B-A4B-it",
            SERVED_MODEL_NAME="google/gemma-4-26B-A4B-it",
        )

    # 4. Reject directory traversal in dataset export paths
    with pytest.raises(ValueError, match="traversal"):
        validate_artifact_jsonl_path("../escape.jsonl", base_dir=tmp_path)

    # 5. Reject malformed GCS URIs
    with pytest.raises(ValidationError, match="gs://"):
        ServingSettings(GCS_ADAPTER_URI="https://storage.googleapis.com/not-gs-uri")


def test_moe_vram_and_hybrid_kv_cache_invariants() -> None:
    # Resident weights depend on 25.2B total params, so BF16 (~50.4 GB) cannot fit on 1x L4 (24 GB)
    # or 2x L4 (48 GB), whereas FP8 (~26.5 GB) fits cleanly on 2x L4 (`g2-standard-24`).
    l4_x1 = VERTEX_GPU_SKUS[0]
    l4_x2 = VERTEX_GPU_SKUS[1]
    a100_80 = VERTEX_GPU_SKUS[4]

    bf16_2xl4 = calculate_vram_budget(GEMMA4_26B_A4B_SPEC, l4_x2, PrecisionFormat.BF16, 16384)
    fp8_2xl4 = calculate_vram_budget(GEMMA4_26B_A4B_SPEC, l4_x2, PrecisionFormat.FP8, 16384)
    awq_1xl4 = calculate_vram_budget(GEMMA4_26B_A4B_SPEC, l4_x1, PrecisionFormat.AWQ_INT4, 16384)
    bf16_a100 = calculate_vram_budget(GEMMA4_26B_A4B_SPEC, a100_80, PrecisionFormat.BF16, 16384)

    assert not bf16_2xl4.fits_in_vram
    assert fp8_2xl4.fits_in_vram and fp8_2xl4.max_concurrent_sequences == 29
    assert awq_1xl4.fits_in_vram and awq_1xl4.max_concurrent_sequences == 14
    assert bf16_a100.fits_in_vram and bf16_a100.max_concurrent_sequences == 51

    # Hybrid attention (25 sliding-window @ 1024 + 5 global @ 16384) slashes KV cache > 70%
    hybrid_mb = compute_hybrid_kv_cache_mb_per_seq(GEMMA4_26B_A4B_SPEC, 16384, "fp8")
    dense_mb = compute_dense_equivalent_kv_mb(GEMMA4_26B_A4B_SPEC, 16384, "fp8")
    assert hybrid_mb < dense_mb * 0.30

    be = compute_cost_break_even(l4_x2, avg_prompt_tokens=1800, avg_output_tokens=320)
    assert 40_000 <= be.break_even_calls_per_day <= 55_000


def test_lora_regex_excludes_vision_clippable_linear_and_router() -> None:
    compact = audit_lora_target_regex(GEMMA4_ATTN_SHARED_ONLY_REGEX, lora_rank=16)
    assert compact.excluded_vision_clippable_modules == 163
    assert compact.excluded_router_gate_modules == 30
    assert compact.matched_language_modules == 30 * 7  # 4 self_attn + 3 shared_expert per layer

    full = audit_lora_target_regex(GEMMA4_LANGUAGE_LORA_REGEX, lora_rank=16)
    assert full.excluded_vision_clippable_modules == 163
    assert full.excluded_router_gate_modules == 30
    assert full.matched_language_modules == 30 * (7 + 128 * 3)

    # Verify an unscoped regex touching `q_proj` everywhere raises ValueError on vision_tower
    with pytest.raises(ValueError, match="Gemma4ClippableLinear"):
        audit_lora_target_regex(r".*(q_proj|k_proj)$", lora_rank=16)


def test_chronological_sft_curation_and_jsonl_export(tmp_path: Path) -> None:
    train_recs, val_recs, report = curate_and_split_sft_dataset(build_raw_coastal_candidates())
    assert report.raw_candidates == 12
    assert report.rejected_malformed_args == 2
    assert report.accepted_records == 10
    assert report.stripped_thought_blocks == 5
    assert len(train_recs) == 8 and len(val_recs) == 2

    # Verify strict chronological ordering across train -> val split
    assert max(r.chronological_index for r in train_recs) < min(
        r.chronological_index for r in val_recs
    )
    for rec in train_recs + val_recs:
        for msg in rec.messages:
            assert "<thought>" not in msg.content

    out_file = export_sft_jsonl(train_recs, "curated_train.jsonl", base_dir=tmp_path)
    assert len(out_file.read_text(encoding="utf-8").strip().splitlines()) == 8


def test_adk_gemma3_wrapper_vs_gemma4_vllm_request_translation() -> None:
    decl = types.FunctionDeclaration(
        name="fetch_noaa_tides",
        description="Fetch NOAA tide table",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={"station_id": types.Schema(type=types.Type.STRING)},
            required=["station_id"],
        ),
    )
    req_for_vllm = LlmRequest(
        model="marine-triage-gemma4-lora",
        contents=[
            types.Content(
                role="user",
                parts=[types.Part.from_text(text="Check tides at 8723970")],
            )
        ],
        config=types.GenerateContentConfig(
            system_instruction="Be concise.",
            tools=[types.Tool(function_declarations=[decl])],
        ),
    )

    oai_payload = translate_adk_request_to_openai_payload(
        llm_request=req_for_vllm,
        served_model_name="marine-triage-gemma4-lora",
        policy=ServingRuntimePolicy(),
    )
    # VllmOpenAiLlm preserves native `system` role and structured `tools` array
    assert oai_payload["model"] == "marine-triage-gemma4-lora"
    assert oai_payload["messages"][0] == {"role": "system", "content": "Be concise."}
    assert oai_payload["tools"][0]["function"]["name"] == "fetch_noaa_tides"

    # By contrast, ADK's built-in `Gemma` class is Gemma-3-only and wipes `config.tools = []`
    gemma3_wrapper = Gemma(model="gemma-3-27b-it")
    req_for_gemma3 = req_for_vllm.model_copy(deep=True)
    gemma3_wrapper._move_function_calls_into_system_instruction(req_for_gemma3)
    assert req_for_gemma3.config.tools == []


async def test_end_to_end_adk_runner_with_vllm_openai_adapter() -> None:
    seen_models: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        seen_models.append(payload["model"])
        if not any(m.get("role") == "tool" for m in payload["messages"]):
            return httpx.Response(
                200,
                json={
                    "id": "cmpl-1",
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
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "lookup_station",
                                            "arguments": json.dumps({"station_id": "8723970"}),
                                        },
                                    }
                                ],
                            },
                        }
                    ],
                    "usage": {"prompt_tokens": 120, "completion_tokens": 20, "total_tokens": 140},
                },
            )
        return httpx.Response(
            200,
            json={
                "id": "cmpl-2",
                "model": payload["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "Vaca Key tide is +1.9 ft incoming.",
                        },
                    }
                ],
                "usage": {"prompt_tokens": 155, "completion_tokens": 12, "total_tokens": 167},
            },
        )

    def lookup_station(station_id: str) -> dict[str, str]:
        """Look up tide height for a station ID."""
        return {"station_id": station_id, "tide": "+1.9 ft incoming"}

    settings = ServingSettings(
        SERVED_MODEL_NAME="marine-triage-gemma4-lora",
        VLLM_API_BASE="http://127.0.0.1:8000/v1",
        GCS_ADAPTER_URI="gs://coastal-ml-artifacts/adapters/marine-v1",
    )
    spec = build_vllm_serving_container_spec(
        settings=settings,
        policy=ServingRuntimePolicy(),
        sku=VERTEX_GPU_SKUS[1],
        precision=PrecisionFormat.FP8,
    )
    assert "--lora-modules=marine-triage-gemma4-lora=gs://coastal-ml-artifacts/adapters/marine-v1" in spec["serving_container_args"]
    assert "--tensor-parallel-size=2" in spec["serving_container_args"]

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        llm = VllmOpenAiLlm(settings=settings, http_client=client)
        agent = LlmAgent(
            name="marine_vllm_agent",
            model=llm,
            instruction="Check tide stations via tools.",
            tools=[lookup_station],
        )
        sessions = SqliteSessionService(db_path=":memory:")
        try:
            await sessions.create_session(app_name="test_app", user_id="steve", session_id="s1")
            runner = Runner(app_name="test_app", agent=agent, session_service=sessions)

            final_texts: list[str] = []
            async for ev in runner.run_async(
                user_id="steve",
                session_id="s1",
                new_message=types.Content(
                    role="user",
                    parts=[types.Part.from_text(text="Check 8723970")],
                ),
            ):
                if ev.content and ev.content.parts:
                    for p in ev.content.parts:
                        if p.text:
                            final_texts.append(p.text)
        finally:
            await sessions.close()

    assert seen_models == ["marine-triage-gemma4-lora", "marine-triage-gemma4-lora"]
    assert any("+1.9 ft incoming" in t for t in final_texts)
