# Gemma 4 26B-A4B MoE LoRA & vLLM Serving on Vertex AI (Part 3)

Companion codebase for **Part 3** of the agent engineering series on [largo.dev](https://largo.dev/tutorials/agents/gemma-4-lora-vllm-vertex-ai/):

1. **MoE VRAM & Hybrid KV-Cache Calculator (`gemma4_serving/sizing.py`)**: Analytical memory sizing for `google/gemma-4-26B-A4B-it` (`25.2B` resident parameters, `3.8B` active per token, `25` sliding-window layers + `5` global unified-KV layers) across `BF16`, `FP8`, and `4-bit AWQ` on Vertex AI `g2` (`NVIDIA L4`) and `a2` (`NVIDIA A100`) SKUs, plus dedicated node-hour vs managed token-API break-even curves.
2. **Chronological SFT Curation & PEFT Language-Backbone Scoping (`gemma4_serving/sft_data.py`, `train_lora.py`)**: Validates tool-call JSON schemas, strips `<thought>` reasoning blocks, splits trajectories chronologically, and scopes `peft.LoraConfig` via regex so `Gemma4ClippableLinear` vision projections and MoE router gates are untouched.
3. **Google ADK 2.14 `VllmOpenAiLlm` Adapter & Vertex AI Deployment (`gemma4_serving/vllm_adk_adapter.py`, `vertex_deploy.py`)**: Lightweight `BaseLlm` implementation connecting ADK `LlmAgent` and `Runner` directly to vLLM's `/v1/chat/completions` endpoint with native `system` role, OpenAI `tools` JSON schemas, guided decoding, and LoRA adapter alias enforcement, paired with `pytorch-vllm-serve` deployment and `undeploy_all()` teardown helpers.

## Quickstart (Offline Verification & Sizing CLI)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Run lint and hermetic pytest suite
.venv/bin/ruff check gemma4_serving tests
.venv/bin/pytest -v

# Print VRAM matrix, LoRA module audit, SFT curation summary, and ADK loopback latency
.venv/bin/python -m gemma4_serving.cli --seq-len 16384
```
