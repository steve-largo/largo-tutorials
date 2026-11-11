"""Language-backbone PEFT LoRA targeting and TRL SFTTrainer recipe for Gemma 4 26B-A4B MoE."""

from __future__ import annotations

import re
from typing import Any

from gemma4_serving.config import ServingRuntimePolicy, ServingSettings
from gemma4_serving.schemas import LoraTargetAudit

# Explicitly scope LoRA adapters to language_model attention and MLP projections so PEFT
# never touches `vision_tower` / `multi_modal_projector` `Gemma4ClippableLinear` wrappers
# or destabilizes sparse MoE top-8 `router` gating logits during small-corpus fine-tuning.
GEMMA4_LANGUAGE_LORA_REGEX = (
    r"^model\.language_model\.layers\.\d+\."
    r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|"
    r"(mlp|shared_expert|experts\.\d+)\.(gate_proj|up_proj|down_proj))$"
)

GEMMA4_ATTN_SHARED_ONLY_REGEX = (
    r"^model\.language_model\.layers\.\d+\."
    r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|"
    r"shared_expert\.(gate_proj|up_proj|down_proj))$"
)


def build_synthetic_gemma4_module_tree(
    num_layers: int = 30,
    num_routed_experts: int = 128,
    vision_layers: int = 27,
) -> list[tuple[str, str, int, int]]:
    """Construct the module name, class type, and `(in_features, out_features)` table for auditing.

    Mirrors `google/gemma-4-26B-A4B-it` layout:
    - `model.vision_tower.*` using `Gemma4ClippableLinear`
    - `model.language_model.layers.{0..29}.self_attn.{q,k,v,o}_proj`
    - `model.language_model.layers.{0..29}.mlp.router` (128-way top-8 gating Linear)
    - `model.language_model.layers.{0..29}.shared_expert.{gate,up,down}_proj`
    - `model.language_model.layers.{0..29}.experts.{0..127}.{gate,up,down}_proj`
    """
    modules: list[tuple[str, str, int, int]] = []

    for v_idx in range(vision_layers):
        for proj in ("q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"):
            modules.append(
                (
                    f"model.vision_tower.encoder.layers.{v_idx}.{proj}",
                    "Gemma4ClippableLinear",
                    1152,
                    1152,
                )
            )
    modules.append(
        ("model.multi_modal_projector.linear", "Gemma4ClippableLinear", 1152, 2560)
    )

    hidden = 2560
    kv_dim = 8 * 256  # 2048
    q_dim = 16 * 256  # 4096
    expert_inter = 768
    shared_inter = 1536

    for l_idx in range(num_layers):
        modules.append(
            (f"model.language_model.layers.{l_idx}.self_attn.q_proj", "Linear", hidden, q_dim)
        )
        modules.append(
            (f"model.language_model.layers.{l_idx}.self_attn.k_proj", "Linear", hidden, kv_dim)
        )
        modules.append(
            (f"model.language_model.layers.{l_idx}.self_attn.v_proj", "Linear", hidden, kv_dim)
        )
        modules.append(
            (f"model.language_model.layers.{l_idx}.self_attn.o_proj", "Linear", q_dim, hidden)
        )
        modules.append(
            (
                f"model.language_model.layers.{l_idx}.mlp.router",
                "Gemma4RouterLinear",
                hidden,
                num_routed_experts,
            )
        )
        modules.append(
            (
                f"model.language_model.layers.{l_idx}.shared_expert.gate_proj",
                "Linear",
                hidden,
                shared_inter,
            )
        )
        modules.append(
            (
                f"model.language_model.layers.{l_idx}.shared_expert.up_proj",
                "Linear",
                hidden,
                shared_inter,
            )
        )
        modules.append(
            (
                f"model.language_model.layers.{l_idx}.shared_expert.down_proj",
                "Linear",
                shared_inter,
                hidden,
            )
        )
        for e_idx in range(num_routed_experts):
            modules.append(
                (
                    f"model.language_model.layers.{l_idx}.experts.{e_idx}.gate_proj",
                    "Linear",
                    hidden,
                    expert_inter,
                )
            )
            modules.append(
                (
                    f"model.language_model.layers.{l_idx}.experts.{e_idx}.up_proj",
                    "Linear",
                    hidden,
                    expert_inter,
                )
            )
            modules.append(
                (
                    f"model.language_model.layers.{l_idx}.experts.{e_idx}.down_proj",
                    "Linear",
                    expert_inter,
                    hidden,
                )
            )

    return modules


def audit_lora_target_regex(
    target_regex: str = GEMMA4_ATTN_SHARED_ONLY_REGEX,
    lora_rank: int = 16,
    total_model_params_billions: float = 25.2,
) -> LoraTargetAudit:
    """Verify module scoping safety and compute exact LoRA trainable parameter count."""
    compiled = re.compile(target_regex)
    tree = build_synthetic_gemma4_module_tree()

    matched = 0
    excluded_vision = 0
    excluded_router = 0
    trainable_params = 0

    for name, cls_name, in_f, out_f in tree:
        is_hit = bool(compiled.match(name))
        if cls_name == "Gemma4ClippableLinear":
            if is_hit:
                raise ValueError(
                    f"Unsafe LoRA regex '{target_regex}' matched vision module '{name}' "
                    f"({cls_name}), which raises a PEFT unsupported-layer ValueError."
                )
            excluded_vision += 1
            continue

        if name.endswith(".mlp.router"):
            if is_hit:
                raise ValueError(
                    f"Unsafe LoRA regex '{target_regex}' matched MoE router '{name}'."
                )
            excluded_router += 1
            continue

        if is_hit:
            matched += 1
            # LoRA adds A (in_f x r) + B (r x out_f) parameters per targeted Linear layer
            trainable_params += lora_rank * (in_f + out_f)

    trainable_m = round(trainable_params / 1_000_000.0, 2)
    pct = round((trainable_params / (total_model_params_billions * 1_000_000_000.0)) * 100.0, 4)
    return LoraTargetAudit(
        total_modules_scanned=len(tree),
        matched_language_modules=matched,
        excluded_vision_clippable_modules=excluded_vision,
        excluded_router_gate_modules=excluded_router,
        trainable_params_millions=trainable_m,
        trainable_fraction_pct=pct,
    )


def build_peft_and_sft_kwargs(
    settings: ServingSettings,
    policy: ServingRuntimePolicy,
    output_dir: str,
    scope: str = "attn_and_shared_expert",
) -> dict[str, Any]:
    """Build validated keyword dictionaries for `peft.LoraConfig` and `trl.SFTConfig`.

    Notes on Gemma 4 + TRL compatibility (`transformers>=5.5.2`, `trl>=1.1.0`):
    - Use `target_modules` regex scoping to `model.language_model` so `Gemma4ClippableLinear`
      in the vision tower is untouched.
    - Keep `gradient_checkpointing=True` with `gradient_checkpointing_kwargs={'use_reentrant': False}`
      so shared global KV layers propagate gradients cleanly.
    """
    regex = (
        GEMMA4_ATTN_SHARED_ONLY_REGEX
        if scope == "attn_and_shared_expert"
        else GEMMA4_LANGUAGE_LORA_REGEX
    )
    return {
        "base_model_id": settings.base_model_id,
        "lora_config_kwargs": {
            "r": policy.lora_rank,
            "lora_alpha": policy.lora_alpha,
            "lora_dropout": policy.lora_dropout,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "target_modules": regex,
        },
        "sft_config_kwargs": {
            "output_dir": output_dir,
            "max_length": 4096,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 8,
            "learning_rate": 1.5e-4,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.08,
            "num_train_epochs": 3,
            "bf16": True,
            "logging_steps": 5,
            "eval_strategy": "steps",
            "eval_steps": 20,
            "save_strategy": "steps",
            "save_steps": 20,
            "gradient_checkpointing": True,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "assistant_only_loss": True,
        },
    }
