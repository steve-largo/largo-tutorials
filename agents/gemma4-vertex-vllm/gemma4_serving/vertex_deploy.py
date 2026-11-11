"""Validated Vertex AI Model Garden `pytorch-vllm-serve` deployment and teardown builders."""

from __future__ import annotations

from typing import Any

from gemma4_serving.config import ServingRuntimePolicy, ServingSettings
from gemma4_serving.schemas import GpuSkuSpec, PrecisionFormat

VLLM_VERTEX_CONTAINER_URI = (
    "us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/"
    "pytorch-vllm-serve:20260815_0916_RC01"
)


def build_vllm_serving_container_spec(
    settings: ServingSettings,
    policy: ServingRuntimePolicy,
    sku: GpuSkuSpec,
    precision: PrecisionFormat = PrecisionFormat.FP8,
) -> dict[str, Any]:
    """Construct allowlisted `pytorch-vllm-serve` container arguments for Vertex AI."""
    if settings.gcs_adapter_uri is None:
        raise ValueError(
            "settings.gcs_adapter_uri ('gs://<bucket>/<adapter-dir>') is required when "
            "building a multi-LoRA Vertex AI serving container spec."
        )

    args: list[str] = [
        f"--model={settings.base_model_id}",
        f"--tensor-parallel-size={sku.accelerator_count}",
        f"--gpu-memory-utilization={policy.gpu_memory_utilization}",
        f"--max-model-len={policy.max_model_len}",
        "--kv-cache-dtype=fp8",
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--enable-auto-tool-choice",
        "--tool-call-parser=gemma4",
        "--enable-lora",
        f"--max-lora-rank={policy.lora_rank}",
        f"--lora-modules={settings.lora_adapter_alias}={settings.gcs_adapter_uri}",
    ]

    if precision == PrecisionFormat.FP8:
        args.append("--quantization=fp8")
    elif precision == PrecisionFormat.AWQ_INT4:
        args.append("--quantization=awq")
    else:
        args.append("--dtype=bfloat16")

    return {
        "display_name": f"gemma4-26b-a4b-{settings.lora_adapter_alias}",
        "serving_container_image_uri": VLLM_VERTEX_CONTAINER_URI,
        "serving_container_command": ["python3", "-m", "vllm.entrypoints.openai.api_server"],
        "serving_container_args": args,
        "serving_container_ports": [8000],
        "serving_container_health_route": "/health",
        "serving_container_predict_route": "/v1/chat/completions",
        "dedicated_endpoint_enabled": True,
        "machine_type": sku.machine_type,
        "accelerator_type": sku.accelerator_type,
        "accelerator_count": sku.accelerator_count,
    }


def deploy_to_vertex_endpoint(
    settings: ServingSettings,
    policy: ServingRuntimePolicy,
    sku: GpuSkuSpec,
    precision: PrecisionFormat = PrecisionFormat.FP8,
) -> Any:
    """Upload the vLLM serving spec and deploy to a dedicated Vertex AI Endpoint."""
    if not settings.google_cloud_project:
        raise ValueError("GOOGLE_CLOUD_PROJECT must be set before calling deploy_to_vertex_endpoint.")

    from google.cloud import aiplatform  # Optional `vertex` extra

    spec = build_vllm_serving_container_spec(
        settings=settings,
        policy=policy,
        sku=sku,
        precision=precision,
    )
    aiplatform.init(
        project=settings.google_cloud_project,
        location=settings.google_cloud_location,
    )
    model = aiplatform.Model.upload(
        display_name=spec["display_name"],
        serving_container_image_uri=spec["serving_container_image_uri"],
        serving_container_command=spec["serving_container_command"],
        serving_container_args=spec["serving_container_args"],
        serving_container_ports=spec["serving_container_ports"],
        serving_container_health_route=spec["serving_container_health_route"],
        serving_container_predict_route=spec["serving_container_predict_route"],
    )
    endpoint = aiplatform.Endpoint.create(
        display_name=f"{spec['display_name']}-ep",
        dedicated_endpoint_enabled=True,
    )
    model.deploy(
        endpoint=endpoint,
        deployed_model_display_name=settings.lora_adapter_alias,
        machine_type=spec["machine_type"],
        accelerator_type=spec["accelerator_type"],
        accelerator_count=spec["accelerator_count"],
        traffic_percentage=100,
        sync=True,
    )
    return endpoint


def teardown_vertex_endpoint(endpoint: Any, delete_endpoint: bool = True) -> None:
    """Undeploy all GPU replicas immediately to stop continuous hourly node billing."""
    endpoint.undeploy_all(sync=True)
    if delete_endpoint:
        endpoint.delete()
