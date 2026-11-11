"""Immutable startup configuration and versioned runtime policy boundaries."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_ALIAS_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{1,63}$")
_HF_REPO_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_GCS_URI_PATTERN = re.compile(r"^gs://[a-z0-9][a-z0-9._-]{1,61}[a-z0-9]/[A-Za-z0-9._/-]+$")


def validate_artifact_jsonl_path(raw_path: str, base_dir: Path | None = None) -> Path:
    """Resolve and validate a local `.jsonl` output path against directory traversal."""
    if not raw_path or "\x00" in raw_path:
        raise ValueError("Dataset path must be non-empty and contain no null bytes.")
    candidate = Path(raw_path)
    if ".." in candidate.parts:
        raise ValueError("Parent directory traversal ('..') is forbidden in dataset paths.")
    if candidate.suffix != ".jsonl":
        raise ValueError("Dataset artifact path must end with '.jsonl'.")

    anchor = (base_dir or Path.cwd()).resolve()
    resolved = (anchor / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    if base_dir is not None:
        prefix = str(anchor) + "/"
        if str(resolved) != str(anchor) and not str(resolved).startswith(prefix):
            raise ValueError(f"Resolved path '{resolved}' escapes allowed directory '{anchor}'.")
    return resolved


class ServingSettings(BaseSettings):
    """Process-startup infrastructure settings loaded strictly from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="",
        frozen=True,
        extra="ignore",
    )

    google_cloud_project: str | None = Field(default=None, alias="GOOGLE_CLOUD_PROJECT")
    google_cloud_location: str = Field(default="us-central1", alias="GOOGLE_CLOUD_LOCATION")
    base_model_id: str = Field(default="google/gemma-4-26B-A4B-it", alias="BASE_MODEL_ID")
    lora_adapter_alias: str = Field(
        default="marine-triage-gemma4-lora",
        alias="LORA_ADAPTER_ALIAS",
    )
    served_model_name: str = Field(
        default="marine-triage-gemma4-lora",
        alias="SERVED_MODEL_NAME",
    )
    vllm_api_base: str = Field(
        default="http://127.0.0.1:8000/v1",
        alias="VLLM_API_BASE",
    )
    vllm_api_key: str | None = Field(default=None, alias="VLLM_API_KEY")
    gcs_adapter_uri: str | None = Field(default=None, alias="GCS_ADAPTER_URI")
    request_timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        alias="REQUEST_TIMEOUT_SECONDS",
    )

    @field_validator("base_model_id")
    @classmethod
    def check_base_model_id(cls, value: str) -> str:
        if not _HF_REPO_PATTERN.match(value):
            raise ValueError(
                f"base_model_id '{value}' must match '<namespace>/<model>' format."
            )
        return value

    @field_validator("lora_adapter_alias")
    @classmethod
    def check_lora_alias(cls, value: str) -> str:
        if not _ALIAS_PATTERN.match(value):
            raise ValueError(
                "lora_adapter_alias must contain only alphanumeric, '.', '_', or '-' characters."
            )
        return value

    @field_validator("vllm_api_base")
    @classmethod
    def check_vllm_api_base(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme not in ("https", "http"):
            raise ValueError("vllm_api_base must use 'https://' or loopback 'http://'.")
        if not parsed.hostname:
            raise ValueError("vllm_api_base must include a valid hostname.")
        if parsed.scheme == "http" and parsed.hostname not in ("127.0.0.1", "localhost"):
            raise ValueError(
                "Cleartext 'http://' is restricted to loopback (127.0.0.1 or localhost); "
                "remote vLLM endpoints must use 'https://'."
            )
        if not parsed.path.rstrip("/").endswith("/v1"):
            raise ValueError("vllm_api_base must end with '/v1' for OpenAI compatibility.")
        return value.rstrip("/")

    @field_validator("gcs_adapter_uri")
    @classmethod
    def check_gcs_uri(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if ".." in value or not _GCS_URI_PATTERN.match(value):
            raise ValueError("gcs_adapter_uri must be a normalized 'gs://<bucket>/<prefix>' URI.")
        return value.rstrip("/")

    @model_validator(mode="after")
    def reject_silent_lora_bypass(self) -> ServingSettings:
        """Catch the vLLM trap where passing base_model_id silently skips the LoRA adapter."""
        if self.served_model_name == self.base_model_id:
            raise ValueError(
                f"served_model_name ('{self.served_model_name}') matches base_model_id "
                f"('{self.base_model_id}'). In vLLM multi-LoRA serving, sending the base "
                "model ID bypasses the loaded LoRA weights; pass lora_adapter_alias instead."
            )
        return self


class ServingRuntimePolicy(BaseModel):
    """Versioned runtime operational policy decoupled from static container image config."""

    model_config = ConfigDict(frozen=True)

    policy_version: str = "2026.11.1"
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=512, ge=32, le=8192)
    lora_rank: int = Field(default=16, ge=4, le=128)
    lora_alpha: int = Field(default=32, ge=4, le=256)
    lora_dropout: float = Field(default=0.05, ge=0.0, le=0.5)
    gpu_memory_utilization: float = Field(default=0.92, ge=0.5, le=0.98)
    max_model_len: int = Field(default=16384, ge=1024, le=262144)
    strip_thought_blocks: bool = True
