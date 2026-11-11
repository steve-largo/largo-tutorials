"""Immutable runtime configuration and boundary validators for memory_bench."""

from __future__ import annotations

import pathlib
import re
from functools import lru_cache

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_AGENT_ENGINE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_ALLOWED_SQLITE_SUFFIXES = frozenset({".db", ".sqlite", ".sqlite3"})


class MemoryBenchSettings(BaseSettings):
    """Startup settings loaded from environment variables with zero hardcoded project IDs."""

    model_config = SettingsConfigDict(
        frozen=True,
        env_prefix="MEMORY_BENCH_",
        extra="ignore",
    )

    google_cloud_project: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "GOOGLE_CLOUD_PROJECT",
            "MEMORY_BENCH_GOOGLE_CLOUD_PROJECT",
        ),
    )
    google_cloud_location: str = Field(
        default="us-central1",
        validation_alias=AliasChoices(
            "GOOGLE_CLOUD_LOCATION",
            "MEMORY_BENCH_GOOGLE_CLOUD_LOCATION",
        ),
    )
    use_vertexai: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "GOOGLE_GENAI_USE_VERTEXAI",
            "MEMORY_BENCH_USE_VERTEXAI",
        ),
    )
    agent_engine_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "AGENT_ENGINE_ID",
            "MEMORY_BENCH_AGENT_ENGINE_ID",
        ),
    )
    gemini_model: str = Field(default="gemini-2.5-flash")
    embedding_model: str = Field(default="gemini-embedding-001")
    sqlite_path: str = Field(default=":memory:")
    sliding_window_sessions: int = Field(default=2, ge=1, le=20)
    vector_top_k: int = Field(default=4, ge=1, le=20)
    min_vector_similarity: float = Field(default=0.18, ge=0.0, le=1.0)

    @field_validator("agent_engine_id")
    @classmethod
    def validate_agent_engine_id(cls, value: str | None) -> str | None:
        if value is None or value == "":
            return None
        if "/" in value or not _AGENT_ENGINE_ID_PATTERN.match(value):
            raise ValueError(
                "agent_engine_id must be the bare ReasoningEngine ID segment "
                "(extract via agent_engine.api_resource.name.split('/')[-1]), "
                f"got: {value!r}"
            )
        return value

    @field_validator("sqlite_path")
    @classmethod
    def validate_sqlite_path(cls, value: str) -> str:
        if value == ":memory:":
            return value
        raw_parts = pathlib.PurePath(value).parts
        if ".." in raw_parts:
            raise ValueError(f"Path traversal ('..') is not allowed in sqlite_path: {value!r}")
        suffix = pathlib.PurePath(value).suffix.lower()
        if suffix not in _ALLOWED_SQLITE_SUFFIXES:
            raise ValueError(
                f"sqlite_path must be ':memory:' or end in {sorted(_ALLOWED_SQLITE_SUFFIXES)}, "
                f"got: {value!r}"
            )
        return value


@lru_cache(maxsize=1)
def load_settings() -> MemoryBenchSettings:
    """Return cached process-level settings."""
    return MemoryBenchSettings()
