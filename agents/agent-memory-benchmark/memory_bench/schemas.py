"""Frozen Pydantic contracts for longitudinal session replay, fact reconciliation, and evaluation."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SAFE_SCOPE_ID_PATTERN = r"^[A-Za-z0-9_.-]{1,64}$"
SAFE_SLOT_KEY_PATTERN = r"^[a-z0-9_.:-]{2,80}$"
_SCOPE_ID_RE = re.compile(SAFE_SCOPE_ID_PATTERN)

FactCategory = Literal[
    "vessel_spec",
    "home_marina",
    "crew_safety",
    "tackle_preference",
    "ephemeral_telemetry",
]
ReconcileAction = Literal["ADD", "UPDATE", "NOOP", "IGNORE_EPHEMERAL"]


def validate_tenant_scope(app_name: str, user_id: str) -> tuple[str, str]:
    """Validate (app_name, user_id) identifiers at the service boundary."""
    if not _SCOPE_ID_RE.match(app_name):
        raise ValueError(f"Invalid app_name scope identifier: {app_name!r}")
    if not _SCOPE_ID_RE.match(user_id):
        raise ValueError(f"Invalid user_id scope identifier: {user_id!r}")
    return app_name, user_id


class CandidateFact(BaseModel):
    """A candidate fact extracted from a completed session before reconciliation."""

    model_config = ConfigDict(frozen=True)

    category: FactCategory
    slot_key: str = Field(pattern=SAFE_SLOT_KEY_PATTERN)
    fact_text: str = Field(min_length=3, max_length=400)
    is_ephemeral: bool = False
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class FactExtractionBatch(BaseModel):
    """Structured batch schema for LLM-backed fact extraction."""

    model_config = ConfigDict(frozen=True)

    facts: list[CandidateFact] = Field(default_factory=list)


class CanonicalMemory(BaseModel):
    """A persisted, versioned memory record inside a tenant scope."""

    model_config = ConfigDict(frozen=True)

    memory_id: str = Field(min_length=4, max_length=64)
    app_name: str = Field(pattern=SAFE_SCOPE_ID_PATTERN)
    user_id: str = Field(pattern=SAFE_SCOPE_ID_PATTERN)
    category: FactCategory
    slot_key: str = Field(pattern=SAFE_SLOT_KEY_PATTERN)
    fact_text: str = Field(min_length=3, max_length=400)
    revision: int = Field(ge=1)
    source_session_id: str | None = None
    updated_at: str
    active: bool = True
    superseded_by: str | None = None


class SessionTurn(BaseModel):
    """A single conversational turn inside a historical session."""

    model_config = ConfigDict(frozen=True)

    role: Literal["user", "model"]
    text: str = Field(min_length=1, max_length=2000)
    thought_text: str | None = None


class LongitudinalSession(BaseModel):
    """A timestamped multi-turn conversation in the 8-session coastal benchmark."""

    model_config = ConfigDict(frozen=True)

    session_id: str = Field(pattern=SAFE_SCOPE_ID_PATTERN)
    recorded_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    summary_label: str = Field(min_length=3, max_length=140)
    turns: tuple[SessionTurn, ...]


class MemoryProbe(BaseModel):
    """An evaluation query checking active fact recall, supersession, and noise immunity."""

    model_config = ConfigDict(frozen=True)

    probe_id: str = Field(pattern=SAFE_SCOPE_ID_PATTERN)
    user_id: str = Field(pattern=SAFE_SCOPE_ID_PATTERN)
    query: str = Field(min_length=5, max_length=500)
    required_substrings: tuple[str, ...] = ()
    forbidden_stale_substrings: tuple[str, ...] = ()
    forbidden_noise_substrings: tuple[str, ...] = ()
    expect_empty: bool = False


class ProbeOutcome(BaseModel):
    """Measured retrieval outcome for a single probe against a memory strategy."""

    model_config = ConfigDict(frozen=True)

    probe_id: str
    passed_recall: bool
    has_stale_contradiction: bool
    has_telemetry_noise: bool
    leaked_cross_tenant: bool
    retrieved_count: int
    approx_prompt_tokens: int
    retrieved_snippets: tuple[str, ...] = ()


class StrategyEvaluationSummary(BaseModel):
    """Aggregated scorecard across all evaluation probes for one memory strategy."""

    model_config = ConfigDict(frozen=True)

    strategy_name: str
    probes_total: int
    exact_pass_count: int
    recall_pass_count: int
    stale_contradiction_count: int
    telemetry_noise_count: int
    cross_tenant_leak_count: int
    mean_prompt_tokens: float
    ingest_wall_ms: float
    search_wall_ms: float
    outcomes: tuple[ProbeOutcome, ...]
