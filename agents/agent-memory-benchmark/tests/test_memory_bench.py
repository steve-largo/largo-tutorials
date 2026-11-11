"""Hermetic unit, security, and ADK Runner integration tests for memory_bench."""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest
from google.adk.events.event import Event
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions.sqlite_session_service import SqliteSessionService
from google.genai import types
from pydantic import Field, ValidationError

from memory_bench.config import MemoryBenchSettings
from memory_bench.dataset import BENCHMARK_APP_NAME, PRIMARY_USER_ID
from memory_bench.evaluator import run_comparative_benchmark
from memory_bench.schemas import validate_tenant_scope
from memory_bench.services import (
    ConsolidatedMemoryService,
    SqliteVectorMemoryService,
    build_memory_aware_captain_agent,
)


class RecordingFakeLlm(BaseLlm):
    """Deterministic ADK BaseLlm test double that records incoming LlmRequests."""

    model: str = "fake-gemini-memory"
    requests: list[LlmRequest] = Field(default_factory=list)
    issue_load_memory_call: bool = False

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        _ = stream
        self.requests.append(llm_request.model_copy(deep=True))

        # In on-demand LoadMemoryTool mode, emit a tool call on turn 1 before answering on turn 2
        if self.issue_load_memory_call and len(self.requests) == 1:
            yield LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part.from_function_call(
                            name="load_memory",
                            args={"query": "boat draft inches marina"},
                        )
                    ],
                )
            )
            return

        yield LlmResponse(
            content=types.Content(
                role="model",
                parts=[
                    types.Part(
                        text="Based on your 16-inch draft Contender at Bud N' Mary's, wait for rising water."
                    )
                ],
            )
        )


def test_security_boundary_validators() -> None:
    """Verify agent_engine_id, sqlite_path, and tenant scope validators block unsafe inputs."""
    with pytest.raises(ValidationError):
        MemoryBenchSettings(
            agent_engine_id="projects/demo/locations/us-central1/reasoningEngines/999"
        )

    with pytest.raises(ValidationError):
        MemoryBenchSettings(sqlite_path="../outside/escaped.db")

    with pytest.raises(ValidationError):
        MemoryBenchSettings(sqlite_path="/tmp/not_a_database.txt")

    with pytest.raises(ValueError, match="Invalid user_id"):
        validate_tenant_scope("keys_marine_captain", "captain_steve' OR '1'='1")

    valid = MemoryBenchSettings(agent_engine_id="4829104", sqlite_path="local_store.db")
    assert valid.agent_engine_id == "4829104"


@pytest.mark.asyncio
async def test_adk_sqlite_session_service_four_scope_hierarchy(tmp_path) -> None:
    """Verify ADK SqliteSessionService persists user: keys across sessions while isolating session keys."""
    db_file = tmp_path / "sessions.db"
    session_service = SqliteSessionService(db_path=str(db_file))

    s1 = await session_service.create_session(
        app_name=BENCHMARK_APP_NAME,
        user_id=PRIMARY_USER_ID,
        session_id="session-may",
        state={
            "user:preferred_units": "imperial_feet_mph",
            "app:tide_datum": "MLLW",
            "current_inlet": "Snake Creek",
        },
    )
    assert s1.state["user:preferred_units"] == "imperial_feet_mph"

    # A brand-new session for the same user inherits user: and app: keys, not unprefixed session keys
    s2 = await session_service.create_session(
        app_name=BENCHMARK_APP_NAME,
        user_id=PRIMARY_USER_ID,
        session_id="session-nov",
    )
    assert s2.state.get("user:preferred_units") == "imperial_feet_mph"
    assert s2.state.get("app:tide_datum") == "MLLW"
    assert "current_inlet" not in s2.state

    # Different user_id only sees app: state, never captain_steve's user: state
    s_other = await session_service.create_session(
        app_name=BENCHMARK_APP_NAME,
        user_id="angler_bob",
        session_id="session-bob",
    )
    assert "user:preferred_units" not in s_other.state
    assert s_other.state.get("app:tide_datum") == "MLLW"


@pytest.mark.asyncio
async def test_thought_parts_stripped_before_memory_ingestion() -> None:
    """Ensure internal model reasoning thoughts (part.thought=True) never enter vector or consolidated memory."""
    vec_service = SqliteVectorMemoryService(sqlite_path=":memory:")
    cons_service = ConsolidatedMemoryService(sqlite_path=":memory:")
    try:
        events = [
            Event(
                invocation_id="inv-1",
                author="user",
                content=types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            text="I run a 25ft Contender bay boat that draws 16 inches."
                        )
                    ],
                ),
            ),
            Event(
                invocation_id="inv-1",
                author="KeysMemoryCaptain",
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            text="SECRET_CHAIN_OF_THOUGHT_TOKEN_42",
                            thought=True,
                        ),
                        types.Part(text="Logged your 16-inch draft Contender."),
                    ],
                ),
            ),
        ]
        await vec_service.add_events_to_memory(
            app_name=BENCHMARK_APP_NAME,
            user_id=PRIMARY_USER_ID,
            events=events,
            session_id="sess-thought",
        )
        await cons_service.add_events_to_memory(
            app_name=BENCHMARK_APP_NAME,
            user_id=PRIMARY_USER_ID,
            events=events,
            session_id="sess-thought",
        )

        vec_res = await vec_service.search_memory(
            app_name=BENCHMARK_APP_NAME,
            user_id=PRIMARY_USER_ID,
            query="Contender boat draft SECRET_CHAIN_OF_THOUGHT_TOKEN_42",
        )
        for mem in vec_res.memories:
            assert "SECRET_CHAIN_OF_THOUGHT_TOKEN_42" not in (mem.content.parts[0].text or "")

        cons_res = await cons_service.search_memory(
            app_name=BENCHMARK_APP_NAME,
            user_id=PRIMARY_USER_ID,
            query="Contender boat draft",
        )
        assert len(cons_res.memories) == 1
        assert "16 inches" in (cons_res.memories[0].content.parts[0].text or "")
    finally:
        await vec_service.close()
        await cons_service.close()


@pytest.mark.asyncio
async def test_adk_runner_preload_vs_load_memory_round_trips(tmp_path) -> None:
    """Verify PreloadMemoryTool uses 1 LLM call while LoadMemoryTool uses 2 LLM calls when invoked."""
    memory_service = ConsolidatedMemoryService(sqlite_path=":memory:")
    session_service = SqliteSessionService(db_path=str(tmp_path / "runner.db"))

    try:
        # Session 1: User states vessel & marina; after_agent_callback auto-ingests into memory_service
        preload_llm = RecordingFakeLlm()
        agent_preload = build_memory_aware_captain_agent(
            model=preload_llm,
            retrieval_mode="preload",
            auto_ingest_on_turn_end=True,
        )
        runner_preload = Runner(
            app_name=BENCHMARK_APP_NAME,
            agent=agent_preload,
            session_service=session_service,
            memory_service=memory_service,
        )
        await session_service.create_session(
            app_name=BENCHMARK_APP_NAME,
            user_id=PRIMARY_USER_ID,
            session_id="s-ingest",
        )
        async for _ in runner_preload.run_async(
            user_id=PRIMARY_USER_ID,
            session_id="s-ingest",
            new_message=types.Content(
                role="user",
                parts=[
                    types.Part(
                        text="Picked up a 25ft Contender bay boat, draws 16 inches, at Bud N' Mary's in Islamorada."
                    )
                ],
            ),
        ):
            pass

        # Session 2A: PreloadMemoryTool injects <PAST_CONVERSATIONS> before the single LLM call
        preload_llm.requests.clear()
        await session_service.create_session(
            app_name=BENCHMARK_APP_NAME,
            user_id=PRIMARY_USER_ID,
            session_id="s-query-preload",
        )
        async for _ in runner_preload.run_async(
            user_id=PRIMARY_USER_ID,
            session_id="s-query-preload",
            new_message=types.Content(
                role="user",
                parts=[types.Part(text="Can my boat clear a 12-inch flat tomorrow?")],
            ),
        ):
            pass

        assert len(preload_llm.requests) == 1
        serialized_prompt = str(preload_llm.requests[0].contents)
        assert "<PAST_CONVERSATIONS>" in serialized_prompt
        assert "16 inches" in serialized_prompt

        # Session 2B: LoadMemoryTool exposes FunctionDeclaration and takes 2 LLM calls when invoked
        ondemand_llm = RecordingFakeLlm(issue_load_memory_call=True)
        agent_ondemand = build_memory_aware_captain_agent(
            model=ondemand_llm,
            retrieval_mode="on_demand",
            auto_ingest_on_turn_end=False,
        )
        runner_ondemand = Runner(
            app_name=BENCHMARK_APP_NAME,
            agent=agent_ondemand,
            session_service=session_service,
            memory_service=memory_service,
        )
        await session_service.create_session(
            app_name=BENCHMARK_APP_NAME,
            user_id=PRIMARY_USER_ID,
            session_id="s-query-ondemand",
        )
        async for _ in runner_ondemand.run_async(
            user_id=PRIMARY_USER_ID,
            session_id="s-query-ondemand",
            new_message=types.Content(
                role="user",
                parts=[types.Part(text="Can my boat clear a 12-inch flat tomorrow?")],
            ),
        ):
            pass

        assert len(ondemand_llm.requests) == 2
    finally:
        await memory_service.close()


@pytest.mark.asyncio
async def test_comparative_8_session_benchmark_invariants() -> None:
    """Verify exact behavior of all four memory architectures on the 8-session coastal benchmark."""
    summaries, consolidated = await run_comparative_benchmark(
        window_sessions=2,
        vector_top_k=4,
    )
    try:
        by_name = {s.strategy_name: s for s in summaries}
        sliding = by_name["1. Sliding Session Window (last 2)"]
        keyword = by_name["2. ADK InMemoryMemoryService (Keyword)"]
        vec_rag = by_name["3. Append-Only Vector RAG (top-4)"]
        cons = by_name["4. Consolidated Fact Memory (Memory Bank)"]

        # 1. Sliding window evicts early sessions (misses boat upgrade + tarpon rig)
        assert sliding.recall_pass_count < cons.recall_pass_count
        assert sliding.cross_tenant_leak_count == 0

        # 2. Keyword & Append-Only Vector RAG suffer stale contradictions on mutated boat/crew facts
        assert keyword.stale_contradiction_count >= 1
        assert vec_rag.stale_contradiction_count >= 1

        # 3. Consolidated Memory passes 5/5 probes with zero stale contradictions, noise, or tenant leaks
        assert cons.exact_pass_count == 5
        assert cons.recall_pass_count == 5
        assert cons.stale_contradiction_count == 0
        assert cons.telemetry_noise_count == 0
        assert cons.cross_tenant_leak_count == 0

        # 4. Verify reconciliation audit counts: 4 initial slots added, 3 slot updates, 3 weather turns ignored
        assert consolidated.reconciliation_counts["ADD"] == 4
        assert consolidated.reconciliation_counts["UPDATE"] == 3
        assert consolidated.reconciliation_counts["IGNORE_EPHEMERAL"] == 3

        profile = await consolidated.retrieve_profile(
            app_name=BENCHMARK_APP_NAME,
            user_id=PRIMARY_USER_ID,
        )
        assert len(profile) == 4
        by_slot = {m.slot_key: m for m in profile}
        assert by_slot["vessel.hull_and_draft"].revision == 2
        assert "16 inches" in by_slot["vessel.hull_and_draft"].fact_text
        assert by_slot["marina.home_slip"].revision == 2
        assert "Bud N' Mary" in by_slot["marina.home_slip"].fact_text
        assert by_slot["crew.sarah_sea_state_limit"].revision == 2
        assert "3.0 ft" in by_slot["crew.sarah_sea_state_limit"].fact_text
    finally:
        await consolidated.close()
