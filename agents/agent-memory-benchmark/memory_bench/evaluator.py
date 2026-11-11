"""Longitudinal evaluation harness comparing four ADK memory strategies across eight sessions."""

from __future__ import annotations

import datetime as dt
import time
from collections.abc import Sequence

from google.adk.events.event import Event
from google.adk.memory import BaseMemoryService, InMemoryMemoryService
from google.adk.sessions import InMemorySessionService, Session
from google.genai import types

from memory_bench.dataset import (
    BENCHMARK_APP_NAME,
    COASTAL_EVALUATION_PROBES,
    COASTAL_LONGITUDINAL_SESSIONS,
    PRIMARY_USER_ID,
)
from memory_bench.embeddings import DeterministicCoastalEmbedder, EmbeddingAdapter
from memory_bench.extractor import FactExtractor, HeuristicCoastalFactExtractor
from memory_bench.schemas import (
    LongitudinalSession,
    MemoryProbe,
    ProbeOutcome,
    StrategyEvaluationSummary,
)
from memory_bench.services import (
    ConsolidatedMemoryService,
    SlidingWindowMemoryService,
    SqliteVectorMemoryService,
    extract_plain_text,
)


async def build_adk_session(
    session_service: InMemorySessionService,
    *,
    app_name: str,
    user_id: str,
    scenario: LongitudinalSession,
) -> Session:
    """Hydrate a realistic ADK Session with timestamped user/model Events (including thought parts)."""
    session = await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=scenario.session_id,
    )
    base_dt = dt.datetime.fromisoformat(f"{scenario.recorded_date}T12:00:00+00:00")

    for idx, turn in enumerate(scenario.turns):
        parts: list[types.Part] = []
        if turn.thought_text:
            parts.append(types.Part(text=turn.thought_text, thought=True))
        parts.append(types.Part(text=turn.text))
        event = Event(
            invocation_id=f"{scenario.session_id}-inv-{idx}",
            author="user" if turn.role == "user" else "KeysMemoryCaptain",
            content=types.Content(role=turn.role, parts=parts),
            timestamp=(base_dt + dt.timedelta(minutes=idx * 2)).timestamp(),
        )
        await session_service.append_event(session=session, event=event)

    refreshed = await session_service.get_session(
        app_name=app_name,
        user_id=user_id,
        session_id=scenario.session_id,
    )
    assert refreshed is not None
    return refreshed


async def evaluate_memory_strategy(
    *,
    strategy_name: str,
    memory_service: BaseMemoryService,
    sessions: Sequence[LongitudinalSession] = COASTAL_LONGITUDINAL_SESSIONS,
    probes: Sequence[MemoryProbe] = COASTAL_EVALUATION_PROBES,
    app_name: str = BENCHMARK_APP_NAME,
    primary_user_id: str = PRIMARY_USER_ID,
) -> StrategyEvaluationSummary:
    """Ingest all longitudinal sessions and score every evaluation probe."""
    session_service = InMemorySessionService()

    t0 = time.perf_counter()
    for scenario in sessions:
        adk_session = await build_adk_session(
            session_service,
            app_name=app_name,
            user_id=primary_user_id,
            scenario=scenario,
        )
        await memory_service.add_session_to_memory(adk_session)
    ingest_wall_ms = (time.perf_counter() - t0) * 1000.0

    outcomes: list[ProbeOutcome] = []
    t_search_start = time.perf_counter()
    for probe in probes:
        response = await memory_service.search_memory(
            app_name=app_name,
            user_id=probe.user_id,
            query=probe.query,
        )
        snippets = tuple(
            txt
            for m in response.memories
            if (txt := extract_plain_text(m.content))
        )
        joined = "\n".join(snippets)
        joined_lower = joined.lower()
        approx_tokens = max(0, round(len(joined) / 4))

        if probe.expect_empty:
            leaked = len(snippets) > 0
            outcomes.append(
                ProbeOutcome(
                    probe_id=probe.probe_id,
                    passed_recall=not leaked,
                    has_stale_contradiction=False,
                    has_telemetry_noise=False,
                    leaked_cross_tenant=leaked,
                    retrieved_count=len(snippets),
                    approx_prompt_tokens=approx_tokens,
                    retrieved_snippets=snippets,
                )
            )
            continue

        passed_recall = all(req.lower() in joined_lower for req in probe.required_substrings)
        has_stale = any(
            stale.lower() in joined_lower for stale in probe.forbidden_stale_substrings
        )
        has_noise = any(
            noise.lower() in joined_lower for noise in probe.forbidden_noise_substrings
        )
        outcomes.append(
            ProbeOutcome(
                probe_id=probe.probe_id,
                passed_recall=passed_recall,
                has_stale_contradiction=has_stale,
                has_telemetry_noise=has_noise,
                leaked_cross_tenant=False,
                retrieved_count=len(snippets),
                approx_prompt_tokens=approx_tokens,
                retrieved_snippets=snippets,
            )
        )
    search_wall_ms = (time.perf_counter() - t_search_start) * 1000.0

    exact_passes = sum(
        1
        for o in outcomes
        if o.passed_recall
        and not o.has_stale_contradiction
        and not o.has_telemetry_noise
        and not o.leaked_cross_tenant
    )
    recall_passes = sum(1 for o in outcomes if o.passed_recall)
    stale_count = sum(1 for o in outcomes if o.has_stale_contradiction)
    noise_count = sum(1 for o in outcomes if o.has_telemetry_noise)
    leak_count = sum(1 for o in outcomes if o.leaked_cross_tenant)
    non_empty_probes = [o for o in outcomes if not o.leaked_cross_tenant]
    mean_tokens = (
        sum(o.approx_prompt_tokens for o in non_empty_probes) / len(non_empty_probes)
        if non_empty_probes
        else 0.0
    )

    return StrategyEvaluationSummary(
        strategy_name=strategy_name,
        probes_total=len(outcomes),
        exact_pass_count=exact_passes,
        recall_pass_count=recall_passes,
        stale_contradiction_count=stale_count,
        telemetry_noise_count=noise_count,
        cross_tenant_leak_count=leak_count,
        mean_prompt_tokens=round(mean_tokens, 1),
        ingest_wall_ms=round(ingest_wall_ms, 2),
        search_wall_ms=round(search_wall_ms, 2),
        outcomes=tuple(outcomes),
    )


async def run_comparative_benchmark(
    *,
    embedder: EmbeddingAdapter | None = None,
    extractor: FactExtractor | None = None,
    window_sessions: int = 2,
    vector_top_k: int = 4,
) -> tuple[list[StrategyEvaluationSummary], ConsolidatedMemoryService]:
    """Run all four memory architectures against identical 8-session coastal history."""
    emb = embedder or DeterministicCoastalEmbedder()
    ext = extractor or HeuristicCoastalFactExtractor()

    sliding = SlidingWindowMemoryService(window_sessions=window_sessions)
    keyword = InMemoryMemoryService()
    vector_rag = SqliteVectorMemoryService(
        sqlite_path=":memory:",
        embedder=emb,
        top_k=vector_top_k,
        index_model_turns=True,
    )
    consolidated = ConsolidatedMemoryService(
        sqlite_path=":memory:",
        extractor=ext,
        embedder=emb,
        top_k=vector_top_k,
    )

    try:
        summaries = [
            await evaluate_memory_strategy(
                strategy_name=f"1. Sliding Session Window (last {window_sessions})",
                memory_service=sliding,
            ),
            await evaluate_memory_strategy(
                strategy_name="2. ADK InMemoryMemoryService (Keyword)",
                memory_service=keyword,
            ),
            await evaluate_memory_strategy(
                strategy_name=f"3. Append-Only Vector RAG (top-{vector_top_k})",
                memory_service=vector_rag,
            ),
            await evaluate_memory_strategy(
                strategy_name="4. Consolidated Fact Memory (Memory Bank)",
                memory_service=consolidated,
            ),
        ]
    finally:
        await vector_rag.close()

    return summaries, consolidated
