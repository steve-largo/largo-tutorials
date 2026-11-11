"""Google ADK 2.x implementation of the Marine Report Agent.

Demonstrates two idiomatic ADK patterns on top of the shared MarineDataClient:
1. Graph-based `Workflow` (`google.adk.workflow.Workflow`, replacing the legacy
   `SequentialAgent` / `ParallelAgent` classes deprecated in ADK 2.x):
   - Compile-time validated `state_schema` (`MarineWorkflowState`)
   - Conditional route dictionary (`{"ok": (...), "error": ...}`) that short-circuits
     before any downstream API or LLM call when geocoding fails
   - Explicit `JoinNode` synchronizing the 4 concurrent collectors before
     `CaptainReportSynthesizer` (`LlmAgent`) runs
2. Conversational `LlmAgent` (`build_adk_tool_agent`) exposing a single async
   `collect_marine_telemetry` tool for interactive sessions in `adk web`.
"""

import os
import time
from datetime import date
from typing import Any

from google.adk.agents import Context, LlmAgent
from google.adk.models.base_llm import BaseLlm
from google.adk.runners import InMemoryRunner
from google.adk.tools import ToolContext
from google.adk.workflow import START, JoinNode, RetryConfig, Workflow, node
from google.genai import types
from pydantic import BaseModel, Field

from marine_bench.clients import LocationResolutionError, MarineDataClient
from marine_bench.config import MarineBenchSettings
from marine_bench.domain import (
    SYNTHESIS_SYSTEM_INSTRUCTION,
    calculate_lunar_phase,
    format_snapshot_for_synthesis,
    resolve_target_date,
)
from marine_bench.schemas import (
    LunarPhaseInfo,
    MarineBriefContext,
    ResolvedLocation,
    RunTelemetry,
    SeaState,
    TideWindow,
    WeatherWindow,
)


class MarineWorkflowState(BaseModel):
    """Typed session state contract validated at Workflow construction time."""

    location_query: str = ""
    date_query: str = "tomorrow"
    target_date: str = ""
    resolved_location: dict[str, Any] | None = None
    resolution_error: str | None = None
    tides: dict[str, Any] | None = None
    sea_state: dict[str, Any] | None = None
    weather: dict[str, Any] | None = None
    lunar: dict[str, Any] | None = None
    degraded_noaa_tides: str | None = None
    degraded_open_meteo_marine: str | None = None
    degraded_nws_weather: str | None = None
    degraded_sources: list[str] = Field(default_factory=list)
    formatted_snapshot: str = ""
    final_fishing_report: str = ""


def configure_adk_vertex_env(settings: MarineBenchSettings) -> None:
    """Propagate validated Vertex AI settings into the environment expected by google-genai."""
    if settings.use_vertexai:
        os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "TRUE")
    if settings.google_cloud_project:
        os.environ.setdefault("GOOGLE_CLOUD_PROJECT", settings.google_cloud_project)
    os.environ.setdefault("GOOGLE_CLOUD_LOCATION", settings.google_cloud_location)


def build_adk_workflow_pipeline(
    client: MarineDataClient,
    model: str | BaseLlm = "gemini-2.5-flash",
    reference_date: date | None = None,
) -> Workflow:
    """Build the ADK 2.x Workflow graph with conditional fan-out and JoinNode."""
    upstream_retry = RetryConfig(max_attempts=2, initial_delay=0.2, backoff_factor=2.0)

    @node(name="LocationIntake", timeout=15.0)
    async def resolve_location_step(
        ctx: Context,
        location_query: str,
        date_query: str = "tomorrow",
    ) -> types.Content | None:
        target = resolve_target_date(date_query, reference_date=reference_date)
        ctx.state["target_date"] = target.isoformat()

        try:
            resolved = await client.resolve_location(location_query)
        except LocationResolutionError as exc:
            ctx.state["resolution_error"] = str(exc)
            ctx.route = "error"
            return types.Content(
                role="model",
                parts=[types.Part(text=f"Location resolution failed: {exc}")],
            )

        ctx.state["resolved_location"] = resolved.model_dump(mode="json")
        ctx.state["resolution_error"] = None
        ctx.route = "ok"
        return None

    @node(name="TideCollector", retry_config=upstream_retry, timeout=12.0)
    async def fetch_tides_step(
        ctx: Context,
        resolved_location: ResolvedLocation,
        target_date: str,
    ) -> None:
        try:
            tides = await client.fetch_tides(
                resolved_location.station,
                date.fromisoformat(target_date),
            )
            ctx.state["tides"] = tides.model_dump(mode="json")
        except Exception as exc:
            ctx.state["tides"] = None
            ctx.state["degraded_noaa_tides"] = str(exc)

    @node(name="SeaStateCollector", retry_config=upstream_retry, timeout=12.0)
    async def fetch_sea_state_step(
        ctx: Context,
        resolved_location: ResolvedLocation,
        target_date: str,
    ) -> None:
        try:
            sea = await client.fetch_sea_state(
                resolved_location.latitude,
                resolved_location.longitude,
                date.fromisoformat(target_date),
            )
            ctx.state["sea_state"] = sea.model_dump(mode="json")
        except Exception as exc:
            ctx.state["sea_state"] = None
            ctx.state["degraded_open_meteo_marine"] = str(exc)

    @node(name="WeatherCollector", timeout=12.0)
    async def fetch_weather_step(
        ctx: Context,
        resolved_location: ResolvedLocation,
        target_date: str,
    ) -> None:
        try:
            weather = await client.fetch_weather(
                resolved_location.latitude,
                resolved_location.longitude,
                date.fromisoformat(target_date),
            )
            ctx.state["weather"] = weather.model_dump(mode="json")
        except Exception as exc:
            ctx.state["weather"] = None
            ctx.state["degraded_nws_weather"] = str(exc)

    @node(name="LunarCollector")
    async def compute_lunar_step(ctx: Context, target_date: str) -> None:
        lunar = calculate_lunar_phase(date.fromisoformat(target_date))
        ctx.state["lunar"] = lunar.model_dump(mode="json")

    @node(name="SnapshotAssembler")
    async def assemble_snapshot_step(
        ctx: Context,
        resolved_location: ResolvedLocation,
        target_date: str,
        lunar: LunarPhaseInfo,
        tides: TideWindow | None = None,
        sea_state: SeaState | None = None,
        weather: WeatherWindow | None = None,
        degraded_noaa_tides: str | None = None,
        degraded_open_meteo_marine: str | None = None,
        degraded_nws_weather: str | None = None,
    ) -> None:
        degraded: list[str] = []
        if degraded_noaa_tides:
            degraded.append("noaa_tides")
        if degraded_open_meteo_marine:
            degraded.append("open_meteo_marine")
        if degraded_nws_weather:
            degraded.append("nws_weather")

        snapshot = MarineBriefContext(
            location=resolved_location,
            target_date=target_date,
            tides=tides,
            sea_state=sea_state,
            weather=weather,
            lunar=lunar,
            degraded_sources=tuple(degraded),
        )
        ctx.state["degraded_sources"] = degraded
        ctx.state["formatted_snapshot"] = format_snapshot_for_synthesis(snapshot)

    join_collectors = JoinNode(name="JoinMarineCollectors")
    collectors = (
        fetch_tides_step,
        fetch_sea_state_step,
        fetch_weather_step,
        compute_lunar_step,
    )

    synthesizer = LlmAgent(
        name="CaptainReportSynthesizer",
        model=model,
        instruction=(
            SYNTHESIS_SYSTEM_INSTRUCTION
            + "\n\nTelemetry Snapshot:\n{formatted_snapshot}"
        ),
        output_key="final_fishing_report",
    )

    return Workflow(
        name="MarineReportWorkflow",
        state_schema=MarineWorkflowState,
        edges=[
            (
                START,
                resolve_location_step,
                {
                    "ok": collectors,
                },
            ),
            (
                collectors,
                join_collectors,
                assemble_snapshot_step,
                synthesizer,
            ),
        ],
    )


def build_adk_tool_agent(
    client: MarineDataClient,
    model: str | BaseLlm = "gemini-2.5-flash",
    reference_date: date | None = None,
) -> LlmAgent:
    """Build a conversational LlmAgent backed by a concurrent snapshot tool."""

    async def collect_marine_telemetry(
        location: str,
        target_date: str = "tomorrow",
        tool_context: ToolContext | None = None,
    ) -> dict[str, Any]:
        """Resolve a coastal US location and fetch NOAA tides, Open-Meteo waves, NWS wind, and moon phase concurrently."""
        snapshot = await client.collect_snapshot(
            location_query=location,
            date_query=target_date,
            reference_date=reference_date,
        )
        formatted = format_snapshot_for_synthesis(snapshot)
        if tool_context is not None:
            tool_context.state["last_snapshot"] = snapshot.model_dump(mode="json")
        return {
            "formatted_snapshot": formatted,
            "degraded_sources": list(snapshot.degraded_sources),
        }

    return LlmAgent(
        name="ConversationalMarineAgent",
        model=model,
        instruction=(
            "Always call collect_marine_telemetry first with the requested location "
            "and target_date before writing the report.\n\n"
            + SYNTHESIS_SYSTEM_INSTRUCTION
        ),
        tools=[collect_marine_telemetry],
        output_key="final_fishing_report",
    )


async def run_adk_workflow(
    *,
    location_query: str,
    date_query: str = "tomorrow",
    client: MarineDataClient,
    model: str | BaseLlm = "gemini-2.5-flash",
    reference_date: date | None = None,
) -> tuple[str, RunTelemetry, list[str]]:
    """Execute the ADK 2.x Workflow graph and return (report_text, telemetry, event_authors)."""
    pipeline = build_adk_workflow_pipeline(
        client=client,
        model=model,
        reference_date=reference_date,
    )
    runner = InMemoryRunner(agent=pipeline, app_name="largo_marine_bench")
    session = await runner.session_service.create_session(
        app_name="largo_marine_bench",
        user_id="angler_01",
        state={"location_query": location_query, "date_query": date_query},
    )

    t0 = time.perf_counter()
    authors: list[str] = []
    chunks: list[str] = []
    llm_calls = 0

    user_msg = types.Content(
        role="user",
        parts=[
            types.Part(
                text=f"Generate fishing conditions report for {location_query} ({date_query})."
            )
        ],
    )

    async for event in runner.run_async(
        user_id=session.user_id,
        session_id=session.id,
        new_message=user_msg,
    ):
        step_label = (
            event.node_info.path.replace("/", ".").split(".")[-1].split("@")[0]
            if event.node_info and event.node_info.path
            else event.author
        )
        authors.append(step_label)
        if event.author == "CaptainReportSynthesizer" and event.content:
            llm_calls += 1
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    chunks.append(part.text)

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
    updated_session = await runner.session_service.get_session(
        app_name="largo_marine_bench",
        user_id=session.user_id,
        session_id=session.id,
    )
    has_error = bool(updated_session and updated_session.state.get("resolution_error"))
    degraded = tuple(
        (updated_session.state.get("degraded_sources") if updated_session else None)
        or ()
    )

    telemetry = RunTelemetry(
        framework="google-adk-workflow",
        llm_calls=llm_calls,
        tool_invocations=1 if has_error else 5,
        wall_time_ms=elapsed_ms,
        degraded_sources=degraded,
    )
    return "".join(chunks).strip(), telemetry, authors
