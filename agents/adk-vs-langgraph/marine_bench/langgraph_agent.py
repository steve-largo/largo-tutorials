"""LangGraph StateGraph implementation of the Marine Report Agent.

Models the exact same dependency DAG as an explicit cyclic/parallel state machine:
  START -> resolve_location --(error)--> format_resolution_error -> END
                            --(ok)-----> [fetch_tides, fetch_sea_state, fetch_weather, compute_lunar]
                                                      |
                                              synthesize_report -> END
"""

import operator
import time
from datetime import date
from typing import Annotated, Any, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

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


class MarineGraphState(TypedDict, total=False):
    """Typed state channel schema with an additive reducer for degraded_sources."""

    location_query: str
    date_query: str
    target_date: str
    resolved_location: dict[str, Any] | None
    resolution_error: str | None
    tides: dict[str, Any] | None
    sea_state: dict[str, Any] | None
    weather: dict[str, Any] | None
    lunar: dict[str, Any] | None
    degraded_sources: Annotated[list[str], operator.add]
    formatted_snapshot: str
    final_report: str
    llm_calls: int


def create_vertex_chat_model(settings: MarineBenchSettings) -> ChatGoogleGenerativeAI:
    """Instantiate Gemini on Vertex AI via langchain-google-genai."""
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        vertexai=settings.use_vertexai,
        project=settings.google_cloud_project,
        location=settings.google_cloud_location,
        temperature=0.2,
    )


def build_langgraph_marine_app(
    *,
    client: MarineDataClient,
    chat_model: BaseChatModel,
    checkpointer: BaseCheckpointSaver | None = None,
    reference_date: date | None = None,
):
    """Compile the explicit fan-out/fan-in StateGraph with optional checkpointing."""

    async def resolve_node(state: MarineGraphState) -> dict[str, Any]:
        loc_query = state.get("location_query", "").strip()
        target = resolve_target_date(
            state.get("date_query"),
            reference_date=reference_date,
        )

        # Reuse checkpointed station across follow-up turns when location_query is unchanged
        existing_raw = state.get("resolved_location")
        if existing_raw and existing_raw.get("query") == loc_query:
            return {
                "target_date": target.isoformat(),
                "resolution_error": None,
            }

        try:
            resolved = await client.resolve_location(loc_query)
            return {
                "resolved_location": resolved.model_dump(mode="json"),
                "target_date": target.isoformat(),
                "resolution_error": None,
            }
        except LocationResolutionError as exc:
            return {
                "resolved_location": None,
                "target_date": target.isoformat(),
                "resolution_error": str(exc),
            }

    def route_after_resolve(state: MarineGraphState) -> list[str] | str:
        if state.get("resolution_error") or not state.get("resolved_location"):
            return "handle_resolution_error"
        return [
            "fetch_tides",
            "fetch_sea_state",
            "fetch_weather",
            "compute_lunar",
        ]

    async def handle_resolution_error_node(state: MarineGraphState) -> dict[str, Any]:
        err = state.get("resolution_error", "Unknown location")
        return {
            "final_report": f"Location resolution failed: {err}",
            "llm_calls": 0,
        }

    async def fetch_tides_node(state: MarineGraphState) -> dict[str, Any]:
        loc = ResolvedLocation.model_validate(state["resolved_location"])
        target = date.fromisoformat(state["target_date"])
        try:
            tides = await client.fetch_tides(loc.station, target)
            return {"tides": tides.model_dump(mode="json")}
        except Exception:
            return {"tides": None, "degraded_sources": ["noaa_tides"]}

    async def fetch_sea_state_node(state: MarineGraphState) -> dict[str, Any]:
        loc = ResolvedLocation.model_validate(state["resolved_location"])
        target = date.fromisoformat(state["target_date"])
        try:
            sea = await client.fetch_sea_state(loc.latitude, loc.longitude, target)
            return {"sea_state": sea.model_dump(mode="json")}
        except Exception:
            return {"sea_state": None, "degraded_sources": ["open_meteo_marine"]}

    async def fetch_weather_node(state: MarineGraphState) -> dict[str, Any]:
        loc = ResolvedLocation.model_validate(state["resolved_location"])
        target = date.fromisoformat(state["target_date"])
        try:
            weather = await client.fetch_weather(loc.latitude, loc.longitude, target)
            return {"weather": weather.model_dump(mode="json")}
        except Exception:
            return {"weather": None, "degraded_sources": ["nws_weather"]}

    async def compute_lunar_node(state: MarineGraphState) -> dict[str, Any]:
        target = date.fromisoformat(state["target_date"])
        lunar = calculate_lunar_phase(target)
        return {"lunar": lunar.model_dump(mode="json")}

    async def synthesize_node(state: MarineGraphState) -> dict[str, Any]:
        degraded_unique = tuple(sorted(set(state.get("degraded_sources") or [])))
        snapshot = MarineBriefContext(
            location=ResolvedLocation.model_validate(state["resolved_location"]),
            target_date=state["target_date"],
            tides=(
                TideWindow.model_validate(state["tides"])
                if state.get("tides")
                else None
            ),
            sea_state=(
                SeaState.model_validate(state["sea_state"])
                if state.get("sea_state")
                else None
            ),
            weather=(
                WeatherWindow.model_validate(state["weather"])
                if state.get("weather")
                else None
            ),
            lunar=LunarPhaseInfo.model_validate(state["lunar"]),
            degraded_sources=degraded_unique,
        )
        formatted = format_snapshot_for_synthesis(snapshot)
        ai_msg = await chat_model.ainvoke(
            [
                SystemMessage(content=SYNTHESIS_SYSTEM_INSTRUCTION),
                HumanMessage(content=f"Telemetry Snapshot:\n{formatted}"),
            ]
        )
        content_str = (
            ai_msg.content
            if isinstance(ai_msg.content, str)
            else str(ai_msg.content)
        )
        return {
            "formatted_snapshot": formatted,
            "final_report": content_str.strip(),
            "llm_calls": int(state.get("llm_calls", 0)) + 1,
        }

    builder = StateGraph(MarineGraphState)
    builder.add_node("resolve_location", resolve_node)
    builder.add_node("handle_resolution_error", handle_resolution_error_node)
    builder.add_node("fetch_tides", fetch_tides_node)
    builder.add_node("fetch_sea_state", fetch_sea_state_node)
    builder.add_node("fetch_weather", fetch_weather_node)
    builder.add_node("compute_lunar", compute_lunar_node)
    builder.add_node("synthesize_report", synthesize_node)

    builder.add_edge(START, "resolve_location")
    builder.add_conditional_edges(
        "resolve_location",
        route_after_resolve,
        [
            "handle_resolution_error",
            "fetch_tides",
            "fetch_sea_state",
            "fetch_weather",
            "compute_lunar",
        ],
    )
    builder.add_edge("handle_resolution_error", END)

    for branch in ("fetch_tides", "fetch_sea_state", "fetch_weather", "compute_lunar"):
        builder.add_edge(branch, "synthesize_report")
    builder.add_edge("synthesize_report", END)

    return builder.compile(checkpointer=checkpointer or InMemorySaver())


async def run_langgraph_workflow(
    *,
    location_query: str,
    date_query: str = "tomorrow",
    client: MarineDataClient,
    chat_model: BaseChatModel,
    thread_id: str = "angler_thread_01",
    checkpointer: BaseCheckpointSaver | None = None,
    reference_date: date | None = None,
) -> tuple[str, RunTelemetry]:
    """Execute the compiled LangGraph app and return (report_text, telemetry)."""
    app = build_langgraph_marine_app(
        client=client,
        chat_model=chat_model,
        checkpointer=checkpointer,
        reference_date=reference_date,
    )

    t0 = time.perf_counter()
    final_state = await app.ainvoke(
        {
            "location_query": location_query,
            "date_query": date_query,
            "degraded_sources": [],
            "llm_calls": 0,
        },
        config={"configurable": {"thread_id": thread_id}},
    )
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)

    has_error = bool(final_state.get("resolution_error"))
    degraded = tuple(sorted(set(final_state.get("degraded_sources") or [])))
    telemetry = RunTelemetry(
        framework="langgraph-stategraph",
        llm_calls=int(final_state.get("llm_calls", 0)),
        tool_invocations=1 if has_error else 5,
        wall_time_ms=elapsed_ms,
        degraded_sources=degraded,
    )
    return str(final_state.get("final_report", "")), telemetry
