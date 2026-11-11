"""Hermetic unit and orchestration tests for ADK and LangGraph marine agents."""

from collections.abc import AsyncGenerator
from datetime import date

import httpx
import pytest
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import ValidationError

from marine_bench.adk_agent import run_adk_workflow
from marine_bench.clients import MarineDataClient
from marine_bench.config import MarineBenchSettings
from marine_bench.domain import (
    calculate_lunar_phase,
    degrees_to_cardinal,
    parse_wind_speed_mph,
    resolve_target_date,
)
from marine_bench.langgraph_agent import build_langgraph_marine_app, run_langgraph_workflow
from marine_bench.schemas import NoaaStation


class FakeAdkLlm(BaseLlm):
    """Deterministic in-memory BaseLlm double for testing ADK runners offline."""

    model: str = "fake-gemini-2.5-flash"
    prompts_seen: list[str] = []

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        rendered_parts: list[str] = []
        if llm_request.config and llm_request.config.system_instruction:
            rendered_parts.append(str(llm_request.config.system_instruction))
        for content in llm_request.contents or []:
            for part in content.parts or []:
                if part.text:
                    rendered_parts.append(part.text)
        joined = "\n".join(rendered_parts)
        self.prompts_seen.append(joined)
        yield LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part(text=f"ADK Captain Verdict based on: {joined[:120]}")],
            ),
            turn_complete=True,
        )


class FakeLangChainChatModel(BaseChatModel):
    """Deterministic in-memory BaseChatModel double for testing LangGraph offline."""

    prompts_seen: list[str] = []

    @property
    def _llm_type(self) -> str:
        return "fake-vertex-gemini"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager=None,
        **kwargs,
    ) -> ChatResult:
        joined = "\n".join(str(m.content) for m in messages)
        self.prompts_seen.append(joined)
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content=f"LangGraph Captain Verdict based on: {joined[:120]}"
                    )
                )
            ]
        )


def make_mock_transport(*, fail_nws: bool = False, geocode_calls: list[int] | None = None):
    """Build a hermetic httpx.MockTransport simulating Nominatim, NOAA, Open-Meteo, and NWS."""

    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        path = request.url.path

        if host == "nominatim.openstreetmap.org":
            if geocode_calls is not None:
                geocode_calls.append(1)
            q = request.url.params.get("q", "")
            if "Atlantis" in q:
                return httpx.Response(200, json=[])
            return httpx.Response(
                200,
                json=[
                    {
                        "lat": "24.7136",
                        "lon": "-81.0903",
                        "display_name": "Marathon, Monroe County, Florida, United States",
                    }
                ],
            )

        if host == "api.tidesandcurrents.noaa.gov" and path.endswith("/stations.json"):
            return httpx.Response(
                200,
                json={
                    "stations": [
                        {
                            "id": "8723970",
                            "name": "VACA KEY, FLORIDA BAY",
                            "lat": 24.7117,
                            "lng": -81.1067,
                        },
                        {
                            "id": "8724580",
                            "name": "KEY WEST",
                            "lat": 24.5557,
                            "lng": -81.8079,
                        },
                    ]
                },
            )

        if host == "api.tidesandcurrents.noaa.gov" and path.endswith("/datagetter"):
            return httpx.Response(
                200,
                json={
                    "predictions": [
                        {"t": "2026-11-12 01:46", "v": "1.36", "type": "H"},
                        {"t": "2026-11-12 08:11", "v": "0.11", "type": "L"},
                    ]
                },
            )

        if host == "marine-api.open-meteo.com":
            return httpx.Response(
                200,
                json={
                    "daily": {
                        "wave_height_max": [1.2],
                        "swell_wave_height_max": [1.0],
                        "wave_period_max": [4.5],
                        "wave_direction_dominant": [135.0],
                    },
                    "hourly": {"sea_surface_temperature": [82.4, 82.8]},
                },
            )

        if host == "api.weather.gov":
            if fail_nws:
                return httpx.Response(503, json={"title": "Service Unavailable"})
            if path.startswith("/points/"):
                return httpx.Response(
                    200,
                    json={
                        "properties": {
                            "forecast": "https://api.weather.gov/gridpoints/KEY/65,48/forecast"
                        }
                    },
                )
            return httpx.Response(
                200,
                json={
                    "properties": {
                        "periods": [
                            {
                                "name": "Thursday",
                                "startTime": "2026-11-12T06:00:00-05:00",
                                "isDaytime": True,
                                "temperature": 82,
                                "windSpeed": "10 to 15 mph",
                                "windDirection": "ESE",
                                "shortForecast": "Mostly Sunny",
                                "probabilityOfPrecipitation": {"value": 10},
                            }
                        ]
                    }
                },
            )

        return httpx.Response(404, json={"error": "unhandled host"})

    return httpx.MockTransport(handler)


def test_pure_domain_helpers() -> None:
    ref = date(2026, 11, 11)  # Wednesday
    assert resolve_target_date("tomorrow", reference_date=ref) == date(2026, 11, 12)
    assert resolve_target_date("Saturday", reference_date=ref) == date(2026, 11, 14)
    assert parse_wind_speed_mph("Calm") == 0
    assert parse_wind_speed_mph("10 to 15 mph") == 15
    assert degrees_to_cardinal(135.0) == "SE"

    lunar = calculate_lunar_phase(date(2026, 11, 12))
    assert 0 <= lunar.illumination_pct <= 100
    assert lunar.target_date == "2026-11-12"


def test_security_url_and_schema_guards() -> None:
    with pytest.raises(ValidationError):
        MarineBenchSettings(noaa_stations_url="http://api.tidesandcurrents.noaa.gov/stations.json")

    with pytest.raises(ValidationError):
        MarineBenchSettings(nws_base_url="https://evil.example.com")

    with pytest.raises(ValidationError):
        NoaaStation(
            station_id="8723970/../admin",
            station_name="Bad Station",
            latitude=24.7,
            longitude=-81.1,
            distance_miles=1.0,
        )


async def test_adk_workflow_happy_path_and_degradation() -> None:
    settings = MarineBenchSettings()
    fake_llm = FakeAdkLlm(prompts_seen=[])

    async with httpx.AsyncClient(transport=make_mock_transport(fail_nws=True)) as http:
        client = MarineDataClient(http_client=http, settings=settings)
        report, telemetry, authors = await run_adk_workflow(
            location_query="Marathon, FL",
            date_query="tomorrow",
            client=client,
            model=fake_llm,
            reference_date=date(2026, 11, 11),
        )

    assert "ADK Captain Verdict" in report
    assert telemetry.llm_calls == 1
    assert telemetry.degraded_sources == ("nws_weather",)
    assert "LocationIntake" in authors
    assert "CaptainReportSynthesizer" in authors
    assert "Degraded Sources: nws_weather" in fake_llm.prompts_seen[0]


async def test_adk_workflow_escalates_on_unknown_location_without_llm_call() -> None:
    settings = MarineBenchSettings()
    fake_llm = FakeAdkLlm(prompts_seen=[])

    async with httpx.AsyncClient(transport=make_mock_transport()) as http:
        client = MarineDataClient(http_client=http, settings=settings)
        report, telemetry, authors = await run_adk_workflow(
            location_query="Atlantis",
            date_query="tomorrow",
            client=client,
            model=fake_llm,
            reference_date=date(2026, 11, 11),
        )

    assert "Location resolution failed" in report
    assert telemetry.llm_calls == 0
    assert fake_llm.prompts_seen == []
    assert "CaptainReportSynthesizer" not in authors


async def test_langgraph_parallel_degradation_and_checkpoint_reuse() -> None:
    settings = MarineBenchSettings()
    geocode_calls: list[int] = []
    fake_chat = FakeLangChainChatModel(prompts_seen=[])
    saver = InMemorySaver()

    async with httpx.AsyncClient(
        transport=make_mock_transport(fail_nws=True, geocode_calls=geocode_calls)
    ) as http:
        client = MarineDataClient(http_client=http, settings=settings)
        app = build_langgraph_marine_app(
            client=client,
            chat_model=fake_chat,
            checkpointer=saver,
            reference_date=date(2026, 11, 11),
        )

        # Turn 1: geocodes Marathon, FL and records degraded NWS weather
        state_1 = await app.ainvoke(
            {"location_query": "Marathon, FL", "date_query": "tomorrow"},
            config={"configurable": {"thread_id": "keys_skiff_01"}},
        )
        assert len(geocode_calls) == 1
        assert "nws_weather" in state_1["degraded_sources"]
        assert "LangGraph Captain Verdict" in state_1["final_report"]

        # Turn 2 on same thread_id: shifts date to Saturday and reuses checkpointed station
        state_2 = await app.ainvoke(
            {"location_query": "Marathon, FL", "date_query": "Saturday"},
            config={"configurable": {"thread_id": "keys_skiff_01"}},
        )
        assert len(geocode_calls) == 1  # Zero redundant Nominatim calls!
        assert state_2["target_date"] == "2026-11-14"


async def test_langgraph_short_circuits_on_unknown_location() -> None:
    settings = MarineBenchSettings()
    fake_chat = FakeLangChainChatModel(prompts_seen=[])

    async with httpx.AsyncClient(transport=make_mock_transport()) as http:
        client = MarineDataClient(http_client=http, settings=settings)
        report, telemetry = await run_langgraph_workflow(
            location_query="Atlantis",
            date_query="tomorrow",
            client=client,
            chat_model=fake_chat,
            reference_date=date(2026, 11, 11),
        )

    assert "Location resolution failed" in report
    assert telemetry.llm_calls == 0
    assert fake_chat.prompts_seen == []
