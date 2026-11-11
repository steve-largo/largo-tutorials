"""Exposes `root_agent` for `adk run` and `adk web --host 127.0.0.1 .`."""

import httpx

from marine_bench.adk_agent import build_adk_tool_agent, configure_adk_vertex_env
from marine_bench.clients import MarineDataClient
from marine_bench.config import load_settings

_settings = load_settings()
configure_adk_vertex_env(_settings)

# Shared async HTTP client scoped to the ADK server process
_http_client = httpx.AsyncClient(timeout=_settings.http_timeout_seconds)
_marine_client = MarineDataClient(http_client=_http_client, settings=_settings)

root_agent = build_adk_tool_agent(
    client=_marine_client,
    model=_settings.gemini_model,
)
