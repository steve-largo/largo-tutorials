# Agent Orchestration in Practice: Google ADK vs LangGraph

Companion repository for Part 1 of the three-part Vertex AI & Agent Architecture series at [largo.dev](https://largo.dev).

Implements the same live coastal marine and fishing report workflow in **Google Agent Development Kit (`google-adk`)** and **LangGraph (`langgraph`)** on top of a shared, framework-agnostic NOAA / Open-Meteo / NWS domain layer, anchored against the original [AWS Strands implementation](https://github.com/steve-largo/fishing-agent).

## Architecture

```
marine_bench/
├── config.py           # Immutable BaseSettings with HTTPS host allowlist
├── schemas.py          # Frozen Pydantic v2 contracts (NOAA, SeaState, NWS, Telemetry)
├── domain.py           # Pure date, wind, haversine, lunar, and synthesis formatting
├── clients.py          # Injectable async MarineDataClient (httpx)
├── adk_agent.py        # Google ADK SequentialAgent -> ParallelAgent -> LlmAgent + tool agent
├── langgraph_agent.py  # LangGraph StateGraph with parallel supersteps + InMemorySaver
└── cli.py              # CLI comparing data-only, ADK, and LangGraph execution
marine_agent_app/
└── agent.py            # Exposes `root_agent` for `adk web --host 127.0.0.1 .`
tests/
└── test_marine_bench.py # Hermetic offline test suite (MockTransport + fake LLMs)
```

## Quick Start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 1. Run hermetic unit & orchestration tests (zero cloud credentials needed)
.venv/bin/pytest -v

# 2. Fetch live NOAA + Open-Meteo + NWS telemetry (free public APIs, zero keys needed)
.venv/bin/python -m marine_bench.cli --mode data-only --location "Marathon, FL" --date tomorrow

# 3. Run both ADK and LangGraph against Gemini on Vertex AI
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT="your-personal-gcp-project"
export GOOGLE_CLOUD_LOCATION="us-central1"
export GOOGLE_GENAI_USE_VERTEXAI="TRUE"

.venv/bin/python -m marine_bench.cli --mode both --location "Marathon, FL" --date tomorrow

# 4. Inspect ADK event & state_delta traces locally (bound strictly to loopback)
.venv/bin/adk web --host 127.0.0.1 .
```

## License

MIT
