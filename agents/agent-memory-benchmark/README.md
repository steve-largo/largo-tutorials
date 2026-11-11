# Stateful Agent Memory Benchmark: Rolling Sessions vs Vector RAG vs Vertex AI Memory Bank

Companion codebase for **Part 2** of the Vertex AI engineering series on [largo.dev](https://largo.dev/tutorials/agents/stateful-agent-memory-vertex-ai/).

This repository replays an 8-session longitudinal Florida Keys marine captain dataset (May through November) across four `google-adk` `2.14.x` memory architectures:

1. **Sliding Session Window (`SlidingWindowMemoryService` + `SqliteSessionService`)**: Retains only the last `k` sessions (`user:` state vs rolling event history).
2. **ADK Built-In Keyword Memory (`InMemoryMemoryService`)**: Word-overlap matching over raw session events.
3. **Append-Only Local Vector RAG (`SqliteVectorMemoryService`)**: SQLite + L2-normalized cosine similarity over conversation turns.
4. **Consolidated Fact & Profile Memory (`ConsolidatedMemoryService` / `VertexAiMemoryBankService`)**: Thought-stripped ingestion, ephemeral forecast filtering, slot-level fact reconciliation (`ADD` / `UPDATE` / `NOOP`), and deterministic scope-keyed profile lookup.

## Quickstart (Keyless Offline Benchmark & Tests)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Run the offline unit, security, and ADK Runner integration tests
.venv/bin/pytest -v

# Run the 8-session comparative evaluation scorecard and revision audit ledger
.venv/bin/python -m memory_bench.cli --mode offline-eval
```

## Live Run Against Vertex AI Gemini, Embeddings, and Memory Bank

Install the Vertex AI Agent Engine extra (`google-cloud-agentplatform[agent-engines]>=2.2,<3`) and authenticate via Application Default Credentials:

```bash
.venv/bin/pip install "google-cloud-agentplatform[agent-engines]>=2.2.0,<3.0.0"
gcloud auth application-default login

export GOOGLE_CLOUD_PROJECT="your-personal-gcp-project"
export GOOGLE_CLOUD_LOCATION="us-central1"
export GOOGLE_GENAI_USE_VERTEXAI="TRUE"

# Run the 8-session benchmark using live Gemini structured extraction + gemini-embedding-001
.venv/bin/python -m memory_bench.cli --mode vertex-live
```

When provisioning a managed `VertexAiMemoryBankService` instance via `build_production_services(settings)`, pass only the trailing ReasoningEngine ID segment (`AGENT_ENGINE_ID="456"`, extracted via `agent_engine.api_resource.name.split("/")[-1]`).
