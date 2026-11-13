# Largo Tutorials

Companion code for applied AI engineering tutorials on [largo.dev](https://largo.dev).

Each directory is a standalone, runnable Python package with frozen Pydantic v2 contracts, environment-driven configuration, deterministic offline `pytest` suites, and reproducible hardware benchmark CLIs.

## Companion Repositories (10 Packages)

### AI Agents & Orchestration (`agents/`)
- **[Agent Orchestration in Practice: Google ADK vs LangGraph](agents/adk-vs-langgraph/)** ([Tutorial](https://largo.dev/tutorials/agents/adk-vs-langgraph-vs-strands/)): Head-to-head implementation of a live coastal marine report workflow in Google ADK 2.x (`Workflow`, `@node`, `JoinNode`, `LlmAgent`) and LangGraph (`StateGraph`, `Annotated` reducers, `InMemorySaver`) over a shared NOAA, Open-Meteo, and NWS domain layer.
- **[Stateful Agent Memory: Rolling Sessions vs Vector RAG vs Vertex AI Memory Bank](agents/agent-memory-benchmark/)** ([Tutorial](https://largo.dev/tutorials/agents/stateful-agent-memory-vertex-ai/)): Eight-session longitudinal benchmark comparing sliding-window context, ADK `InMemoryMemoryService`, SQLite vector RAG, and consolidated slot-reconciling memory (`BaseMemoryService`, `VertexAiMemoryBankService`, and `SqliteSessionService`).
- **[Open-Weight Agent Serving: LoRA-Tuning Gemma 4 26B-A4B MoE on Vertex AI with vLLM](agents/gemma4-vertex-vllm/)** ([Tutorial](https://largo.dev/tutorials/agents/gemma-4-lora-vllm-vertex-ai/)): MoE VRAM and 5:1 hybrid sliding-window KV-cache sizing engine, chronological SFT trajectory curation, PEFT language-backbone regex scoping (`Gemma4ClippableLinear` exclusion), and a `VllmOpenAiLlm(BaseLlm)` adapter for Google ADK 2.14+.

### Embeddings & Reranking (`embeddings/`)
- **[Sentence Embeddings from Scratch](embeddings/sentence-embeddings-from-scratch/)** ([Tutorial](https://largo.dev/tutorials/embeddings/sentence-embeddings-from-scratch/)): PyTorch BiLSTM sentence encoder built from first principles, covering custom vocabulary construction, masked mean/max pooling, and cosine similarity geometry.
- **[Bi-Encoders: Fast Semantic Search at Scale](embeddings/bi-encoders-semantic-search/)** ([Tutorial](https://largo.dev/tutorials/embeddings/bi-encoders-semantic-search/)): Unit-L2-normalized dense vector index (`BAAI/bge-small-en-v1.5`) evaluating symmetric paraphrase and asymmetric short-query retrieval (`Recall@5 = 1.000`, `MRR@5 = 1.000`), single-query vs batched BLAS matrix throughput (`56.3 QPS` vs `27.9 QPS` on CPU), and pickle-free `.npz`/`.json` persistence.
- **[Cross-Encoders: Precision Reranking for Search](embeddings/cross-encoders-reranking/)** ([Tutorial](https://largo.dev/tutorials/embeddings/cross-encoders-reranking/)): Two-stage dense + `cross-encoder/ms-marco-MiniLM-L-6-v2` reranking pipeline across a 120-passage polysemy, negation, hardware-constraint, and error-code corpus (`Precision@1: 0.917 -> 1.000`, `MRR@10: 0.958 -> 1.000`) with candidate-pool latency scaling (`top-10`, `top-50`, `top-100`).

### Retrieval Systems (`retrieval-systems/`)
- **[Two-Stage Hybrid Retrieval: BM25, Dense Embeddings, RRF & Reranking](retrieval-systems/hybrid-search-reranking/)** ([Tutorial](https://largo.dev/tutorials/retrieval-systems/hybrid-search-reranking/)): Four-stage benchmark comparing Okapi BM25, dense `bge-small-en-v1.5` embeddings, scale-free Reciprocal Rank Fusion (`k=60`), and MS MARCO cross-encoder reranking across 48 infrastructure runbook chunks and 28 exact-identifier vs semantic-paraphrase queries (`Recall@5: 0.661 -> 0.821`, `nDCG@10: 0.767 -> 0.906`).
- **[CFP Oracle: Semantic Search for College Football History](retrieval-systems/cfp-oracle/)** ([Tutorial](https://largo.dev/tutorials/retrieval-systems/cfp-oracle/)): Narrative semantic search and metadata filtering over 36 historical College Football Playoff (2015-2025) and BCS Championship (2006-2014) matchups using Amazon Bedrock Titan Text Embeddings V2 (`1,024` dimensions) and Amazon S3 Vectors, with an offline cosine-ranking and client-contract test suite.

### Production ML (`production-ml/`)
- **[Production Model Serving: FastAPI & PyTorch Runtime Optimization](production-ml/model-serving/)** ([Tutorial](https://largo.dev/tutorials/production-ml/model-serving/)): Typed FastAPI inference service (`/health`, `/ready` `503` readiness gating, batched `/predict` bounds, request telemetry) paired with a 4.26M-parameter Transformer benchmark across `eager_fp32`, `inference_mode_fp32`, `autocast_low_precision`, and `torch.compile` (`0.00000` logit drift).

### Predictive Maintenance (`predictive-maintenance/`)
- **[Hard Drive Failure Prediction](predictive-maintenance/hard-drive-failure/)** ([Tutorial](https://largo.dev/tutorials/production-ml/hard-drive-failure-prediction/)): Predict drive failures using Backblaze SMART telemetry (24.8M records, Q4 2023), comparing XGBoost, Bidirectional LSTM, Conv1D + Transformer, and Mamba (SSM) architectures under extreme class imbalance (`0.01%` positive rate).

  | Model | Test AUC | F2 | Recall | Notes |
  | :--- | ---: | ---: | ---: | :--- |
  | **XGBoost** | **0.920** | **0.095** | 54% | Rolling SMART feature engineering |
  | Transformer | 0.916 | 0.006 | 80% | Conv1D preprocessing |
  | LSTM | 0.907 | 0.006 | 78% | Bidirectional sequence encoder |
  | Mamba (SSM) | 0.901 | 0.017 | 69% | Linear `O(n)` selective state space |

## Quickstart

Clone the repository and create an isolated virtual environment inside the tutorial package you want to run:

```bash
git clone https://github.com/steve-largo/largo-tutorials.git
cd largo-tutorials
```

### Run Offline Tests & Benchmarks (Example: Two-Stage Hybrid Retrieval)

```bash
cd retrieval-systems/hybrid-search-reranking
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/ruff check .
.venv/bin/pytest -v
.venv/bin/python benchmark.py
```

### Run FastAPI Model Serving & Runtime Benchmark

```bash
cd production-ml/model-serving
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/ruff check .
.venv/bin/pytest -v
.venv/bin/python benchmark.py
.venv/bin/uvicorn app:app --host 127.0.0.1 --port 8000
```

Every package includes its own `README.md`, `requirements.txt`, and environment variable reference (`HYBRID_RETRIEVAL_*`, `BI_ENCODER_*`, `CROSS_ENCODER_*`, `MODEL_SERVING_*`, `CFP_ORACLE_*`) for overriding model IDs, batch sizes, and hardware targets without editing code.

## Interactive Simulators

Explore the analytical models behind these benchmarks in the [largo.dev AI Systems Engineering Lab](https://largo.dev/lab/):
- **GPU VRAM & KV-Cache**: [largo.dev/lab/](https://largo.dev/lab/)
- **Two-Stage Retrieval & Reranking**: [largo.dev/lab/?tool=retrieval](https://largo.dev/lab/?tool=retrieval)
- **Stateful Agent Context & Memory**: [largo.dev/lab/?tool=memory](https://largo.dev/lab/?tool=memory)

## Author

**Steven W. White** · [largo.dev](https://largo.dev) · [LinkedIn](https://www.linkedin.com/in/stvwhite)

## License

MIT
