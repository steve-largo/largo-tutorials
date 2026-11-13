# Two-Stage Hybrid Retrieval Benchmark: BM25 + Dense RRF + Cross-Encoder Reranking

Companion codebase for **Part 4** of the *Embeddings & Reranking Foundations* series on [largo.dev](https://largo.dev/tutorials/retrieval-systems/hybrid-search-reranking/).

This project benchmarks four progressive retrieval pipelines against a self-contained 48-chunk cloud and ML infrastructure corpus and 28 graded evaluation queries split evenly across two diagnostic slices:

1. **Exact Identifiers (`exact_id`, 14 queries)**: TLS error codes (`ERR_SSL_PROTOCOL_ERROR`, `X509_V_ERR_UNABLE_TO_GET_ISSUER_CERT_LOCALLY`), AWS GPU instance SKUs (`g5.12xlarge` vs `g5.2xlarge`), Envoy response flags (`UO` vs `UF`), CLI flags (`--max-num-batched-tokens`), Kubernetes exit codes (`139` vs `137`), PostgreSQL `SQLSTATE` codes (`55P03`, `25P03`), and Linux kernel sysctls (`net.ipv4.tcp_tw_reuse`, `TcpExtListenOverflows`).
2. **Semantic Paraphrases (`semantic_paraphrase`, 14 queries)**: Operational symptom descriptions that share minimal vocabulary with canonical runbook terminology (for example, describing synchronized retry storms against runbooks titled *Decorrelated Backoff Jitter and Half-Open Canary Probing*).

## Measured Benchmark Scorecard (8-vCPU AMD EPYC 7B12, CPU)

Models: `BAAI/bge-small-en-v1.5` (384-dim dense bi-encoder) and `cross-encoder/ms-marco-MiniLM-L-6-v2` (stage-2 reranker), `rrf_k=60`, `stage1_top_k=25`, `rerank_top_k=5`.

| Pipeline | Overall Recall@5 | Exact nDCG@10 | Semantic nDCG@10 | Overall nDCG@10 | CPU p50 (ms) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 1. BM25 Only | 0.661 | 0.896 | 0.637 | 0.767 | 0.16 |
| 2. Dense Bi-Encoder Only | 0.786 | 0.891 | 0.884 | 0.888 | 20.12 |
| 3. Hybrid BM25 + Dense RRF (`k=60`) | 0.732 | 0.929 | 0.778 | 0.854 | 23.16 |
| 4. Two-Stage Hybrid RRF + Cross-Encoder | **0.821** | **0.931** | **0.881** | **0.906** | 542.14 |

## Project Layout

- `schemas.py`: Frozen Pydantic v2 data contracts (`CorpusChunk`, `EvalQuery`, `QuerySlice`, `ScoredCandidate`, `StageMetrics`, `BenchmarkReport`).
- `config.py`: Immutable `RetrievalSettings` (`BaseSettings`) loaded from `HYBRID_RETRIEVAL_*` environment variables with boundary validation.
- `corpus.py`: 48 infrastructure runbook chunks across 8 domains plus 28 graded evaluation queries (`grade=2` primary target, `grade=1` companion runbook).
- `retrievers.py`: Identifier-preserving `BM25Okapi` retriever, L2-normalized `SentenceTransformer` dense retriever, scale-free Reciprocal Rank Fusion (`1 / (k + rank)`), and `CrossEncoder` top-K reranker.
- `benchmark.py`: Reproducible 4-stage evaluator outputting `Recall@5`, `Recall@25`, `MRR@10`, `nDCG@10`, and `p50`/`p95` latency in milliseconds.
- `tests/test_hybrid_retrieval.py`: Unit, RRF scale-invariance, metric math, protocol fake, and end-to-end invariant tests.

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Run linter and unit + end-to-end invariant tests
.venv/bin/ruff check .
.venv/bin/pytest -v

# Run the 4-pipeline benchmark scorecard (add --json for full structured output)
.venv/bin/python benchmark.py
```

## Runtime Configuration

Override defaults via `HYBRID_RETRIEVAL_*` environment variables:

```bash
export HYBRID_RETRIEVAL_BI_ENCODER_MODEL="BAAI/bge-small-en-v1.5"
export HYBRID_RETRIEVAL_CROSS_ENCODER_MODEL="cross-encoder/ms-marco-MiniLM-L-6-v2"
export HYBRID_RETRIEVAL_RRF_K=60
export HYBRID_RETRIEVAL_STAGE1_TOP_K=25
export HYBRID_RETRIEVAL_RERANK_TOP_K=5
export HYBRID_RETRIEVAL_DEVICE="cpu"
```
