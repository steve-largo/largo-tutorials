# Cross-Encoders: Precision Reranking for Search

Companion codebase for **Part 3** of the *Embeddings & Reranking Foundations* series on [largo.dev](https://largo.dev/tutorials/embeddings/cross-encoders-reranking/).

This package implements a two-stage semantic retrieval pipeline pairing fast dense candidate generation (`BAAI/bge-small-en-v1.5`) with joint query-passage cross-encoder reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`) over a 120-passage technical corpus and 12 disambiguation queries spanning:

- **Polysemy**: Disambiguating shared nouns (`python` neural network training vs tropical `python` snake ecology).
- **Negation**: Respecting explicit exclusions (`without Docker containers`, `without mutex locks`, `without a networked server`).
- **Numeric & Hardware Constraints**: Enforcing resource ceilings (`under a single 24 GB GPU`, `512-token truncation`, `CPU without GPU`).
- **Technical Specificity**: Distinguishing adjacent operational codes (`SQLSTATE 55P03` vs `40P01`, Kubernetes exit `137` vs `139`, dual-CA mTLS rotation).

## Measured Benchmark Scorecard (AMD EPYC 7B12 CPU)

Models: `BAAI/bge-small-en-v1.5` (stage-1 dense retrieval) and `cross-encoder/ms-marco-MiniLM-L-6-v2` (stage-2 cross-encoder), `corpus=120` passages, `rerank_batch_size=32`.

| Pipeline | MRR@10 | Precision@1 | Pool Recall | Stage-1 p50 (ms) | Rerank p50 (ms) | Total p50 (ms) | Pairs/s |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage-1 Bi-Encoder Only (`top-10`) | 0.958 | 0.917 | 1.000 | 21.27 | 0.00 | 21.27 | 0.0 |
| Two-Stage Cross-Encoder (`top-10`) | **1.000** | **1.000** | 1.000 | 21.25 | 155.65 | 182.28 | 66.7 |
| Two-Stage Cross-Encoder (`top-50`) | **1.000** | **1.000** | 1.000 | 22.04 | 666.27 | 688.30 | 69.5 |
| Two-Stage Cross-Encoder (`top-100`) | **1.000** | **1.000** | 1.000 | 20.11 | 1286.66 | 1306.29 | 74.0 |

Cross-encoder reranking resolves the bi-encoder rank-1 miss (`Precision@1: 0.917 -> 1.000`, `MRR@10: 0.958 -> 1.000`) while exhibiting linear `O(N)` reranking latency across candidate pool depths (`155.65 ms` at `top-10`, `666.27 ms` at `top-50`, and `1286.66 ms` at `top-100` on CPU).

## Project Layout

- `schemas.py`: Frozen Pydantic v2 data contracts (`PassageRecord`, `RerankEvalQuery`, `RankedPassage`, `SearchDiagnostics`, `PoolScaleResult`, `CrossEncoderBenchmarkReport`).
- `config.py`: Immutable `CrossEncoderSettings` (`BaseSettings`) loaded from `CROSS_ENCODER_*` environment variables.
- `reranker.py`: Injectable `BiEncoderBackend` and `CrossEncoderScorer` protocols, `SentenceTransformerBiEncoder` and `MsMarcoCrossEncoder` adapters, overflow-safe `sigmoid_probability` + score blending, and `TwoStageReranker`.
- `benchmark.py`: 120-passage corpus, 12 disambiguation queries, `MRR@10` / `Precision@1` / `Recall@Pool` evaluators, and candidate-pool scaling runner (`top-10`, `top-50`, `top-100`).
- `tests/test_cross_encoder.py`: Offline-safe deterministic unit tests verifying negation trap promotion, sigmoid score fusion stability, and exact candidate-pool pair counts.

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Run linter and fast offline unit tests
.venv/bin/ruff check .
.venv/bin/pytest -v

# Run the live MS MARCO reranking and pool-scaling benchmark (add --json for JSON output)
.venv/bin/python benchmark.py
```

## Runtime Configuration

Override defaults via `CROSS_ENCODER_*` environment variables:

```bash
export CROSS_ENCODER_BI_ENCODER_MODEL="BAAI/bge-small-en-v1.5"
export CROSS_ENCODER_CROSS_ENCODER_MODEL="cross-encoder/ms-marco-MiniLM-L-6-v2"
export CROSS_ENCODER_RERANK_BATCH_SIZE=32
export CROSS_ENCODER_FUSION_ALPHA=0.0
export CROSS_ENCODER_DEVICE="cpu"
```

## Author

**Steven W. White** ([largo.dev](https://largo.dev))
