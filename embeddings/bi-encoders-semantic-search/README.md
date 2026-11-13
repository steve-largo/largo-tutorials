# Bi-Encoders: Fast Semantic Search at Scale

Companion codebase for **Part 2** of the *Embeddings & Reranking Foundations* series on [largo.dev](https://largo.dev/tutorials/embeddings/bi-encoders-semantic-search/).

This package implements a unit-L2-normalized dense vector index over a 24-document multi-domain corpus and evaluates semantic retrieval across 16 benchmark queries split evenly between two retrieval regimes:

1. **Symmetric Paraphrase Retrieval (`symmetric`, 8 queries)**: Full-sentence paraphrases matching passage titles and summaries of comparable length.
2. **Asymmetric Short-Query Retrieval (`asymmetric`, 8 queries)**: Terse 3-to-6 word search queries matched against multi-sentence technical passages.

It also benchmarks sequential single-query search (`search_exact`) against vectorized batched matrix search (`search_batched`) and persists indexes to `.npz` + `.json` artifacts without `pickle`.

## Measured Benchmark Scorecard (AMD EPYC 7B12 CPU)

Model: `BAAI/bge-small-en-v1.5` (384-dim dense bi-encoder), `eval_top_k=5`, `encode_batch_size=32`, `timing_repeats=5`.

### Retrieval Quality (`Recall@5` and `MRR@5`)

| Query Slice | Queries | Recall@5 | MRR@5 |
| :--- | ---: | ---: | ---: |
| Overall | 16 | 1.000 | 1.000 |
| Symmetric Paraphrase | 8 | 1.000 | 1.000 |
| Asymmetric Short Query | 8 | 1.000 | 1.000 |

### Single-Query Loop vs Batched Matrix Throughput

| Execution Mode | Batch Total (ms) | p50 / Query (ms) | Throughput (QPS) |
| :--- | ---: | ---: | ---: |
| `single_query_loop` | 572.76 | 30.53 | 27.9 |
| `batched_matrix` | **283.98** | **14.35** | **56.3** |

## Project Layout

- `schemas.py`: Frozen Pydantic v2 data contracts (`DocumentRecord`, `BenchmarkQuery`, `SearchHit`, `RetrievalMetrics`, `SearchModeTiming`, `BiEncoderBenchmarkReport`).
- `config.py`: Immutable `BiEncoderSettings` (`BaseSettings`) loaded from `BI_ENCODER_*` environment variables with model ID and device validation.
- `index.py`: `TextEncoder` protocol, `SentenceTransformerEncoder` adapter, `l2_normalize_rows`, `cosine_similarity_matrix`, and `DenseVectorIndex` (`search_exact`, `search_batched`, and pickle-free `.npz`/`.json` persistence).
- `benchmark.py`: 24-document corpus, 16 symmetric/asymmetric queries, `Recall@k` and `MRR@k` evaluators, and CLI benchmark runner.
- `tests/test_bi_encoder.py`: Fast offline unit tests covering schema validation, L2 scale invariance, zero-vector guards, exact vs batched equivalence, `.npz` persistence roundtrips, and metric math.

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Run linter and fast deterministic unit tests
.venv/bin/ruff check .
.venv/bin/pytest -v

# Run the live bi-encoder quality and batch-throughput benchmark (add --json for JSON output)
.venv/bin/python benchmark.py
```

## Runtime Configuration

Override defaults via `BI_ENCODER_*` environment variables (for example, switching between `BAAI/bge-small-en-v1.5` and `sentence-transformers/all-MiniLM-L6-v2`):

```bash
export BI_ENCODER_MODEL_NAME="BAAI/bge-small-en-v1.5"
export BI_ENCODER_ENCODE_BATCH_SIZE=32
export BI_ENCODER_EVAL_TOP_K=5
export BI_ENCODER_DEVICE="cpu"
```

## Author

**Steven W. White** ([largo.dev](https://largo.dev))
