# Production Model Serving: FastAPI & PyTorch Runtime Optimization

Companion codebase for **Model Serving on AWS** (*Production ML Systems*, Part 4) on [largo.dev](https://largo.dev/tutorials/production-ml/model-serving/).

This package provides a typed FastAPI inference service (`/health`, `/ready`, and `/predict`) backed by an injectable `InferencePredictor` protocol alongside a reproducible PyTorch latency, throughput, and numerical-parity benchmark comparing four execution runtimes across batch sizes `1`, `8`, and `32`:

1. **`eager_fp32`**: Baseline `torch.no_grad()` eager execution in FP32.
2. **`inference_mode_fp32`**: `torch.inference_mode()` disabling view tracking and version counters.
3. **`autocast_low_precision`**: `torch.autocast` (`FP16` on CUDA Tensor Cores, `BF16` on CPU).
4. **`torch_compile`**: TorchDynamo + TorchInductor graph compilation (`mode="reduce-overhead"`).

## Measured CPU Benchmark Scorecard (AMD EPYC 7B12)

Architecture: 4-layer pre-norm Transformer sequence classifier (`4,257,666` parameters, `seq_len=64`, `benchmark_runs=20`, `device=cpu`).

| Runtime | DType | Batch | p50 (ms) | p95 (ms) | Req/s | Items/s | Max \|dLogit\| |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `eager_fp32` | fp32 | 1 | 5.72 | 8.02 | 167.2 | 167.2 | 0.00000 |
| `inference_mode_fp32` | fp32 | 1 | 5.78 | 6.88 | 168.3 | 168.3 | 0.00000 |
| `autocast_low_precision` | bf16 | 1 | 23.22 | 49.96 | 35.9 | 35.9 | 0.03480 |
| `torch_compile` | fp32 | 1 | **5.00** | **5.67** | **196.8** | **196.8** | 0.00000 |
| `eager_fp32` | fp32 | 8 | 32.33 | 38.00 | 30.4 | 243.0 | 0.00000 |
| `inference_mode_fp32` | fp32 | 8 | 36.96 | 51.52 | 28.4 | 227.0 | 0.00000 |
| `autocast_low_precision` | bf16 | 8 | 192.03 | 216.75 | 5.4 | 42.8 | 0.04527 |
| `torch_compile` | fp32 | 8 | **30.87** | **34.74** | **35.6** | **284.5** | 0.00000 |
| `eager_fp32` | fp32 | 32 | 94.17 | 107.44 | 10.3 | 331.1 | 0.00000 |
| `inference_mode_fp32` | fp32 | 32 | 97.28 | 113.13 | 10.0 | 320.0 | 0.00000 |
| `autocast_low_precision` | bf16 | 32 | 743.66 | 816.52 | 1.4 | 44.1 | 0.04527 |
| `torch_compile` | fp32 | 32 | **89.82** | **97.24** | **11.2** | **358.7** | 0.00000 |

Note on hardware precision selection: `torch_compile` improves single-item p95 latency by 29% (`8.02 ms -> 5.67 ms`) and throughput across all batch sizes on CPU with zero numerical drift. By contrast, `bf16` autocast slows down on Zen 2 (`AMD EPYC 7B12`) CPUs that lack native hardware BF16 matrix instructions due to software cast overhead, whereas GPU Tensor Cores (`NVIDIA L4`, `L40S`, `H100`) accelerate `FP16`/`BF16` kernels. Always benchmark target hardware before enabling autocast in production.

## Project Layout

- `schemas.py`: Frozen Pydantic v2 contracts (`PredictRequest` supporting single `text` or batched `texts`, `PredictResponse`, `HealthResponse`, `ReadyResponse`, `ServingBenchmarkReport`).
- `config.py`: Immutable `ServingSettings` (`BaseSettings`) loaded from `MODEL_SERVING_*` environment variables.
- `model.py`: `InferencePredictor` protocol, deterministic `HashedSubwordTokenizer`, `TransformerSentimentClassifier`, and `TorchSentimentPredictor`.
- `app.py`: FastAPI application factory (`create_app`) with lifespan warmup, `/health` liveness telemetry, `/ready` `503` readiness gating, and `/predict` batch bounds enforcement.
- `benchmark.py`: Multi-runtime latency, throughput, and logit-drift benchmark CLI.
- `tests/test_serving.py`: `TestClient` API contract tests (readiness gating, single/batch validation, telemetry counters) and PyTorch logit parity tests.

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Run linter and TestClient + PyTorch invariant suite
.venv/bin/ruff check .
.venv/bin/pytest -v

# Run the multi-runtime latency and throughput benchmark (add --json for JSON output)
.venv/bin/python benchmark.py

# Start the local FastAPI inference server on http://127.0.0.1:8000
.venv/bin/uvicorn app:app --host 127.0.0.1 --port 8000
```

## Runtime Configuration

Override defaults via `MODEL_SERVING_*` environment variables:

```bash
export MODEL_SERVING_DEVICE="cpu"
export MODEL_SERVING_RUNTIME="inference_mode_fp32"
export MODEL_SERVING_COMPILE_MODE="reduce-overhead"
export MODEL_SERVING_MAX_BATCH_SIZE=32
```

## Author

**Steven W. White** ([largo.dev](https://largo.dev))
