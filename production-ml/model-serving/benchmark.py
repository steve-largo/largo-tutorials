"""PyTorch serving benchmark comparing eager, inference_mode, autocast, and torch.compile."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable

import numpy as np
import torch

from config import ServingSettings, load_settings
from model import (
    HashedSubwordTokenizer,
    build_seeded_classifier,
    execute_forward,
    resolve_autocast_dtype,
)
from schemas import ExecutionRuntime, RuntimeLatencyRow, ServingBenchmarkReport

_SAMPLE_PROMPTS: tuple[str, ...] = (
    "This inference endpoint responds with fast tail latency and clean predictions.",
    "Upstream requests stalled behind a slow lock timeout and crashed the worker.",
    "Canary rollout metrics stayed accurate and reliable across peak batch load.",
    "Terrible cold start delays caused repeated client errors during failover.",
)


def _precision_label(runtime: ExecutionRuntime, device: torch.device) -> str:
    if runtime == "autocast_low_precision":
        amp_dtype = resolve_autocast_dtype(device)
        return "fp16" if amp_dtype == torch.float16 else "bf16"
    return "fp32"


def _build_batch_texts(batch_size: int) -> list[str]:
    return [_SAMPLE_PROMPTS[idx % len(_SAMPLE_PROMPTS)] for idx in range(batch_size)]


def run_benchmark(settings: ServingSettings | None = None) -> ServingBenchmarkReport:
    """Measure p50/p95 latency, QPS, items/sec, and logit drift across PyTorch runtimes."""
    cfg = settings or load_settings()
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device(
        "cuda" if cfg.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    tokenizer = HashedSubwordTokenizer(
        vocab_size=cfg.vocab_size,
        max_seq_length=cfg.max_seq_length,
    )
    base_model = build_seeded_classifier(cfg, tokenizer).to(device)
    param_count = sum(p.numel() for p in base_model.parameters())

    runtimes: list[ExecutionRuntime] = [
        "eager_fp32",
        "inference_mode_fp32",
        "autocast_low_precision",
    ]
    if cfg.include_torch_compile:
        runtimes.append("torch_compile")

    compiled_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None
    if cfg.include_torch_compile:
        compiled_fn = torch.compile(base_model, mode=cfg.compile_mode)  # type: ignore[assignment]

    rows: list[RuntimeLatencyRow] = []

    for batch_size in cfg.benchmark_batch_sizes:
        texts = _build_batch_texts(batch_size)
        input_ids, attention_mask = tokenizer.encode_batch(texts, device=device)

        with torch.no_grad():
            reference_logits = base_model(input_ids, attention_mask).float()

        for runtime in runtimes:
            fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = (
                compiled_fn if runtime == "torch_compile" and compiled_fn is not None else base_model
            )

            warmup_iters = max(cfg.warmup_runs, 6) if runtime == "torch_compile" else cfg.warmup_runs
            for _ in range(warmup_iters):
                execute_forward(fn, input_ids, attention_mask, runtime=runtime, device=device)

            sample_logits = execute_forward(
                fn, input_ids, attention_mask, runtime=runtime, device=device
            )
            max_diff = float((sample_logits - reference_logits).abs().max().item())

            latencies_ms: list[float] = []
            for _ in range(cfg.benchmark_runs):
                t0 = time.perf_counter()
                execute_forward(fn, input_ids, attention_mask, runtime=runtime, device=device)
                latencies_ms.append((time.perf_counter() - t0) * 1000.0)

            mean_ms = max(float(np.mean(latencies_ms)), 1e-6)
            p50_ms = float(np.percentile(latencies_ms, 50))
            p95_ms = float(np.percentile(latencies_ms, 95))
            rps = 1000.0 / mean_ms
            ips = rps * batch_size

            rows.append(
                RuntimeLatencyRow(
                    runtime=runtime,
                    precision_dtype=_precision_label(runtime, device),
                    batch_size=batch_size,
                    mean_ms=round(mean_ms, 3),
                    p50_ms=round(p50_ms, 3),
                    p95_ms=round(p95_ms, 3),
                    requests_per_second=round(rps, 1),
                    items_per_second=round(ips, 1),
                    max_logit_abs_diff=round(max_diff, 5),
                )
            )

    return ServingBenchmarkReport(
        device=str(device),
        parameter_count=param_count,
        seq_length=cfg.max_seq_length,
        warmup_runs=cfg.warmup_runs,
        benchmark_runs=cfg.benchmark_runs,
        rows=tuple(rows),
    )


def format_report_table(report: ServingBenchmarkReport) -> str:
    """Render an ASCII latency, throughput, and numerical-drift summary table."""
    lines: list[str] = []
    lines.append(
        f"=== PyTorch Model Serving Benchmark "
        f"(params={report.parameter_count:,}, seq_len={report.seq_length}, "
        f"runs={report.benchmark_runs}, device={report.device}) ==="
    )
    lines.append("")
    header = (
        f"{'Runtime':<24} | {'DType':<5} | {'Batch':>5} | {'p50 ms':>7} | "
        f"{'p95 ms':>7} | {'Req/s':>8} | {'Items/s':>9} | {'Max |dLogit|':>12}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for row in report.rows:
        lines.append(
            f"{row.runtime:<24} | {row.precision_dtype:<5} | {row.batch_size:>5d} | "
            f"{row.p50_ms:>7.2f} | {row.p95_ms:>7.2f} | {row.requests_per_second:>8.1f} | "
            f"{row.items_per_second:>9.1f} | {row.max_logit_abs_diff:>12.5f}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark PyTorch eager, inference_mode, BF16/FP16 autocast, and torch.compile."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the validated ServingBenchmarkReport as formatted JSON.",
    )
    args = parser.parse_args()
    report = run_benchmark()
    print(format_report_table(report))
    if args.json:
        print("\n" + json.dumps(report.model_dump(), indent=2))


if __name__ == "__main__":
    main()
