"""Command-line runner for the 8-session stateful agent memory benchmark."""

from __future__ import annotations

import argparse
import asyncio

from memory_bench.config import MemoryBenchSettings
from memory_bench.dataset import BENCHMARK_APP_NAME, PRIMARY_USER_ID
from memory_bench.embeddings import DeterministicCoastalEmbedder, VertexGenAiEmbedder
from memory_bench.evaluator import run_comparative_benchmark
from memory_bench.extractor import GeminiStructuredFactExtractor, HeuristicCoastalFactExtractor


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare rolling sessions, keyword memory, SQLite vector RAG, and consolidated Memory Bank."
    )
    parser.add_argument(
        "--mode",
        choices=("offline-eval", "vertex-live"),
        default="offline-eval",
        help="Use deterministic keyless adapters ('offline-eval') or Vertex AI Gemini + Embeddings ('vertex-live').",
    )
    parser.add_argument(
        "--window-sessions",
        type=int,
        default=2,
        help="Number of recent sessions retained by the sliding-window baseline.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Maximum retrieved entries for vector RAG and consolidated memory.",
    )
    return parser


async def _run_async(args: argparse.Namespace) -> int:
    settings = MemoryBenchSettings(
        sliding_window_sessions=args.window_sessions,
        vector_top_k=args.top_k,
    )

    if args.mode == "vertex-live":
        if not settings.google_cloud_project:
            raise SystemExit(
                "GOOGLE_CLOUD_PROJECT must be set when running --mode vertex-live."
            )
        embedder = VertexGenAiEmbedder(settings)
        extractor = GeminiStructuredFactExtractor(settings)
    else:
        embedder = DeterministicCoastalEmbedder()
        extractor = HeuristicCoastalFactExtractor()

    summaries, consolidated = await run_comparative_benchmark(
        embedder=embedder,
        extractor=extractor,
        window_sessions=settings.sliding_window_sessions,
        vector_top_k=settings.vector_top_k,
    )

    try:
        print("\n=== 8-Session Coastal Captain Memory Benchmark (May -> November) ===")
        header = (
            f"{'Strategy':<43} | {'Exact':<5} | {'Recall':<6} | "
            f"{'Stale':<5} | {'Noise':<5} | {'Leaks':<5} | {'Avg Tok':<7} | {'Search ms':<9}"
        )
        print(header)
        print("-" * len(header))
        for s in summaries:
            print(
                f"{s.strategy_name:<43} | "
                f"{s.exact_pass_count}/{s.probes_total:<3} | "
                f"{s.recall_pass_count}/{s.probes_total:<4} | "
                f"{s.stale_contradiction_count:<5} | "
                f"{s.telemetry_noise_count:<5} | "
                f"{s.cross_tenant_leak_count:<5} | "
                f"{s.mean_prompt_tokens:<7.1f} | "
                f"{s.search_wall_ms:<9.2f}"
            )

        print("\n=== Reconciliation Operations (Consolidated Memory) ===")
        print(consolidated.reconciliation_counts)

        print("\n=== Revision Audit Ledger (captain_steve) ===")
        history = await consolidated.list_revision_history(
            app_name=BENCHMARK_APP_NAME,
            user_id=PRIMARY_USER_ID,
        )
        for rec in history:
            state_tag = "ACTIVE" if rec.active else f"SUPERSEDED -> {rec.superseded_by}"
            print(
                f"  [{rec.slot_key} v{rec.revision}] ({state_tag}): {rec.fact_text}"
            )
    finally:
        await consolidated.close()

    return 0


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_run_async(args)))


if __name__ == "__main__":
    main()
