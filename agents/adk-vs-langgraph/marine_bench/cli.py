"""CLI runner comparing Google ADK, LangGraph, and raw data fan-out telemetry."""

import argparse
import asyncio
import json

import httpx

from marine_bench.adk_agent import configure_adk_vertex_env, run_adk_workflow
from marine_bench.clients import MarineDataClient
from marine_bench.config import load_settings
from marine_bench.domain import format_snapshot_for_synthesis
from marine_bench.langgraph_agent import create_vertex_chat_model, run_langgraph_workflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Google ADK and LangGraph marine report orchestration."
    )
    parser.add_argument(
        "--mode",
        choices=("data-only", "adk", "langgraph", "both"),
        default="both",
        help="Execution path to run (`data-only` requires zero cloud credentials).",
    )
    parser.add_argument(
        "--location",
        default="Marathon, FL",
        help="Coastal US place name or 'lat, lon' coordinates.",
    )
    parser.add_argument(
        "--date",
        default="tomorrow",
        help="Target date (YYYY-MM-DD or natural phrase such as 'tomorrow' or 'Saturday').",
    )
    return parser.parse_args()


async def main_async() -> None:
    args = parse_args()
    settings = load_settings()

    async with httpx.AsyncClient(timeout=settings.http_timeout_seconds) as http_client:
        marine_client = MarineDataClient(http_client=http_client, settings=settings)

        if args.mode == "data-only":
            snapshot = await marine_client.collect_snapshot(args.location, args.date)
            print(format_snapshot_for_synthesis(snapshot))
            return

        configure_adk_vertex_env(settings)

        if args.mode in {"adk", "both"}:
            report, telemetry, authors = await run_adk_workflow(
                location_query=args.location,
                date_query=args.date,
                client=marine_client,
                model=settings.gemini_model,
            )
            print("=== Google ADK Workflow ===")
            print("Event trace authors:", " -> ".join(authors))
            print("Telemetry:", json.dumps(telemetry.model_dump(mode="json")))
            print("\n" + report + "\n")

        if args.mode in {"langgraph", "both"}:
            chat_model = create_vertex_chat_model(settings)
            report, telemetry = await run_langgraph_workflow(
                location_query=args.location,
                date_query=args.date,
                client=marine_client,
                chat_model=chat_model,
            )
            print("=== LangGraph StateGraph ===")
            print("Telemetry:", json.dumps(telemetry.model_dump(mode="json")))
            print("\n" + report + "\n")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
