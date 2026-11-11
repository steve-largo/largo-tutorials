"""Curate, sanitize, and chronologically split SFT trajectories for Gemma 4 tool calling."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from gemma4_serving.config import validate_artifact_jsonl_path
from gemma4_serving.schemas import (
    DatasetCurationReport,
    SftConversationRecord,
    SftTurnMessage,
)

_THOUGHT_TAG_RE = re.compile(r"<thought>[\s\S]*?</thought>\s*", re.IGNORECASE)


class TideToolArgs(BaseModel):
    """Strict argument contract for `fetch_noaa_tides`."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    station_id: str = Field(pattern=r"^\d{7}$")
    hours_ahead: int = Field(ge=6, le=72)


class MarineWeatherToolArgs(BaseModel):
    """Strict argument contract for `fetch_marine_conditions`."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    latitude: float = Field(ge=24.0, le=31.0)
    longitude: float = Field(ge=-83.0, le=-79.5)
    zone_id: str = Field(pattern=r"^GMZ\d{3}$")


class MemorySlotArgs(BaseModel):
    """Strict argument contract for `commit_memory_slot`."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    slot_key: Literal[
        "vessel.hull_and_draft",
        "marina.home_slip",
        "crew.sarah_sea_state_limit",
        "tackle.tarpon_hook_rig",
        "ephemeral.weather",
    ]
    operation: Literal["ADD", "UPDATE", "NOOP", "IGNORE_EPHEMERAL"]
    canonical_fact: str = Field(min_length=5, max_length=280)


_TOOL_SCHEMAS: dict[str, type[BaseModel]] = {
    "fetch_noaa_tides": TideToolArgs,
    "fetch_marine_conditions": MarineWeatherToolArgs,
    "commit_memory_slot": MemorySlotArgs,
}

AGENT_TOOL_DECLARATIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": model_cls.__doc__ or name,
            "parameters": model_cls.model_json_schema(),
        },
    }
    for name, model_cls in _TOOL_SCHEMAS.items()
]


def build_raw_coastal_candidates() -> list[dict[str, Any]]:
    """Build 16 chronological raw trajectories (including 2 malformed rows and 5 thought traces)."""
    stations = [
        ("8723970", "Vaca Key / Marathon", 24.711, -81.106, "GMZ053"),
        ("8724580", "Key West Harbor", 24.551, -81.808, "GMZ054"),
        ("8723214", "Virginia Key / Biscayne", 25.731, -80.162, "GMZ052"),
        ("8722670", "Lake Worth Pier", 26.613, -80.034, "GMZ052"),
    ]
    raw: list[dict[str, Any]] = []
    idx = 0

    for cycle in range(3):
        st_id, label, lat, lon, zone = stations[cycle % len(stations)]
        hours = 24 if cycle % 2 == 0 else 12
        thought = (
            f"<thought>Map '{label}' to NOAA station {st_id} and NWS marine zone {zone}.</thought>\n"
            if cycle in (0, 2)
            else ""
        )
        raw.append(
            {
                "record_id": f"marine-dispatch-{idx:02d}",
                "task_family": "marine_tool_dispatch",
                "chronological_index": idx,
                "assistant_content": thought,
                "user_prompt": (
                    f"Check the next {hours}h tide table and reef swell near {label} before dawn."
                ),
                "tool_calls": [
                    {
                        "id": f"call_tide_{idx}",
                        "name": "fetch_noaa_tides",
                        "arguments": json.dumps({"station_id": st_id, "hours_ahead": hours}),
                    },
                    {
                        "id": f"call_wx_{idx}",
                        "name": "fetch_marine_conditions",
                        "arguments": json.dumps(
                            {"latitude": lat, "longitude": lon, "zone_id": zone}
                        ),
                    },
                ],
                "tool_outputs": [
                    f"High tide 06:42 EDT (+1.9 ft), Low tide 12:58 EDT (+0.2 ft) at {st_id}.",
                    f"Zone {zone}: E winds 11 kt, seas 2.1 ft at 7s period.",
                ],
                "final_reply": (
                    f"{label} shows an incoming morning push peaking at 06:42 EDT (+1.9 ft) "
                    f"with 2.1 ft seas at 7s in {zone}."
                ),
            }
        )
        idx += 1

    memory_cases = [
        (
            "Sold the 18ft flats skiff and moved up to a 25ft Contender bay boat drawing 16 inches.",
            "<thought>Durable equipment mutation on vessel.hull_and_draft -> UPDATE.</thought>",
            "vessel.hull_and_draft",
            "UPDATE",
            "Primary boat is a 25ft Contender bay boat that draws 16 inches.",
        ),
        (
            "NWS posted a Small Craft Advisory for 28 kt gusts through midnight tonight.",
            "<thought>Transient weather advisory -> IGNORE_EPHEMERAL.</thought>",
            "ephemeral.weather",
            "IGNORE_EPHEMERAL",
            "Transient 28 kt Small Craft Advisory expires at midnight.",
        ),
        (
            "Moved our permanent slip down to Bud N' Mary's Marina in Islamorada.",
            "",
            "marina.home_slip",
            "UPDATE",
            "Home slip is at Bud N' Mary's Marina in Islamorada.",
        ),
        (
            "Sarah is comfortable up to 3.0 ft seas now whenever swell period is 7 seconds or longer.",
            "<thought>Crew comfort threshold updated -> UPDATE.</thought>",
            "crew.sarah_sea_state_limit",
            "UPDATE",
            "Sarah handles up to 3.0 ft seas when swell period is 7s or longer.",
        ),
        (
            "Rigging Channel Five tarpon leaders with barbless 2/0 circle hooks on 60 lb fluorocarbon.",
            "",
            "tackle.tarpon_hook_rig",
            "ADD",
            "Tarpon rig uses barbless 2/0 circle hooks on 60 lb fluorocarbon leader.",
        ),
        (
            "Passing squall line dropped visibility to 1 mile at Snake Creek for the next 30 minutes.",
            "",
            "ephemeral.weather",
            "IGNORE_EPHEMERAL",
            "Short-lived squall line near Snake Creek.",
        ),
        (
            "Still running barbless 2/0 circle hooks on 60 lb fluorocarbon for bridge tarpon.",
            "",
            "tackle.tarpon_hook_rig",
            "NOOP",
            "Tarpon rig uses barbless 2/0 circle hooks on 60 lb fluorocarbon leader.",
        ),
    ]

    for user_text, thought_prefix, slot_key, op, fact in memory_cases:
        raw.append(
            {
                "record_id": f"mem-reconcile-{idx:02d}",
                "task_family": "memory_slot_reconcile",
                "chronological_index": idx,
                "assistant_content": thought_prefix,
                "user_prompt": user_text,
                "tool_calls": [
                    {
                        "id": f"call_mem_{idx}",
                        "name": "commit_memory_slot",
                        "arguments": json.dumps(
                            {
                                "slot_key": slot_key,
                                "operation": op,
                                "canonical_fact": fact,
                            }
                        ),
                    }
                ],
                "tool_outputs": [f"Recorded {op} on {slot_key}."],
                "final_reply": f"Memory reconciliation complete ({op} on {slot_key}).",
            }
        )
        idx += 1

    # Inject 2 intentionally malformed rows to verify schema boundary rejection during curation
    raw.append(
        {
            "record_id": f"malformed-station-{idx:02d}",
            "task_family": "marine_tool_dispatch",
            "chronological_index": idx,
            "assistant_content": "",
            "user_prompt": "Check tides at invalid station.",
            "tool_calls": [
                {
                    "id": "bad_1",
                    "name": "fetch_noaa_tides",
                    "arguments": json.dumps({"station_id": "INVALID", "hours_ahead": 200}),
                }
            ],
            "tool_outputs": ["error"],
            "final_reply": "error",
        }
    )
    idx += 1
    raw.append(
        {
            "record_id": f"malformed-json-{idx:02d}",
            "task_family": "memory_slot_reconcile",
            "chronological_index": idx,
            "assistant_content": "",
            "user_prompt": "Corrupted JSON arguments.",
            "tool_calls": [
                {
                    "id": "bad_2",
                    "name": "commit_memory_slot",
                    "arguments": "{unquoted_broken_json:",
                }
            ],
            "tool_outputs": ["error"],
            "final_reply": "error",
        }
    )
    return raw


def curate_and_split_sft_dataset(
    raw_rows: list[dict[str, Any]],
    train_ratio: float = 0.8,
    strip_thoughts: bool = True,
) -> tuple[list[SftConversationRecord], list[SftConversationRecord], DatasetCurationReport]:
    """Validate tool-call JSON schemas, strip `<thought>` traces, and split chronologically."""
    accepted: list[SftConversationRecord] = []
    rejected = 0
    stripped_count = 0

    for row in sorted(raw_rows, key=lambda r: int(r["chronological_index"])):
        raw_assistant = str(row.get("assistant_content", ""))
        if strip_thoughts and _THOUGHT_TAG_RE.search(raw_assistant):
            stripped_count += 1
            clean_assistant = _THOUGHT_TAG_RE.sub("", raw_assistant).strip()
        else:
            clean_assistant = raw_assistant.strip()

        validated_calls: list[dict[str, Any]] = []
        row_valid = True
        for tc in row.get("tool_calls", []):
            fn_name = tc.get("name", "")
            schema_cls = _TOOL_SCHEMAS.get(fn_name)
            if schema_cls is None:
                row_valid = False
                break
            try:
                parsed_args = json.loads(tc.get("arguments", "{}"))
                validated_model = schema_cls.model_validate(parsed_args)
            except (json.JSONDecodeError, ValidationError):
                row_valid = False
                break

            validated_calls.append(
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": fn_name,
                        "arguments": validated_model.model_dump_json(),
                    },
                }
            )

        if not row_valid or not validated_calls:
            rejected += 1
            continue

        system_prompt = (
            "You are a Florida Keys marine triage and memory-consolidation agent. "
            "Invoke typed tools with exact NOAA station IDs, NWS marine zones, or canonical "
            "memory slot operations."
        )
        messages: list[SftTurnMessage] = [
            SftTurnMessage(role="system", content=system_prompt),
            SftTurnMessage(role="user", content=str(row["user_prompt"])),
            SftTurnMessage(
                role="assistant",
                content=clean_assistant,
                tool_calls=validated_calls,
            ),
        ]
        for tc_obj, tool_out in zip(validated_calls, row["tool_outputs"], strict=True):
            messages.append(
                SftTurnMessage(
                    role="tool",
                    content=str(tool_out),
                    tool_call_id=tc_obj["id"],
                )
            )
        messages.append(SftTurnMessage(role="assistant", content=str(row["final_reply"])))

        accepted.append(
            SftConversationRecord(
                record_id=str(row["record_id"]),
                task_family=row["task_family"],
                chronological_index=int(row["chronological_index"]),
                messages=messages,
                tools=AGENT_TOOL_DECLARATIONS,
            )
        )

    split_idx = max(1, min(len(accepted) - 1, int(len(accepted) * train_ratio)))
    train_records = accepted[:split_idx]
    val_records = accepted[split_idx:]

    report = DatasetCurationReport(
        raw_candidates=len(raw_rows),
        accepted_records=len(accepted),
        rejected_malformed_args=rejected,
        stripped_thought_blocks=stripped_count,
        train_count=len(train_records),
        val_count=len(val_records),
    )
    return train_records, val_records, report


def export_sft_jsonl(
    records: list[SftConversationRecord],
    output_path: str,
    base_dir: Path | None = None,
) -> Path:
    """Write validated SFT records to JSONL after checking path traversal boundaries."""
    safe_path = validate_artifact_jsonl_path(output_path, base_dir=base_dir)
    safe_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [rec.model_dump_json() for rec in records]
    safe_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return safe_path
