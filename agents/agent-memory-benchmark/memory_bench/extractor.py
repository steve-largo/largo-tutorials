"""Fact extraction layer separating durable user facts from ephemeral forecast telemetry."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol

from google import genai
from google.genai import types

from memory_bench.config import MemoryBenchSettings
from memory_bench.schemas import CandidateFact, FactExtractionBatch, SessionTurn

EXTRACTION_SYSTEM_INSTRUCTION = """You extract durable user profile facts and constraints from completed sessions.
Rules:
1. Extract ONLY facts stated or confirmed by the user (vessel specifications, home marina/launch ramp, crew safety limits, tackle preferences).
2. Classify transient weather reports, daily tide heights, wind advisories, or one-off trip conditions as is_ephemeral=true under category='ephemeral_telemetry'.
3. Assign a stable, canonical slot_key (for example: 'vessel.hull_and_draft', 'marina.home_slip', 'crew.sarah_sea_state_limit', 'tackle.tarpon_hook_rig') so updated facts overwrite superseded values cleanly.
"""


class FactExtractor(Protocol):
    """Protocol for extracting structured candidate facts from session turns."""

    async def extract_facts(self, turns: Sequence[SessionTurn]) -> list[CandidateFact]: ...


class HeuristicCoastalFactExtractor:
    """Deterministic extractor used for reproducible offline benchmarks and unit tests."""

    _EPHEMERAL_PATTERNS = (
        re.compile(r"\bsmall craft advisory\b", re.IGNORECASE),
        re.compile(r"\bhigh tide was\b", re.IGNORECASE),
        re.compile(r"\bstayed at the dock\b", re.IGNORECASE),
        re.compile(r"\bnws forecast\b", re.IGNORECASE),
    )

    async def extract_facts(self, turns: Sequence[SessionTurn]) -> list[CandidateFact]:
        extracted: list[CandidateFact] = []
        for turn in turns:
            # Never extract durable user profile memories from model-authored turns
            if turn.role != "user":
                continue
            text = " ".join(turn.text.split())
            lower = text.lower()

            if any(pat.search(text) for pat in self._EPHEMERAL_PATTERNS):
                extracted.append(
                    CandidateFact(
                        category="ephemeral_telemetry",
                        slot_key="telemetry.daily_conditions",
                        fact_text=text,
                        is_ephemeral=True,
                        confidence=0.95,
                    )
                )
                continue

            # Slot 1: Vessel hull and draft
            draft_match = re.search(r"draws\s+(\d+)\s+inches", lower)
            if draft_match:
                draft_in = draft_match.group(1)
                if "contender" in lower:
                    vessel_desc = (
                        f"Primary boat is a 25ft Contender bay boat that draws {draft_in} inches."
                    )
                elif "action craft" in lower or "skiff" in lower:
                    vessel_desc = (
                        f"Primary boat is an 18ft Action Craft flats skiff that draws {draft_in} inches."
                    )
                else:
                    vessel_desc = f"Primary boat draws {draft_in} inches."
                extracted.append(
                    CandidateFact(
                        category="vessel_spec",
                        slot_key="vessel.hull_and_draft",
                        fact_text=vessel_desc,
                        is_ephemeral=False,
                    )
                )

            # Slot 2: Home marina / launch ramp
            if "bud n' mary" in lower or "islamorada" in lower:
                extracted.append(
                    CandidateFact(
                        category="home_marina",
                        slot_key="marina.home_slip",
                        fact_text="Home slip is at Bud N' Mary's Marina in Islamorada.",
                        is_ephemeral=False,
                    )
                )
            elif "tavernier creek" in lower:
                extracted.append(
                    CandidateFact(
                        category="home_marina",
                        slot_key="marina.home_slip",
                        fact_text="Home launch ramp is Tavernier Creek Marina.",
                        is_ephemeral=False,
                    )
                )

            # Slot 3: Crew sea-state / motion-sickness constraint
            if "sarah" in lower and ("chop" in lower or "seas" in lower or "period" in lower):
                if "3.0" in lower and "7 seconds" in lower:
                     extracted.append(
                        CandidateFact(
                            category="crew_safety",
                            slot_key="crew.sarah_sea_state_limit",
                            fact_text=(
                                "Crew safety rule for Sarah: comfortable up to 3.0 ft seas when "
                                "swell period is 7 seconds or longer; still avoid short chop under 5 seconds."
                            ),
                            is_ephemeral=False,
                        )
                    )
                elif "2.5" in lower:
                    extracted.append(
                        CandidateFact(
                            category="crew_safety",
                            slot_key="crew.sarah_sea_state_limit",
                            fact_text=(
                                "Crew safety rule for Sarah: turn around if chop exceeds 2.5 ft "
                                "or wave period drops under 5 seconds."
                            ),
                            is_ephemeral=False,
                        )
                    )

            # Slot 4: Tackle & species rig preference
            if "barbless" in lower and "2/0" in lower and "tarpon" in lower:
                extracted.append(
                    CandidateFact(
                        category="tackle_preference",
                        slot_key="tackle.tarpon_hook_rig",
                        fact_text=(
                            "Tarpon rig preference at Channel Five bridge: barbless 2/0 circle hooks "
                            "on 60 lb fluorocarbon leader."
                        ),
                        is_ephemeral=False,
                    )
                )

        return extracted


class GeminiStructuredFactExtractor:
    """LLM-backed structured fact extractor on Vertex AI using Gemini JSON schema output."""

    def __init__(self, settings: MemoryBenchSettings) -> None:
        self._model = settings.gemini_model
        self._client = genai.Client(
            vertexai=settings.use_vertexai,
            project=settings.google_cloud_project,
            location=settings.google_cloud_location,
        )

    async def extract_facts(self, turns: Sequence[SessionTurn]) -> list[CandidateFact]:
        user_lines = [f"[{idx + 1}] {t.role.upper()}: {t.text}" for idx, t in enumerate(turns)]
        if not user_lines:
            return []
        prompt = "Extract durable user facts from this session transcript:\n\n" + "\n".join(
            user_lines
        )
        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=EXTRACTION_SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=FactExtractionBatch,
                temperature=0.0,
            ),
        )
        parsed = FactExtractionBatch.model_validate_json(response.text or '{"facts": []}')
        return parsed.facts
