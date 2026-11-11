"""Eight-session longitudinal Florida Keys dataset and evaluation probes (May to November)."""

from __future__ import annotations

from memory_bench.schemas import LongitudinalSession, MemoryProbe, SessionTurn

BENCHMARK_APP_NAME = "keys_marine_captain"
PRIMARY_USER_ID = "captain_steve"
SECONDARY_USER_ID = "angler_bob"

COASTAL_LONGITUDINAL_SESSIONS: tuple[LongitudinalSession, ...] = (
    LongitudinalSession(
        session_id="sess-01-may-skiff",
        recorded_date="2026-05-04",
        summary_label="Initial boat and launch marina setup",
        turns=(
            SessionTurn(
                role="user",
                text=(
                    "Logging my setup for the season: I run an 18ft Action Craft flats skiff "
                    "that draws 9 inches, launching out of Tavernier Creek."
                ),
            ),
            SessionTurn(
                role="model",
                text=(
                    "Logged your 18ft Action Craft skiff (9-inch draft) out of Tavernier Creek. "
                    "Tomorrow's morning high tide at Tavernier is +1.9 ft at 7:15 AM with E winds at 8 mph."
                ),
            ),
        ),
    ),
    LongitudinalSession(
        session_id="sess-02-may-blowout",
        recorded_date="2026-05-19",
        summary_label="Ephemeral rough-weather observation",
        turns=(
            SessionTurn(
                role="user",
                text=(
                    "High tide was +2.1 ft at 3:10 PM today with 22 mph east gusts and whitecaps "
                    "in Hawk Channel, so we stayed at the dock."
                ),
            ),
            SessionTurn(
                role="model",
                text=(
                    "Smart call staying in port; NWS posted a Small Craft Exercise Caution through midnight "
                    "with 4.5 ft wind waves in Hawk Channel."
                ),
            ),
        ),
    ),
    LongitudinalSession(
        session_id="sess-03-jun-crew-limit",
        recorded_date="2026-06-08",
        summary_label="Initial crew sea-state threshold for Sarah",
        turns=(
            SessionTurn(
                role="user",
                text=(
                    "Quick crew note: whenever my wife Sarah comes out to the reef, we turn around "
                    "if chop exceeds 2.5 feet or wave period drops under 5 seconds."
                ),
            ),
            SessionTurn(
                role="model",
                thought_text=(
                    "Internal reasoning trace: user mentioned family member threshold 2.5ft / 5s; "
                    "do not persist internal scratchpad reasoning into memory."
                ),
                text=(
                    "Noted. When Sarah is aboard, I will flag any reef window exceeding 2.5 ft chop "
                    "or periods shorter than 5 seconds."
                ),
            ),
        ),
    ),
    LongitudinalSession(
        session_id="sess-04-jul-tarpon-rig",
        recorded_date="2026-07-02",
        summary_label="Tarpon tackle preference at Channel Five",
        turns=(
            SessionTurn(
                role="user",
                text=(
                    "For evening tarpon drifts under Channel Five bridge, I switched exclusively "
                    "to barbless 2/0 circle hooks on 60 lb fluorocarbon leader."
                ),
            ),
            SessionTurn(
                role="model",
                text="Got it: barbless 2/0 circle hooks on 60 lb fluorocarbon for Channel Five tarpon.",
            ),
        ),
    ),
    LongitudinalSession(
        session_id="sess-05-aug-boat-upgrade",
        recorded_date="2026-08-14",
        summary_label="Vessel upgrade and marina relocation (State Mutation 1)",
        turns=(
            SessionTurn(
                role="user",
                text=(
                    "Big update: sold the Action Craft skiff last weekend and picked up a 25ft Contender "
                    "bay boat. She draws 16 inches now, and we moved our slip to Bud N' Mary's in Islamorada."
                ),
            ),
            SessionTurn(
                role="model",
                text=(
                    "Congrats on the 25ft Contender! Updated your draft to 16 inches and departure "
                    "marina to Bud N' Mary's in Islamorada."
                ),
            ),
        ),
    ),
    LongitudinalSession(
        session_id="sess-06-sep-advisory",
        recorded_date="2026-09-09",
        summary_label="Ephemeral September tropical wave advisory",
        turns=(
            SessionTurn(
                role="user",
                text=(
                    "NWS forecast just issued a Small Craft Advisory for 28 mph NE winds and "
                    "6 ft seas offshore Islamorada through Wednesday."
                ),
            ),
            SessionTurn(
                role="model",
                text=(
                    "Confirmed: Small Craft Advisory active with 28 mph NE winds and 6 ft offshore seas. "
                    "Backcountry basins inside Florida Bay remain the only sheltered option."
                ),
            ),
        ),
    ),
    LongitudinalSession(
        session_id="sess-07-oct-crew-update",
        recorded_date="2026-10-03",
        summary_label="Refined crew sea-state rule for Sarah (State Mutation 2)",
        turns=(
            SessionTurn(
                role="user",
                text=(
                    "Update on Sarah's reef rule: relief bands worked well on the Contender, so she is "
                    "comfortable up to 3.0 ft seas now when swell period is 7 seconds or longer, "
                    "still avoiding short chop under 5 seconds."
                ),
            ),
            SessionTurn(
                role="model",
                text=(
                    "Updated Sarah's comfort envelope: up to 3.0 ft seas at >= 7s period, "
                    "still holding the 5-second minimum period floor."
                ),
            ),
        ),
    ),
    LongitudinalSession(
        session_id="sess-08-oct-bait-report",
        recorded_date="2026-10-19",
        summary_label="Recent ephemeral fall bait run check-in",
        turns=(
            SessionTurn(
                role="user",
                text=(
                    "High tide was +2.4 ft at Whale Harbor this morning with finger mullet pushing "
                    "south on 12 mph north winds; stayed at the dock working on electronics."
                ),
            ),
            SessionTurn(
                role="model",
                text="Classic October mullet push on the king tide; winds ease to 8 mph ENE by Friday.",
            ),
        ),
    ),
)

COASTAL_EVALUATION_PROBES: tuple[MemoryProbe, ...] = (
    MemoryProbe(
        probe_id="probe-1-draft-and-marina-mutation",
        user_id=PRIMARY_USER_ID,
        query=(
            "Low tide across Snake Creek flat drops to 12 inches tomorrow morning. "
            "Can my boat clear it safely, and which marina are we leaving from?"
        ),
        required_substrings=("16 inches", "Bud N' Mary"),
        forbidden_stale_substrings=("9 inches", "9-inch", "Tavernier Creek"),
        forbidden_noise_substrings=("28 mph", "22 mph", "+2.1 ft"),
    ),
    MemoryProbe(
        probe_id="probe-2-crew-rule-mutation",
        user_id=PRIMARY_USER_ID,
        query=(
            "Thinking about taking Sarah out to Alligator Reef tomorrow; swell is 2.8 ft "
            "at an 8-second period. Does that fit our crew rules?"
        ),
        required_substrings=("3.0", "7 seconds"),
        forbidden_stale_substrings=("exceeds 2.5", "2.5 ft chop"),
        forbidden_noise_substrings=("28 mph", "Small Craft Advisory"),
    ),
    MemoryProbe(
        probe_id="probe-3-long-range-tackle-recall",
        user_id=PRIMARY_USER_ID,
        query="Packing the tackle bag for Channel Five tarpon tonight, what hook setup do I run?",
        required_substrings=("barbless", "2/0"),
        forbidden_stale_substrings=(),
        forbidden_noise_substrings=("28 mph", "+2.1 ft"),
    ),
    MemoryProbe(
        probe_id="probe-4-clean-profile-no-weather-noise",
        user_id=PRIMARY_USER_ID,
        query="Summarize my current boat draft, home marina, and crew wave limits.",
        required_substrings=("16 inches", "Bud N' Mary", "3.0"),
        forbidden_stale_substrings=("9 inches", "Tavernier Creek", "exceeds 2.5"),
        forbidden_noise_substrings=("28 mph", "22 mph", "Small Craft Advisory", "Internal reasoning trace"),
    ),
    MemoryProbe(
        probe_id="probe-5-cross-tenant-isolation",
        user_id=SECONDARY_USER_ID,
        query="What boat draft and marina slip do you have on file for me?",
        required_substrings=(),
        forbidden_stale_substrings=(),
        forbidden_noise_substrings=(),
        expect_empty=True,
    ),
)
