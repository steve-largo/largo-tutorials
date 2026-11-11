"""Pure domain functions for date resolution, marine unit parsing, and synthesis prompts."""

import math
import re
from datetime import date, datetime, timedelta

from dateutil import parser as dateutil_parser

from marine_bench.schemas import LunarPhaseInfo, MarineBriefContext

CARDINAL_DIRECTIONS = (
    "N",
    "NNE",
    "NE",
    "ENE",
    "E",
    "ESE",
    "SE",
    "SSE",
    "S",
    "SSW",
    "SW",
    "WSW",
    "W",
    "WNW",
    "NW",
    "NNW",
)

WEEKDAY_INDEX = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}

SYNTHESIS_SYSTEM_INSTRUCTION = """You are an experienced Florida Keys and coastal US fishing captain.
Given a structured marine telemetry snapshot, write a concise, practical conditions report with:
1. Overall Verdict (Go / Caution / Stay at the Dock) with primary reason
2. Tide & Current Windows (exact high/low times and whether incoming flats or outgoing channels favor the bite)
3. Wind, Sea State & Boat Handling (cite exact wind mph, wave height, swell period, and water temperature)
4. Target Species & Tackle Plan
5. Data Completeness Note (explicitly note if any upstream source is listed under degraded_sources; never invent missing readings)
"""


def resolve_target_date(
    date_text: str | None = None,
    reference_date: date | None = None,
) -> date:
    """Resolve natural-language or ISO date strings relative to reference_date."""
    base = reference_date or date.today()
    if not date_text or not date_text.strip():
        return base

    normalized = date_text.strip().lower()
    if normalized in {"today", "now", "tonight"}:
        return base
    if normalized == "tomorrow":
        return base + timedelta(days=1)
    if normalized == "yesterday":
        return base - timedelta(days=1)

    in_days_match = re.fullmatch(r"in\s+(\d{1,2})\s+days?", normalized)
    if in_days_match:
        return base + timedelta(days=int(in_days_match.group(1)))

    if normalized in {"this weekend", "weekend"}:
        days_ahead = (5 - base.weekday()) % 7
        return base + timedelta(days=days_ahead)

    if normalized == "next weekend":
        days_ahead = ((5 - base.weekday()) % 7) + 7
        return base + timedelta(days=days_ahead)

    for day_name, target_idx in WEEKDAY_INDEX.items():
        if normalized in {day_name, f"this {day_name}"}:
            days_ahead = (target_idx - base.weekday()) % 7
            return base + timedelta(days=days_ahead or 7)
        if normalized == f"next {day_name}":
            days_ahead = ((target_idx - base.weekday()) % 7) + 7
            return base + timedelta(days=days_ahead)

    parsed_dt = dateutil_parser.parse(
        date_text,
        default=datetime(base.year, base.month, base.day),
    )
    return parsed_dt.date()


def parse_wind_speed_mph(raw_wind: str) -> int:
    """Extract upper-bound wind speed in mph (e.g., '10 to 15 mph' -> 15, 'Calm' -> 0)."""
    numbers = [int(match) for match in re.findall(r"\d+", raw_wind)]
    if not numbers:
        return 0
    return max(numbers)


def degrees_to_cardinal(degrees: float) -> str:
    """Convert compass degrees [0, 360) into a 16-point cardinal label."""
    index = round((degrees % 360.0) / 22.5) % 16
    return CARDINAL_DIRECTIONS[index]


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance in statute miles between two WGS84 points."""
    earth_radius_miles = 3958.7613
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    )
    return 2.0 * earth_radius_miles * math.asin(min(1.0, math.sqrt(a)))


def calculate_lunar_phase(target_date: date) -> LunarPhaseInfo:
    """Compute lunar phase and spring/neap tidal current strength."""
    target_dt = datetime(target_date.year, target_date.month, target_date.day, 12, 0)
    reference_new_moon = datetime(2000, 1, 6, 18, 14)
    synodic_month = 29.53058867

    elapsed_days = (target_dt - reference_new_moon).total_seconds() / 86400.0
    phase_age = elapsed_days % synodic_month
    illumination = round((1.0 - math.cos(2.0 * math.pi * phase_age / synodic_month)) * 50.0)

    if phase_age < 1.85 or phase_age >= 27.68:
        phase_name = "New Moon"
    elif phase_age < 7.38:
        phase_name = "Waxing Crescent"
    elif phase_age < 9.23:
        phase_name = "First Quarter"
    elif phase_age < 14.77:
        phase_name = "Waxing Gibbous"
    elif phase_age < 16.61:
        phase_name = "Full Moon"
    elif phase_age < 22.15:
        phase_name = "Waning Gibbous"
    elif phase_age < 23.99:
        phase_name = "Last Quarter"
    else:
        phase_name = "Waning Crescent"

    if phase_name in {"New Moon", "Full Moon"}:
        rating = "Excellent"
        note = "Spring-tide amplitude increases water movement across flats and channels."
    elif phase_name in {"First Quarter", "Last Quarter"}:
        rating = "Good"
        note = "Neap-tide moderate flow; prioritize peak mid-tide velocity windows."
    else:
        rating = "Fair"
        note = "Transitional tidal range; match wind direction with moving water."

    return LunarPhaseInfo(
        target_date=target_date.isoformat(),
        phase_name=phase_name,
        illumination_pct=max(0, min(100, illumination)),
        fishing_rating=rating,
        tidal_note=note,
    )


def format_snapshot_for_synthesis(context: MarineBriefContext) -> str:
    """Serialize MarineBriefContext into deterministic structured text for synthesis."""
    lines = [
        f"Location: {context.location.resolved_name} ({context.location.latitude:.4f}, {context.location.longitude:.4f})",
        f"Nearest NOAA Station: {context.location.station.station_name} (#{context.location.station.station_id}, {context.location.station.distance_miles} mi)",
        f"Target Date: {context.target_date}",
        f"Lunar Phase: {context.lunar.phase_name} ({context.lunar.illumination_pct}% illumination, {context.lunar.fishing_rating}) - {context.lunar.tidal_note}",
    ]

    if context.tides:
        highs = ", ".join(f"{t.time_local} ({t.height_ft:+.2f} ft)" for t in context.tides.high_tides) or "None"
        lows = ", ".join(f"{t.time_local} ({t.height_ft:+.2f} ft)" for t in context.tides.low_tides) or "None"
        lines.append(f"High Tides (MLLW): {highs}")
        lines.append(f"Low Tides (MLLW): {lows}")
    else:
        lines.append("Tides: UNAVAILABLE")

    if context.sea_state:
        sst = (
            f"{context.sea_state.sea_surface_temp_f:.1f} F"
            if context.sea_state.sea_surface_temp_f is not None
            else "n/a"
        )
        lines.append(
            f"Sea State: waves {context.sea_state.wave_height_ft:.1f} ft from {context.sea_state.wave_direction_cardinal} "
            f"@ {context.sea_state.wave_period_sec:.1f}s period, swell {context.sea_state.swell_height_ft:.1f} ft, SST {sst}"
        )
    else:
        lines.append("Sea State: UNAVAILABLE")

    if context.weather:
        lines.append(
            f"NWS Forecast ({context.weather.period_name}): {context.weather.air_temp_f} F, "
            f"wind {context.weather.wind_direction} up to {context.weather.wind_speed_max_mph} mph, "
            f"{context.weather.short_forecast} (precip {context.weather.precipitation_pct}%)"
        )
    else:
        lines.append("NWS Forecast: UNAVAILABLE")

    if context.degraded_sources:
        lines.append(f"Degraded Sources: {', '.join(context.degraded_sources)}")
    else:
        lines.append("Degraded Sources: none")

    return "\n".join(lines)
