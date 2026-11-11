"""Typed, immutable domain contracts shared across ADK and LangGraph orchestrators."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class NoaaStation(BaseModel):
    """Nearest NOAA CO-OPS harmonic tide prediction station."""

    model_config = ConfigDict(frozen=True)

    station_id: str = Field(pattern=r"^\d{7}$")
    station_name: str = Field(min_length=1, max_length=120)
    latitude: float = Field(ge=-90.0, le=90.0)
    longitude: float = Field(ge=-180.0, le=180.0)
    distance_miles: float = Field(ge=0.0)


class ResolvedLocation(BaseModel):
    """Geocoded coastal coordinate paired with its nearest NOAA tide station."""

    model_config = ConfigDict(frozen=True)

    query: str = Field(min_length=1, max_length=160)
    resolved_name: str = Field(min_length=1, max_length=200)
    latitude: float = Field(ge=-90.0, le=90.0)
    longitude: float = Field(ge=-180.0, le=180.0)
    station: NoaaStation


class TideExtreme(BaseModel):
    """Single high or low tide event in station local time."""

    model_config = ConfigDict(frozen=True)

    time_local: str = Field(min_length=1, max_length=32)
    tide_type: Literal["high", "low"]
    height_ft: float


class TideWindow(BaseModel):
    """Daily high and low tide schedule from NOAA CO-OPS."""

    model_config = ConfigDict(frozen=True)

    station_id: str = Field(pattern=r"^\d{7}$")
    station_name: str
    target_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    high_tides: tuple[TideExtreme, ...]
    low_tides: tuple[TideExtreme, ...]


class SeaState(BaseModel):
    """Offshore wave, swell, and sea surface temperature conditions."""

    model_config = ConfigDict(frozen=True)

    target_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    wave_height_ft: float = Field(ge=0.0)
    swell_height_ft: float = Field(ge=0.0)
    wave_period_sec: float = Field(ge=0.0)
    wave_direction_cardinal: str
    sea_surface_temp_f: float | None = None


class WeatherWindow(BaseModel):
    """Surface wind, air temperature, and precipitation forecast from NWS."""

    model_config = ConfigDict(frozen=True)

    target_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    period_name: str
    air_temp_f: int
    wind_speed_max_mph: int = Field(ge=0)
    wind_direction: str
    short_forecast: str
    precipitation_pct: int = Field(ge=0, le=100)


class LunarPhaseInfo(BaseModel):
    """Astronomical lunar phase and tidal current implications."""

    model_config = ConfigDict(frozen=True)

    target_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    phase_name: str
    illumination_pct: int = Field(ge=0, le=100)
    fishing_rating: Literal["Excellent", "Good", "Fair"]
    tidal_note: str


class MarineBriefContext(BaseModel):
    """Aggregated marine telemetry passed into final report synthesis."""

    model_config = ConfigDict(frozen=True)

    location: ResolvedLocation
    target_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    tides: TideWindow | None = None
    sea_state: SeaState | None = None
    weather: WeatherWindow | None = None
    lunar: LunarPhaseInfo
    degraded_sources: tuple[str, ...] = ()


class RunTelemetry(BaseModel):
    """Execution telemetry recorded across ADK and LangGraph runs."""

    model_config = ConfigDict(frozen=True)

    framework: str
    llm_calls: int = Field(ge=0)
    tool_invocations: int = Field(ge=0)
    wall_time_ms: float = Field(ge=0.0)
    degraded_sources: tuple[str, ...] = ()
