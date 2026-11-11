"""Async HTTP adapters for NOAA CO-OPS, Open-Meteo Marine, NWS, and Nominatim."""

import asyncio
import re
from datetime import date, datetime

import httpx

from marine_bench.config import MarineBenchSettings
from marine_bench.domain import (
    calculate_lunar_phase,
    degrees_to_cardinal,
    haversine_miles,
    parse_wind_speed_mph,
    resolve_target_date,
)
from marine_bench.schemas import (
    MarineBriefContext,
    NoaaStation,
    ResolvedLocation,
    SeaState,
    TideExtreme,
    TideWindow,
    WeatherWindow,
)

COORDINATE_PATTERN = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$")


class LocationResolutionError(ValueError):
    """Raised when a coastal query cannot be geocoded or matched to a NOAA station."""


class UpstreamSourceError(RuntimeError):
    """Raised when an external weather, tide, or marine endpoint fails."""


class MarineDataClient:
    """Injectable async client isolating external HTTP calls from agent orchestration."""

    def __init__(
        self,
        http_client: httpx.AsyncClient,
        settings: MarineBenchSettings,
    ) -> None:
        self._http = http_client
        self._settings = settings
        self._station_cache: list[dict] | None = None

    async def _load_noaa_stations(self) -> list[dict]:
        if self._station_cache is not None:
            return self._station_cache

        response = await self._http.get(
            self._settings.noaa_stations_url,
            params={"type": "tidepredictions"},
            headers={"User-Agent": self._settings.user_agent},
        )
        response.raise_for_status()
        payload = response.json()
        stations = payload.get("stations") or []
        if not stations:
            raise UpstreamSourceError("NOAA station catalog returned zero stations")
        self._station_cache = stations
        return stations

    async def _find_nearest_station(self, lat: float, lon: float) -> NoaaStation:
        stations = await self._load_noaa_stations()
        best_station: NoaaStation | None = None

        for raw in stations:
            station_id = str(raw.get("id", ""))
            if not re.fullmatch(r"\d{7}", station_id):
                continue
            try:
                s_lat = float(raw["lat"])
                s_lon = float(raw["lng"])
            except (KeyError, TypeError, ValueError):
                continue

            dist = round(haversine_miles(lat, lon, s_lat, s_lon), 1)
            if best_station is None or dist < best_station.distance_miles:
                best_station = NoaaStation(
                    station_id=station_id,
                    station_name=str(raw.get("name", station_id)),
                    latitude=s_lat,
                    longitude=s_lon,
                    distance_miles=dist,
                )

        if best_station is None:
            raise LocationResolutionError(
                f"No valid 7-digit NOAA tide station found near ({lat:.4f}, {lon:.4f})"
            )
        return best_station

    async def resolve_location(self, location_query: str) -> ResolvedLocation:
        """Geocode a place name or 'lat, lon' string and attach its nearest NOAA station."""
        cleaned = location_query.strip()
        if not cleaned or len(cleaned) > 160:
            raise LocationResolutionError("Location query must be 1 to 160 characters")

        coord_match = COORDINATE_PATTERN.match(cleaned)
        if coord_match:
            lat = float(coord_match.group(1))
            lon = float(coord_match.group(2))
            if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                raise LocationResolutionError(
                    f"Coordinates ({lat}, {lon}) are outside valid WGS84 bounds"
                )
            station = await self._find_nearest_station(lat, lon)
            return ResolvedLocation(
                query=cleaned,
                resolved_name=f"{station.station_name} Area",
                latitude=lat,
                longitude=lon,
                station=station,
            )

        response = await self._http.get(
            self._settings.nominatim_search_url,
            params={"q": cleaned, "format": "jsonv2", "limit": 1, "countrycodes": "us"},
            headers={"User-Agent": self._settings.user_agent},
        )
        response.raise_for_status()
        matches = response.json()
        if not matches:
            raise LocationResolutionError(
                f"Could not geocode coastal US location '{cleaned}'"
            )

        top = matches[0]
        lat = float(top["lat"])
        lon = float(top["lon"])
        display_name = str(top.get("display_name", cleaned)).split(",")[0].strip()
        station = await self._find_nearest_station(lat, lon)
        return ResolvedLocation(
            query=cleaned,
            resolved_name=display_name or cleaned,
            latitude=lat,
            longitude=lon,
            station=station,
        )

    async def fetch_tides(self, station: NoaaStation, target_date: date) -> TideWindow:
        """Fetch daily high/low MLLW tide predictions from NOAA CO-OPS."""
        yyyymmdd = target_date.strftime("%Y%m%d")
        response = await self._http.get(
            self._settings.noaa_datagetter_url,
            params={
                "begin_date": yyyymmdd,
                "end_date": yyyymmdd,
                "station": station.station_id,
                "product": "predictions",
                "datum": "MLLW",
                "time_zone": "lst_ldt",
                "interval": "hilo",
                "units": "english",
                "format": "json",
            },
            headers={"User-Agent": self._settings.user_agent},
        )
        response.raise_for_status()
        payload = response.json()
        if "error" in payload:
            raise UpstreamSourceError(str(payload["error"].get("message", payload["error"])))

        highs: list[TideExtreme] = []
        lows: list[TideExtreme] = []
        for row in payload.get("predictions", []):
            dt = datetime.strptime(row["t"], "%Y-%m-%d %H:%M")
            extreme = TideExtreme(
                time_local=dt.strftime("%-I:%M %p"),
                tide_type="high" if row.get("type") == "H" else "low",
                height_ft=round(float(row["v"]), 2),
            )
            if extreme.tide_type == "high":
                highs.append(extreme)
            else:
                lows.append(extreme)

        return TideWindow(
            station_id=station.station_id,
            station_name=station.station_name,
            target_date=target_date.isoformat(),
            high_tides=tuple(highs),
            low_tides=tuple(lows),
        )

    async def fetch_sea_state(
        self,
        latitude: float,
        longitude: float,
        target_date: date,
    ) -> SeaState:
        """Fetch wave height, swell height, period, and SST from Open-Meteo Marine."""
        iso_day = target_date.isoformat()
        response = await self._http.get(
            self._settings.open_meteo_marine_url,
            params={
                "latitude": round(latitude, 4),
                "longitude": round(longitude, 4),
                "daily": "wave_height_max,swell_wave_height_max,wave_period_max,wave_direction_dominant",
                "hourly": "sea_surface_temperature",
                "temperature_unit": "fahrenheit",
                "length_unit": "imperial",
                "timezone": "America/New_York",
                "start_date": iso_day,
                "end_date": iso_day,
            },
            headers={"User-Agent": self._settings.user_agent},
        )
        response.raise_for_status()
        payload = response.json()
        daily = payload.get("daily") or {}

        wave_ft = float((daily.get("wave_height_max") or [0.0])[0] or 0.0)
        swell_ft = float((daily.get("swell_wave_height_max") or [0.0])[0] or 0.0)
        period_sec = float((daily.get("wave_period_max") or [0.0])[0] or 0.0)
        direction_deg = float((daily.get("wave_direction_dominant") or [90.0])[0] or 90.0)

        sst_values = [
            float(v)
            for v in (payload.get("hourly", {}).get("sea_surface_temperature") or [])
            if v is not None
        ]
        sst_f = round(sum(sst_values) / len(sst_values), 1) if sst_values else None

        return SeaState(
            target_date=iso_day,
            wave_height_ft=round(wave_ft, 1),
            swell_height_ft=round(swell_ft, 1),
            wave_period_sec=round(period_sec, 1),
            wave_direction_cardinal=degrees_to_cardinal(direction_deg),
            sea_surface_temp_f=sst_f,
        )

    async def fetch_weather(
        self,
        latitude: float,
        longitude: float,
        target_date: date,
    ) -> WeatherWindow:
        """Fetch NWS surface forecast period matching target_date."""
        headers = {
            "User-Agent": self._settings.user_agent,
            "Accept": "application/geo+json",
        }
        points_url = f"{self._settings.nws_base_url}/points/{latitude:.4f},{longitude:.4f}"
        points_resp = await self._http.get(points_url, headers=headers)
        if points_resp.status_code >= 400:
            raise UpstreamSourceError(
                f"NWS /points returned HTTP {points_resp.status_code}"
            )

        forecast_url = points_resp.json().get("properties", {}).get("forecast")
        if not forecast_url or not str(forecast_url).startswith(
            f"{self._settings.nws_base_url}/"
        ):
            raise UpstreamSourceError("NWS response missing valid forecast URL")

        forecast_resp = await self._http.get(forecast_url, headers=headers)
        if forecast_resp.status_code >= 400:
            raise UpstreamSourceError(
                f"NWS forecast returned HTTP {forecast_resp.status_code}"
            )

        periods = forecast_resp.json().get("properties", {}).get("periods") or []
        if not periods:
            raise UpstreamSourceError("NWS forecast contained zero periods")

        iso_prefix = target_date.isoformat()
        selected = next(
            (
                p
                for p in periods
                if str(p.get("startTime", "")).startswith(iso_prefix)
                and p.get("isDaytime", True)
            ),
            next(
                (
                    p
                    for p in periods
                    if str(p.get("startTime", "")).startswith(iso_prefix)
                ),
                periods[0],
            ),
        )

        precip = (selected.get("probabilityOfPrecipitation") or {}).get("value")
        return WeatherWindow(
            target_date=iso_prefix,
            period_name=str(selected.get("name", "Forecast")),
            air_temp_f=int(selected.get("temperature", 75)),
            wind_speed_max_mph=parse_wind_speed_mph(str(selected.get("windSpeed", "0 mph"))),
            wind_direction=str(selected.get("windDirection", "Variable")),
            short_forecast=str(selected.get("shortForecast", "Unknown")),
            precipitation_pct=int(precip) if precip is not None else 0,
        )

    async def collect_snapshot(
        self,
        location_query: str,
        date_query: str | None = None,
        reference_date: date | None = None,
    ) -> MarineBriefContext:
        """Resolve location first, then fan out across tide, sea, weather, and moon."""
        target = resolve_target_date(date_query, reference_date=reference_date)
        resolved = await self.resolve_location(location_query)
        lunar = calculate_lunar_phase(target)

        results = await asyncio.gather(
            self.fetch_tides(resolved.station, target),
            self.fetch_sea_state(resolved.latitude, resolved.longitude, target),
            self.fetch_weather(resolved.latitude, resolved.longitude, target),
            return_exceptions=True,
        )

        degraded: list[str] = []
        tides_res, sea_res, weather_res = results

        if isinstance(tides_res, BaseException):
            degraded.append("noaa_tides")
            tides_val = None
        else:
            tides_val = tides_res

        if isinstance(sea_res, BaseException):
            degraded.append("open_meteo_marine")
            sea_val = None
        else:
            sea_val = sea_res

        if isinstance(weather_res, BaseException):
            degraded.append("nws_weather")
            weather_val = None
        else:
            weather_val = weather_res

        return MarineBriefContext(
            location=resolved,
            target_date=target.isoformat(),
            tides=tides_val,
            sea_state=sea_val,
            weather=weather_val,
            lunar=lunar,
            degraded_sources=tuple(degraded),
        )
