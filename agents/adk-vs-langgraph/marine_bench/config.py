"""Immutable startup configuration loaded from environment variables."""

from functools import lru_cache
from urllib.parse import urlparse

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ALLOWED_UPSTREAM_HOSTS = frozenset(
    {
        "api.tidesandcurrents.noaa.gov",
        "marine-api.open-meteo.com",
        "api.weather.gov",
        "nominatim.openstreetmap.org",
    }
)


class MarineBenchSettings(BaseSettings):
    """Process-level configuration for marine data adapters and Vertex AI models."""

    model_config = SettingsConfigDict(
        frozen=True,
        env_prefix="MARINE_BENCH_",
        extra="ignore",
    )

    google_cloud_project: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "GOOGLE_CLOUD_PROJECT",
            "MARINE_BENCH_GOOGLE_CLOUD_PROJECT",
        ),
    )
    google_cloud_location: str = Field(
        default="us-central1",
        validation_alias=AliasChoices(
            "GOOGLE_CLOUD_LOCATION",
            "MARINE_BENCH_GOOGLE_CLOUD_LOCATION",
        ),
    )
    use_vertexai: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "GOOGLE_GENAI_USE_VERTEXAI",
            "MARINE_BENCH_USE_VERTEXAI",
        ),
    )
    gemini_model: str = Field(default="gemini-2.5-flash")
    http_timeout_seconds: float = Field(default=10.0, gt=0.5, le=60.0)
    user_agent: str = Field(default="largo-marine-bench/1.0 (+https://largo.dev)")

    noaa_stations_url: str = Field(
        default="https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json"
    )
    noaa_datagetter_url: str = Field(
        default="https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
    )
    open_meteo_marine_url: str = Field(
        default="https://marine-api.open-meteo.com/v1/marine"
    )
    nws_base_url: str = Field(default="https://api.weather.gov")
    nominatim_search_url: str = Field(
        default="https://nominatim.openstreetmap.org/search"
    )

    @field_validator(
        "noaa_stations_url",
        "noaa_datagetter_url",
        "open_meteo_marine_url",
        "nws_base_url",
        "nominatim_search_url",
    )
    @classmethod
    def validate_https_endpoint(cls, value: str) -> str:
        """Restrict outbound calls to HTTPS endpoints on allowlisted public hosts."""
        parsed = urlparse(value)
        if parsed.scheme != "https":
            raise ValueError(f"Endpoint must use https:// scheme: {value}")
        if parsed.hostname not in ALLOWED_UPSTREAM_HOSTS:
            raise ValueError(
                f"Hostname '{parsed.hostname}' is outside ALLOWED_UPSTREAM_HOSTS"
            )
        return value.rstrip("/")


@lru_cache(maxsize=1)
def load_settings() -> MarineBenchSettings:
    """Load immutable settings once per process."""
    return MarineBenchSettings()
