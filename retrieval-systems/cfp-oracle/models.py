"""Typed Pydantic v2 schemas and environment settings for CFP Oracle."""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Teams(BaseModel):
    """Winning and losing teams for a historical game."""

    model_config = ConfigDict(frozen=True)

    winner: str = Field(min_length=1)
    loser: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_distinct_teams(self) -> Teams:
        if self.winner.strip().lower() == self.loser.strip().lower():
            raise ValueError("winner and loser must be distinct teams")
        return self


class Score(BaseModel):
    """Final box score for a completed game."""

    model_config = ConfigDict(frozen=True)

    winner: int = Field(ge=0)
    loser: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_winning_score(self) -> Score:
        if self.winner <= self.loser:
            raise ValueError("winner score must exceed loser score")
        return self

    @property
    def margin(self) -> int:
        """Point differential between winner and loser."""
        return self.winner - self.loser


class Seeds(BaseModel):
    """Playoff or BCS seeds for the participating teams."""

    model_config = ConfigDict(frozen=True)

    winner: int = Field(ge=1, le=25)
    loser: int = Field(ge=1, le=25)


class KeyStats(BaseModel):
    """Structured game-flow indicators used to enrich narrative embeddings."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    overtime: bool | None = None
    double_overtime: bool | None = None
    comeback: int | None = Field(default=None, ge=1)
    shutout: bool | None = None
    upset: bool | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_flags(cls, raw: Any) -> Any:
        if not isinstance(raw, dict):
            return raw
        normalized = dict(raw)
        if "upset" not in normalized and "upsets" in normalized:
            normalized["upset"] = normalized["upsets"]
        if normalized.get("double_overtime") and normalized.get("overtime") is None:
            normalized["overtime"] = True
        return normalized


class Game(BaseModel):
    """Historical College Football Playoff or BCS bowl game."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1)
    game: str = Field(min_length=1)
    year: int = Field(ge=1998, le=2100)
    date: str | None = None
    teams: Teams
    seeds: Seeds | None = None
    score: Score
    narrative: str = Field(min_length=20)
    venue: str | None = None
    tags: list[str] = Field(default_factory=list)
    key_stats: KeyStats | None = None


class VectorMetadata(BaseModel):
    """Metadata payload stored alongside each vector in Amazon S3 Vectors."""

    model_config = ConfigDict(frozen=True)

    game_id: str = Field(min_length=1)
    game_name: str = Field(min_length=1)
    year: int = Field(ge=1998, le=2100)
    winner: str = Field(min_length=1)
    loser: str = Field(min_length=1)
    winner_score: int = Field(ge=0)
    loser_score: int = Field(ge=0)
    margin: int = Field(gt=0)
    narrative: str = Field(min_length=1, max_length=500)
    tags: str
    venue: str = ""
    overtime: bool | None = None
    comeback_points: int | None = None
    shutout: bool | None = None
    upset: bool | None = None


class SearchResult(BaseModel):
    """Ranked match returned from an S3 Vectors or local cosine query."""

    model_config = ConfigDict(frozen=True)

    key: str = Field(min_length=1)
    distance: float = Field(ge=0.0, le=2.0)
    metadata: VectorMetadata

    @property
    def similarity(self) -> float:
        """Convert cosine distance in [0, 2] to cosine similarity in [-1, 1]."""
        return 1.0 - self.distance


def _default_aws_region() -> str:
    return (
        os.getenv("AWS_REGION")
        or os.getenv("AWS_DEFAULT_REGION")
        or "us-east-1"
    )


class S3VectorConfig(BaseSettings):
    """Immutable infrastructure configuration loaded from CFP_ORACLE_* env vars."""

    model_config = SettingsConfigDict(
        env_prefix="CFP_ORACLE_",
        frozen=True,
        extra="ignore",
    )

    bucket_name: str = Field(default="cfp-oracle-vectors", min_length=3)
    index_name: str = Field(default="historical-games", min_length=1)
    embedding_model: str = Field(
        default="amazon.titan-embed-text-v2:0",
        min_length=1,
    )
    embedding_dimension: int = Field(default=1024, gt=0)
    region: str = Field(default_factory=_default_aws_region, min_length=1)
    batch_size: int = Field(default=50, ge=1, le=100)
