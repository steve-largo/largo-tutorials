"""Offline unit, schema, dataset integrity, and adapter tests for CFP Oracle."""

from __future__ import annotations

import io
import json
from typing import Any

import pytest
from pydantic import ValidationError

from cfp_oracle import (
    cleanup,
    cosine_distance,
    cosine_similarity,
    create_game_text,
    find_similar_games,
    format_result,
    game_to_metadata,
    load_games,
    rank_games_by_similarity,
    setup_infrastructure,
)
from games_data import (
    HISTORICAL_GAMES,
    filter_games,
    get_games_by_tag,
    get_games_by_team,
    get_games_by_year,
    load_historical_games,
)
from models import (
    Game,
    KeyStats,
    S3VectorConfig,
    Score,
    SearchResult,
    Teams,
    VectorMetadata,
)


class _FakeNotFoundError(Exception):
    """Simulates botocore S3 Vectors NotFoundException."""

    def __init__(self, message: str = "not found") -> None:
        super().__init__(message)
        self.response = {"Error": {"Code": "NotFoundException"}}


class _FakeBedrockClient:
    """Deterministic in-memory Bedrock embedding fake."""

    def __init__(self, dimension: int = 8) -> None:
        self.dimension = dimension
        self.calls: list[tuple[str, str]] = []

    def invoke_model(self, *, modelId: str, body: str) -> dict[str, Any]:
        payload = json.loads(body)
        text: str = payload["inputText"]
        self.calls.append((modelId, text))
        lowered = text.lower()
        # Deterministic keyword-sensitive basis vector for offline verification
        features = [
            float(lowered.count("upset")),
            float(lowered.count("comeback")),
            float(lowered.count("shutout")),
            float(lowered.count("overtime")),
            float(lowered.count("quarterback") + lowered.count("vince young")),
            float(lowered.count("miami")),
            float(lowered.count("alabama")),
            0.05,
        ]
        vec = features[: self.dimension]
        encoded = json.dumps({"embedding": vec}).encode("utf-8")
        return {"body": io.BytesIO(encoded)}


class _FakeS3VectorsClient:
    """In-memory fake implementing the S3 Vectors bucket/index/query contract."""

    def __init__(self) -> None:
        self.buckets: set[str] = set()
        self.indexes: dict[tuple[str, str], dict[str, Any]] = {}
        self.vectors: dict[tuple[str, str], list[dict[str, Any]]] = {}

    def get_vector_bucket(self, *, vectorBucketName: str) -> dict[str, Any]:
        if vectorBucketName not in self.buckets:
            raise _FakeNotFoundError(vectorBucketName)
        return {"vectorBucket": {"vectorBucketName": vectorBucketName}}

    def create_vector_bucket(self, *, vectorBucketName: str) -> None:
        self.buckets.add(vectorBucketName)

    def delete_vector_bucket(self, *, vectorBucketName: str) -> None:
        if vectorBucketName not in self.buckets:
            raise _FakeNotFoundError(vectorBucketName)
        self.buckets.remove(vectorBucketName)

    def get_index(self, *, vectorBucketName: str, indexName: str) -> dict[str, Any]:
        key = (vectorBucketName, indexName)
        if key not in self.indexes:
            raise _FakeNotFoundError(indexName)
        return {"index": self.indexes[key]}

    def create_index(
        self,
        *,
        vectorBucketName: str,
        indexName: str,
        dataType: str,
        dimension: int,
        distanceMetric: str,
    ) -> None:
        key = (vectorBucketName, indexName)
        self.indexes[key] = {
            "dataType": dataType,
            "dimension": dimension,
            "distanceMetric": distanceMetric,
        }
        self.vectors[key] = []

    def delete_index(self, *, vectorBucketName: str, indexName: str) -> None:
        key = (vectorBucketName, indexName)
        if key not in self.indexes:
            raise _FakeNotFoundError(indexName)
        del self.indexes[key]
        self.vectors.pop(key, None)

    def put_vectors(
        self,
        *,
        vectorBucketName: str,
        indexName: str,
        vectors: list[dict[str, Any]],
    ) -> None:
        key = (vectorBucketName, indexName)
        self.vectors.setdefault(key, []).extend(vectors)

    def query_vectors(
        self,
        *,
        vectorBucketName: str,
        indexName: str,
        queryVector: dict[str, list[float]],
        topK: int,
        returnDistance: bool = True,
        returnMetadata: bool = True,
        filter: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        q_vec = queryVector["float32"]
        stored = self.vectors.get((vectorBucketName, indexName), [])
        scored: list[dict[str, Any]] = []
        for entry in stored:
            meta = entry["metadata"]
            if filter and "year" in filter and meta.get("year") != filter["year"]:
                continue
            dist = cosine_distance(q_vec, entry["data"]["float32"])
            item: dict[str, Any] = {"key": entry["key"]}
            if returnDistance:
                item["distance"] = dist
            if returnMetadata:
                item["metadata"] = meta
            scored.append(item)
        scored.sort(key=lambda row: (row.get("distance", 0.0), row["key"]))
        return {"vectors": scored[:topK]}


def test_pydantic_schemas_and_invariants() -> None:
    teams = Teams(winner="Texas", loser="USC")
    score = Score(winner=41, loser=38)
    assert score.margin == 3

    with pytest.raises(ValidationError):
        Teams(winner="Texas", loser="texas")

    with pytest.raises(ValidationError):
        Score(winner=21, loser=21)

    with pytest.raises(ValidationError):
        Score(winner=-3, loser=0)

    # KeyStats normalizes double_overtime -> overtime and legacy upsets -> upset
    stats = KeyStats.model_validate({"double_overtime": True, "upsets": True})
    assert stats.overtime is True
    assert stats.upset is True

    meta = VectorMetadata(
        game_id="2006-rose-bowl",
        game_name="Rose Bowl - BCS Championship",
        year=2006,
        winner=teams.winner,
        loser=teams.loser,
        winner_score=score.winner,
        loser_score=score.loser,
        margin=score.margin,
        narrative="Vince Young rushed for 200 yards and scored with 19 seconds left.",
        tags="championship,comeback",
        venue="Rose Bowl",
        comeback_points=12,
    )
    result = SearchResult(key="2006-rose-bowl", distance=0.25, metadata=meta)
    assert result.similarity == pytest.approx(0.75)

    with pytest.raises(ValidationError):
        SearchResult(key="2006-rose-bowl", distance=2.5, metadata=meta)


def test_config_loads_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CFP_ORACLE_BUCKET_NAME", "custom-cfp-bucket")
    monkeypatch.setenv("CFP_ORACLE_INDEX_NAME", "custom-index")
    monkeypatch.setenv("CFP_ORACLE_EMBEDDING_DIMENSION", "512")
    monkeypatch.setenv("CFP_ORACLE_REGION", "us-west-2")

    cfg = S3VectorConfig()
    assert cfg.bucket_name == "custom-cfp-bucket"
    assert cfg.index_name == "custom-index"
    assert cfg.embedding_dimension == 512
    assert cfg.region == "us-west-2"

    with pytest.raises(ValidationError):
        cfg.bucket_name = "mutated"  # type: ignore[misc]


def test_config_falls_back_to_aws_region(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CFP_ORACLE_REGION", raising=False)
    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    cfg = S3VectorConfig()
    assert cfg.region == "eu-west-1"


def test_historical_dataset_integrity_all_36_games() -> None:
    assert len(HISTORICAL_GAMES) == 36
    games = load_historical_games()
    assert len(games) == 36

    ids = [g.id for g in games]
    assert len(set(ids)) == 36

    years = [g.year for g in games]
    assert min(years) == 2006
    assert max(years) == 2025

    cfp_era = [g for g in games if g.year >= 2015]
    bcs_era = [g for g in games if g.year < 2015]
    assert len(cfp_era) == 32
    assert len(bcs_era) == 4

    for game in games:
        assert isinstance(game, Game)
        assert game.score.margin > 0
        assert len(game.tags) >= 1
        text = create_game_text(game)
        assert game.teams.winner in text
        assert game.teams.loser in text
        meta = game_to_metadata(game)
        assert meta.margin == game.score.winner - game.score.loser
        assert len(meta.narrative) <= 500
        dumped = meta.model_dump(exclude_none=True)
        assert "None" not in dumped.values()


def test_narrative_serialization_and_formatting_contract() -> None:
    games = {g.id: g for g in load_historical_games()}

    miami_upset = games["2025-cotton-bowl"]
    miami_text = create_game_text(miami_upset)
    assert "Miami defeated Ohio State 24-14" in miami_text
    assert "This was a major upset." in miami_text
    assert "Key themes: upset, defensive battle, cfp quarterfinal, double digit seed" in miami_text

    rose_bowl_2018 = games["2018-rose-bowl-semi"]
    assert "The game went to overtime." in create_game_text(rose_bowl_2018)

    texas_usc = games["2006-rose-bowl"]
    assert "Featured a 12-point comeback." in create_game_text(texas_usc)

    formatted = format_result(
        SearchResult(
            key=miami_upset.id,
            distance=0.421,
            metadata=game_to_metadata(miami_upset),
        )
    )
    assert "Cotton Bowl - CFP Quarterfinal (2025)" in formatted
    assert "Miami 24 - Ohio State 14" in formatted
    assert "Similarity: 57.9%" in formatted
    assert "#upset" in formatted


def test_filtering_and_local_cosine_ranking() -> None:
    games = load_historical_games()

    shutouts = get_games_by_tag("shutout", games)
    assert {g.id for g in shutouts} == {
        "2017-fiesta-bowl-semi",
        "2016-cotton-bowl-semi",
        "2012-bcs-championship",
    }

    games_2024 = get_games_by_year(2024, games)
    assert len(games_2024) == 3

    texas_games = get_games_by_team("texas", games)
    assert {g.id for g in texas_games} == {"2024-sugar-bowl-semi", "2006-rose-bowl"}

    ot_semis = filter_games(games, tag="semifinal", overtime_only=True)
    assert {g.id for g in ot_semis} == {
        "2024-rose-bowl-semi",
        "2023-fiesta-bowl-semi",
        "2018-rose-bowl-semi",
    }

    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_distance([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)

    with pytest.raises(ValueError, match="dimension mismatch"):
        cosine_similarity([1.0, 2.0], [1.0])

    with pytest.raises(ValueError, match="zero-magnitude"):
        cosine_similarity([0.0, 0.0], [1.0, 0.0])

    subset = games[:3]
    indexed = {
        subset[0].id: [1.0, 0.0, 0.0],
        subset[1].id: [0.8, 0.6, 0.0],
        subset[2].id: [0.0, 1.0, 0.0],
    }
    ranked = rank_games_by_similarity([1.0, 0.0, 0.0], indexed, subset, top_k=2)
    assert [r.key for r in ranked] == [subset[0].id, subset[1].id]
    assert ranked[0].similarity == pytest.approx(1.0)
    assert ranked[1].similarity == pytest.approx(0.8)


def test_end_to_end_adapter_workflow_with_injected_fakes() -> None:
    cfg = S3VectorConfig(
        bucket_name="test-cfp-vectors",
        index_name="test-games",
        embedding_dimension=8,
        batch_size=15,
    )
    s3_fake = _FakeS3VectorsClient()
    bedrock_fake = _FakeBedrockClient(dimension=8)

    setup_infrastructure(config=cfg, s3vectors_client=s3_fake)
    # Second call exercises idempotent already-exists path
    setup_infrastructure(config=cfg, s3vectors_client=s3_fake)

    loaded_count = load_games(
        config=cfg,
        s3vectors_client=s3_fake,
        bedrock_client=bedrock_fake,
    )
    assert loaded_count == 36
    assert len(bedrock_fake.calls) == 36

    shutout_matches = find_similar_games(
        "defensive shutout blanking opponent",
        config=cfg,
        s3vectors_client=s3_fake,
        bedrock_client=bedrock_fake,
        top_k=3,
    )
    assert len(shutout_matches) == 3
    assert shutout_matches[0].metadata.shutout is True

    filtered_2025 = find_similar_games(
        "major upset",
        config=cfg,
        s3vectors_client=s3_fake,
        bedrock_client=bedrock_fake,
        top_k=5,
        year_filter=2025,
    )
    assert all(m.metadata.year == 2025 for m in filtered_2025)
    assert filtered_2025[0].key == "2025-cotton-bowl"

    cleanup(config=cfg, s3vectors_client=s3_fake)
    assert cfg.bucket_name not in s3_fake.buckets
