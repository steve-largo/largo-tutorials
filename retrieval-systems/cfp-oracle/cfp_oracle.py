"""CFP Oracle: semantic search over historical College Football Playoff games.

Uses Amazon S3 Vectors and Amazon Bedrock Titan Text Embeddings V2 for cloud
vector storage and retrieval, while keeping narrative serialization, metadata
projection, cosine similarity math, and filtering as pure, testable functions.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from games_data import load_historical_games
from models import Game, S3VectorConfig, SearchResult, VectorMetadata


def create_game_text(game: Game) -> str:
    """Build a structured narrative string for dense embedding generation."""
    parts = [
        f"{game.game} ({game.year})",
        f"{game.teams.winner} defeated {game.teams.loser} {game.score.winner}-{game.score.loser}",
        game.narrative,
    ]

    if game.key_stats:
        if game.key_stats.overtime:
            parts.append("The game went to overtime.")
        if game.key_stats.comeback:
            parts.append(f"Featured a {game.key_stats.comeback}-point comeback.")
        if game.key_stats.shutout:
            parts.append("This was a shutout victory.")
        if game.key_stats.upset:
            parts.append("This was a major upset.")

    if game.tags:
        readable_tags = [tag.replace("_", " ") for tag in game.tags]
        parts.append(f"Key themes: {', '.join(readable_tags)}")

    return " ".join(parts)


def game_to_metadata(game: Game) -> VectorMetadata:
    """Project a validated Game into the S3 Vectors metadata contract."""
    stats = game.key_stats
    return VectorMetadata(
        game_id=game.id,
        game_name=game.game,
        year=game.year,
        winner=game.teams.winner,
        loser=game.teams.loser,
        winner_score=game.score.winner,
        loser_score=game.score.loser,
        margin=game.score.margin,
        narrative=game.narrative[:500],
        tags=",".join(game.tags),
        venue=game.venue or "",
        overtime=stats.overtime if stats else None,
        comeback_points=stats.comeback if stats else None,
        shutout=stats.shutout if stats else None,
        upset=stats.upset if stats else None,
    )


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Compute cosine similarity in [-1.0, 1.0] between two equal-length vectors."""
    if not vec_a or not vec_b:
        raise ValueError("vectors must be non-empty")
    if len(vec_a) != len(vec_b):
        raise ValueError(
            f"vector dimension mismatch: {len(vec_a)} != {len(vec_b)}"
        )

    dot = sum(a * b for a, b in zip(vec_a, vec_b, strict=True))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError("cosine similarity is undefined for zero-magnitude vectors")

    raw = dot / (norm_a * norm_b)
    return max(-1.0, min(1.0, raw))


def cosine_distance(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Compute cosine distance (1 - cosine_similarity) clamped to [0.0, 2.0]."""
    return max(0.0, min(2.0, 1.0 - cosine_similarity(vec_a, vec_b)))


def rank_games_by_similarity(
    query_vector: Sequence[float],
    indexed_vectors: Mapping[str, Sequence[float]],
    games: Sequence[Game],
    *,
    top_k: int = 5,
    year_filter: int | None = None,
    tag_filter: str | None = None,
) -> list[SearchResult]:
    """Rank games locally by cosine distance with optional metadata pre-filtering."""
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    normalized_tag = tag_filter.strip().lower() if tag_filter else None
    candidates: list[SearchResult] = []

    for game in games:
        if year_filter is not None and game.year != year_filter:
            continue
        if normalized_tag and normalized_tag not in [t.lower() for t in game.tags]:
            continue
        vec = indexed_vectors.get(game.id)
        if vec is None:
            continue

        dist = cosine_distance(query_vector, vec)
        candidates.append(
            SearchResult(
                key=game.id,
                distance=dist,
                metadata=game_to_metadata(game),
            )
        )

    candidates.sort(key=lambda item: (item.distance, item.key))
    return candidates[:top_k]


def format_result(result: SearchResult) -> str:
    """Format a SearchResult into a human-readable terminal card."""
    meta = result.metadata
    lines = [
        "",
        "=" * 60,
        f"{meta.game_name} ({meta.year})",
        f"   {meta.winner} {meta.winner_score} - {meta.loser} {meta.loser_score}",
        f"   Similarity: {result.similarity:.1%}",
        f"\n   {meta.narrative}",
    ]

    if meta.tags:
        tags = [f"#{tag.replace('_', '-')}" for tag in meta.tags.split(",")[:5] if tag]
        if tags:
            lines.append(f"\n   {' '.join(tags)}")

    return "\n".join(lines)


def create_aws_clients(config: S3VectorConfig) -> tuple[Any, Any]:
    """Instantiate boto3 S3 Vectors and Bedrock Runtime clients lazily."""
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError(
            "boto3>=1.42.0 is required for live AWS S3 Vectors and Bedrock operations. "
            "Install dependencies via `pip install -r requirements.txt`."
        ) from exc

    s3vectors_client = boto3.client("s3vectors", region_name=config.region)
    bedrock_client = boto3.client("bedrock-runtime", region_name=config.region)
    return s3vectors_client, bedrock_client


def generate_embedding(
    text: str,
    *,
    config: S3VectorConfig,
    bedrock_client: Any,
) -> list[float]:
    """Generate a dense embedding vector via Amazon Bedrock Titan Text V2."""
    response = bedrock_client.invoke_model(
        modelId=config.embedding_model,
        body=json.dumps({"inputText": text}),
    )
    payload = json.loads(response["body"].read())
    return [float(x) for x in payload["embedding"]]


def _is_not_found_error(exc: Exception) -> bool:
    code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
    return exc.__class__.__name__ == "NotFoundException" or code in {
        "NotFoundException",
        "ResourceNotFoundException",
        "NoSuchBucket",
    }


def setup_infrastructure(*, config: S3VectorConfig, s3vectors_client: Any) -> None:
    """Create the S3 Vectors bucket and cosine index idempotently if missing."""
    print("Setting up S3 Vectors infrastructure...")
    try:
        s3vectors_client.get_vector_bucket(vectorBucketName=config.bucket_name)
        print(f"  Bucket '{config.bucket_name}' already exists")
    except Exception as exc:
        if not _is_not_found_error(exc):
            raise
        print(f"  Creating bucket '{config.bucket_name}'...")
        s3vectors_client.create_vector_bucket(vectorBucketName=config.bucket_name)

    try:
        s3vectors_client.get_index(
            vectorBucketName=config.bucket_name,
            indexName=config.index_name,
        )
        print(f"  Index '{config.index_name}' already exists")
    except Exception as exc:
        if not _is_not_found_error(exc):
            raise
        print(f"  Creating index '{config.index_name}'...")
        s3vectors_client.create_index(
            vectorBucketName=config.bucket_name,
            indexName=config.index_name,
            dataType="float32",
            dimension=config.embedding_dimension,
            distanceMetric="cosine",
        )


def load_games(
    *,
    config: S3VectorConfig,
    s3vectors_client: Any,
    bedrock_client: Any,
    games: Sequence[Game] | None = None,
) -> int:
    """Embed and upsert all historical games into S3 Vectors in batches."""
    catalog = list(games) if games is not None else load_historical_games()
    vectors: list[dict[str, Any]] = []

    for game in catalog:
        game_text = create_game_text(game)
        embedding = generate_embedding(
            game_text,
            config=config,
            bedrock_client=bedrock_client,
        )
        metadata = game_to_metadata(game)
        vectors.append(
            {
                "key": game.id,
                "data": {"float32": embedding},
                "metadata": metadata.model_dump(exclude_none=True),
            }
        )

    for start in range(0, len(vectors), config.batch_size):
        batch = vectors[start : start + config.batch_size]
        s3vectors_client.put_vectors(
            vectorBucketName=config.bucket_name,
            indexName=config.index_name,
            vectors=batch,
        )

    return len(vectors)


def find_similar_games(
    query: str,
    *,
    config: S3VectorConfig,
    s3vectors_client: Any,
    bedrock_client: Any,
    top_k: int = 5,
    year_filter: int | None = None,
) -> list[SearchResult]:
    """Embed a natural-language query and retrieve top-K matching games from S3 Vectors."""
    query_embedding = generate_embedding(
        query,
        config=config,
        bedrock_client=bedrock_client,
    )

    params: dict[str, Any] = {
        "vectorBucketName": config.bucket_name,
        "indexName": config.index_name,
        "queryVector": {"float32": query_embedding},
        "topK": top_k,
        "returnDistance": True,
        "returnMetadata": True,
    }
    if year_filter is not None:
        params["filter"] = {"year": year_filter}

    response = s3vectors_client.query_vectors(**params)
    return [
        SearchResult(
            key=item["key"],
            distance=float(item.get("distance", 0.0)),
            metadata=VectorMetadata.model_validate(item["metadata"]),
        )
        for item in response.get("vectors", [])
    ]


def cleanup(*, config: S3VectorConfig, s3vectors_client: Any) -> None:
    """Delete the S3 Vectors index and bucket created by setup_infrastructure."""
    try:
        s3vectors_client.delete_index(
            vectorBucketName=config.bucket_name,
            indexName=config.index_name,
        )
        print(f"Deleted index '{config.index_name}'")
    except Exception as exc:
        if not _is_not_found_error(exc):
            raise
        print(f"Index '{config.index_name}' already absent")

    try:
        s3vectors_client.delete_vector_bucket(vectorBucketName=config.bucket_name)
        print(f"Deleted bucket '{config.bucket_name}'")
    except Exception as exc:
        if not _is_not_found_error(exc):
            raise
        print(f"Bucket '{config.bucket_name}' already absent")


def run_demo_queries(
    *,
    config: S3VectorConfig,
    s3vectors_client: Any,
    bedrock_client: Any,
) -> None:
    """Execute five curated scenario queries against S3 Vectors."""
    demo_scenarios = [
        "Ohio State vs Texas in a CFP semifinal",
        "underdog double-digit seed upsets a top-2 team",
        "quarterback leads dramatic comeback to win championship",
        "defensive battle with a shutout",
        "overtime thriller in a semifinal",
    ]
    for query in demo_scenarios:
        print(f'\nQuery: "{query}"')
        results = find_similar_games(
            query,
            config=config,
            s3vectors_client=s3vectors_client,
            bedrock_client=bedrock_client,
            top_k=3,
        )
        for item in results:
            print(format_result(item))


def run_interactive_loop(
    *,
    config: S3VectorConfig,
    s3vectors_client: Any,
    bedrock_client: Any,
) -> None:
    """Prompt interactively for natural-language game scenarios."""
    print("CFP Oracle interactive mode (type 'quit' to exit)\n")
    while True:
        query = input("Describe a game scenario: ").strip()
        if query.lower() in {"quit", "exit", "q"}:
            break
        if not query:
            continue
        results = find_similar_games(
            query,
            config=config,
            s3vectors_client=s3vectors_client,
            bedrock_client=bedrock_client,
            top_k=3,
        )
        for item in results:
            print(format_result(item))
        print()


def inspect_dataset() -> None:
    """Print local dataset diagnostics without requiring AWS credentials."""
    games = load_historical_games()
    years = [g.year for g in games]
    cfp_count = sum(1 for g in games if g.year >= 2015)
    bcs_count = sum(1 for g in games if g.year < 2015)
    print(f"Validated games: {len(games)} ({min(years)}-{max(years)})")
    print(f"  CFP era (2015-2025): {cfp_count}")
    print(f"  BCS era (2006-2014): {bcs_count}")
    sample = games[0]
    print(f"\nSample serialized text ({sample.id}):\n{create_game_text(sample)}")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="CFP Oracle: semantic search over 36 historical CFP and BCS games"
    )
    parser.add_argument(
        "--inspect-dataset",
        action="store_true",
        help="Validate and summarize the 36-game dataset locally without AWS calls",
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Create S3 Vectors bucket/index and upload all 36 game embeddings",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run curated semantic search queries against S3 Vectors",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run interactive query prompt against S3 Vectors",
    )
    parser.add_argument("--query", type=str, help="Run a single natural-language query")
    parser.add_argument("--year", type=int, help="Optional year filter for --query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of matches to return")
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete the S3 Vectors index and bucket",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entrypoint."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.inspect_dataset:
        inspect_dataset()
        return

    if not any([args.setup, args.demo, args.interactive, args.query, args.cleanup]):
        parser.print_help()
        return

    config = S3VectorConfig()
    s3vectors_client, bedrock_client = create_aws_clients(config)

    if args.cleanup:
        cleanup(config=config, s3vectors_client=s3vectors_client)
        return

    if args.setup:
        setup_infrastructure(config=config, s3vectors_client=s3vectors_client)
        loaded = load_games(
            config=config,
            s3vectors_client=s3vectors_client,
            bedrock_client=bedrock_client,
        )
        print(f"Loaded {loaded} games into {config.bucket_name}/{config.index_name}")

    if args.demo:
        run_demo_queries(
            config=config,
            s3vectors_client=s3vectors_client,
            bedrock_client=bedrock_client,
        )
    elif args.query:
        results = find_similar_games(
            args.query,
            config=config,
            s3vectors_client=s3vectors_client,
            bedrock_client=bedrock_client,
            top_k=args.top_k,
            year_filter=args.year,
        )
        for item in results:
            print(format_result(item))
    elif args.interactive:
        run_interactive_loop(
            config=config,
            s3vectors_client=s3vectors_client,
            bedrock_client=bedrock_client,
        )


if __name__ == "__main__":
    main()
