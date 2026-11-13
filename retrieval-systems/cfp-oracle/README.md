# CFP Oracle: Semantic Search for College Football History

Companion codebase for [CFP Oracle: Semantic Search for College Football History](https://largo.dev/tutorials/retrieval-systems/cfp-oracle/) on [largo.dev](https://largo.dev/).

This package indexes 36 historical College Football Playoff (2015-2025, 32 games) and BCS Championship (2006-2014, 4 games) matchups as 1,024-dimensional embeddings using Amazon Bedrock Titan Text Embeddings V2 (`amazon.titan-embed-text-v2:0`) and queries them by narrative similarity with Amazon S3 Vectors.

## Project Layout

- `models.py`: Immutable Pydantic v2 data schemas (`Teams`, `Score`, `Seeds`, `KeyStats`, `Game`, `VectorMetadata`, `SearchResult`) and `S3VectorConfig` (`BaseSettings` loaded from `CFP_ORACLE_*` environment variables).
- `games_data.py`: Self-contained 36-game dataset (`HISTORICAL_GAMES`) plus typed filter helpers (`load_historical_games`, `filter_games`, `get_games_by_tag`, `get_games_by_year`, `get_games_by_team`).
- `cfp_oracle.py`: Narrative serializer (`create_game_text`), metadata projector (`game_to_metadata`), local cosine distance/ranking helpers (`cosine_similarity`, `rank_games_by_similarity`), and client-injected S3 Vectors / Bedrock adapters.
- `tests/test_cfp_oracle.py`: Unit and contract tests covering Pydantic validation, 36-game dataset integrity, local cosine ranking/filtering, and mocked S3 Vectors / Bedrock client workflows without requiring live AWS credentials.

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Run linter and offline test suite (no AWS credentials required)
.venv/bin/ruff check .
.venv/bin/pytest -v

# Inspect the 36-game dataset and serialized embedding text locally
.venv/bin/python cfp_oracle.py --inspect-dataset
```

## Running Against Amazon S3 Vectors & Bedrock

Ensure AWS credentials with `s3vectors:*` and `bedrock:InvokeModel` permissions are configured in your shell environment:

```bash
# Override optional infrastructure settings via environment variables
export CFP_ORACLE_BUCKET_NAME="cfp-oracle-vectors"
export CFP_ORACLE_INDEX_NAME="historical-games"
export CFP_ORACLE_EMBEDDING_MODEL="amazon.titan-embed-text-v2:0"
export CFP_ORACLE_EMBEDDING_DIMENSION=1024
export CFP_ORACLE_REGION="us-east-1"

# Create bucket/index and upload all 36 game embeddings
.venv/bin/python cfp_oracle.py --setup

# Run curated scenario queries or a custom natural-language query
.venv/bin/python cfp_oracle.py --demo
.venv/bin/python cfp_oracle.py --query "quarterback leads dramatic comeback to win championship"

# Tear down S3 Vectors index and bucket when finished
.venv/bin/python cfp_oracle.py --cleanup
```
