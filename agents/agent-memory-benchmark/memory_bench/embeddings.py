"""Pluggable embedding adapters (Vertex AI gemini-embedding-001 and deterministic local vectors)."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Protocol

from google import genai

from memory_bench.config import MemoryBenchSettings

_WORD_RE = re.compile(r"[a-z0-9']+")

# 16 domain-grounded semantic concept groups + 80 hashed n-gram dimensions = 96-dim unit vector
_CONCEPT_CLUSTERS: tuple[frozenset[str], ...] = (
    frozenset({"boat", "vessel", "skiff", "contender", "hull", "craft", "bay"}),
    frozenset({"draft", "draws", "drawing", "inches", "inch", "clear", "clearance", "shallow", "flat"}),
    frozenset({"marina", "slip", "ramp", "launch", "launching", "departing", "leaving", "tavernier", "islamorada", "bud"}),
    frozenset({"sarah", "wife", "crew", "family", "passenger", "taking", "rules", "fit"}),
    frozenset({"seasick", "relief", "bands", "comfortable", "comfort", "turn", "safety", "limits", "limit"}),
    frozenset({"chop", "swell", "seas", "wave", "waves", "period", "seconds", "second", "reef", "alligator"}),
    frozenset({"tarpon", "tackle", "hook", "hooks", "circle", "barbless", "channel", "bridge", "rig", "setup", "bag"}),
    frozenset({"tide", "tides", "low", "high", "mllw", "falling", "outgoing", "snake", "creek"}),
    frozenset({"wind", "winds", "gusts", "advisory", "tropical", "forecast", "nws", "weather", "mph", "knots"}),
    frozenset({"specs", "specifications", "profile", "remind", "summary", "setup"}),
    frozenset({"2", "0", "barbless", "leader", "fluorocarbon"}),
    frozenset({"9", "12", "16", "18ft", "25ft", "foot"}),
    frozenset({"2", "5", "3", "7", "8", "short"}),
    frozenset({"may", "june", "july", "august", "september", "october", "november", "yesterday", "tomorrow"}),
    frozenset({"stayed", "dock", "blown", "small", "advisories"}),
    frozenset({"upgrade", "sold", "picked", "moved", "switched", "updated"}),
)


def _l2_normalize(values: list[float]) -> list[float]:
    norm = math.sqrt(math.fsum(v * v for v in values))
    if norm <= 1e-12:
        return [0.0 for _ in values]
    return [v / norm for v in values]


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Compute exact cosine similarity between two equal-length vectors."""
    if len(vec_a) != len(vec_b) or not vec_a:
        return 0.0
    norm_a = math.sqrt(math.fsum(x * x for x in vec_a))
    norm_b = math.sqrt(math.fsum(y * y for y in vec_b))
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return 0.0
    return math.fsum(x * y for x, y in zip(vec_a, vec_b, strict=True)) / (norm_a * norm_b)


class EmbeddingAdapter(Protocol):
    """Protocol for async text embedding backends."""

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]: ...


class DeterministicCoastalEmbedder:
    """Fast, keyless 96-dim embedder for reproducible offline unit tests and CI."""

    dim: int = 96
    _hash_dims: int = 80

    def _encode_one(self, text: str) -> list[float]:
        lowered = text.lower()
        tokens = _WORD_RE.findall(lowered)
        token_set = set(tokens)
        vec = [0.0] * self.dim

        # Concept dimensions (0..15) capture semantic equivalence across vocabulary shifts
        for idx, cluster in enumerate(_CONCEPT_CLUSTERS):
            hits = len(token_set & cluster)
            if hits:
                vec[idx] = 1.6 * math.log1p(hits)

        # Hashed word unigrams + character 4-grams (16..95) preserve lexical specificity
        features = [f"w:{tok}" for tok in tokens]
        compact = " ".join(tokens)
        features.extend(f"c:{compact[i : i + 4]}" for i in range(max(0, len(compact) - 3)))

        for feat in features:
            digest = hashlib.sha256(feat.encode("utf-8")).digest()
            bucket = 16 + (int.from_bytes(digest[:2], "big") % self._hash_dims)
            sign = 1.0 if (digest[2] & 1) == 0 else -1.0
            vec[bucket] += sign * 0.35

        return _l2_normalize(vec)

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._encode_one(t) for t in texts]


class VertexGenAiEmbedder:
    """Production embedding adapter backed by Vertex AI (gemini-embedding-001)."""

    def __init__(self, settings: MemoryBenchSettings) -> None:
        self._model = settings.embedding_model
        self._client = genai.Client(
            vertexai=settings.use_vertexai,
            project=settings.google_cloud_project,
            location=settings.google_cloud_location,
        )

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        response = await self._client.aio.models.embed_content(
            model=self._model,
            contents=list(texts),
        )
        vectors: list[list[float]] = []
        for item in response.embeddings or ():
            raw = list(item.values or ())
            vectors.append(_l2_normalize(raw))
        return vectors
