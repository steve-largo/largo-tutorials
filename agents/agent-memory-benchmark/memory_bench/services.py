"""ADK BaseMemoryService implementations: sliding window, SQLite vector RAG, consolidated memory, and Vertex AI Memory Bank wiring."""

from __future__ import annotations

import datetime as dt
import json
import uuid
from collections import deque
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager

import aiosqlite
from google.adk.agents import Context, LlmAgent
from google.adk.events.event import Event
from google.adk.memory import BaseMemoryService, VertexAiMemoryBankService
from google.adk.memory.base_memory_service import SearchMemoryResponse
from google.adk.memory.memory_entry import MemoryEntry
from google.adk.models.base_llm import BaseLlm
from google.adk.sessions import Session, VertexAiSessionService
from google.adk.sessions.sqlite_session_service import SqliteSessionService
from google.adk.tools.load_memory_tool import LoadMemoryTool
from google.adk.tools.preload_memory_tool import PreloadMemoryTool
from google.genai import types
from typing_extensions import override

from memory_bench.config import MemoryBenchSettings
from memory_bench.embeddings import (
    DeterministicCoastalEmbedder,
    EmbeddingAdapter,
    cosine_similarity,
)
from memory_bench.extractor import FactExtractor, HeuristicCoastalFactExtractor
from memory_bench.schemas import (
    CanonicalMemory,
    ReconcileAction,
    SessionTurn,
    validate_tenant_scope,
)


def strip_thought_parts(content: types.Content | None) -> types.Content | None:
    """Strip model reasoning thought parts before memory ingestion, matching VertexAiMemoryBankService."""
    if not content or not content.parts:
        return content
    if not any(part.thought for part in content.parts):
        return content
    kept_parts = [part for part in content.parts if not part.thought]
    if not kept_parts:
        return None
    return content.model_copy(update={"parts": kept_parts})


def extract_plain_text(content: types.Content | None) -> str:
    """Extract concatenated non-thought text parts from ADK Content."""
    cleaned = strip_thought_parts(content)
    if not cleaned or not cleaned.parts:
        return ""
    chunks = [part.text.strip() for part in cleaned.parts if part.text and part.text.strip()]
    return " ".join(chunks)


class SlidingWindowMemoryService(BaseMemoryService):
    """Retains only the most recent `window_sessions` sessions per (app_name, user_id)."""

    def __init__(self, window_sessions: int = 2) -> None:
        if window_sessions < 1:
            raise ValueError("window_sessions must be >= 1")
        self._window_sessions = window_sessions
        self._recent_by_scope: dict[tuple[str, str], deque[tuple[str, list[MemoryEntry]]]] = {}

    @override
    async def add_session_to_memory(self, session: Session) -> None:
        scope = validate_tenant_scope(session.app_name, session.user_id)
        await self.add_events_to_memory(
            app_name=scope[0],
            user_id=scope[1],
            events=session.events,
            session_id=session.id,
        )

    @override
    async def add_events_to_memory(
        self,
        *,
        app_name: str,
        user_id: str,
        events: Sequence[Event],
        session_id: str | None = None,
        custom_metadata: Mapping[str, object] | None = None,
    ) -> None:
        _ = custom_metadata
        scope = validate_tenant_scope(app_name, user_id)
        entries: list[MemoryEntry] = []
        for event in events:
            text = extract_plain_text(event.content)
            if not text:
                continue
            ts = (
                dt.datetime.fromtimestamp(event.timestamp, tz=dt.UTC).isoformat()
                if event.timestamp
                else None
            )
            role = event.content.role if event.content and event.content.role else "user"
            entries.append(
                MemoryEntry(
                    id=event.id,
                    author=event.author or role,
                    content=types.Content(role=role, parts=[types.Part(text=text)]),
                    timestamp=ts,
                )
            )
        if not entries:
            return
        bucket = self._recent_by_scope.setdefault(
            scope, deque(maxlen=self._window_sessions)
        )
        sid = session_id or str(uuid.uuid4())
        # Replace if same session_id is re-ingested, otherwise append and evict oldest beyond maxlen
        filtered = deque((k, v) for k, v in bucket if k != sid)
        while len(filtered) >= self._window_sessions:
            filtered.popleft()
        filtered.append((sid, entries))
        self._recent_by_scope[scope] = deque(filtered, maxlen=self._window_sessions)

    @override
    async def search_memory(
        self, *, app_name: str, user_id: str, query: str
    ) -> SearchMemoryResponse:
        _ = query
        scope = validate_tenant_scope(app_name, user_id)
        bucket = self._recent_by_scope.get(scope)
        if not bucket:
            return SearchMemoryResponse(memories=[])
        flattened: list[MemoryEntry] = []
        for _, session_entries in bucket:
            flattened.extend(session_entries)
        return SearchMemoryResponse(memories=flattened)


class SqliteVectorMemoryService(BaseMemoryService):
    """Append-only SQLite semantic vector store demonstrating raw-chunk RAG trade-offs."""

    _SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS vector_chunks (
        chunk_id TEXT PRIMARY KEY,
        app_name TEXT NOT NULL,
        user_id TEXT NOT NULL,
        session_id TEXT,
        author TEXT NOT NULL,
        role TEXT NOT NULL,
        chunk_text TEXT NOT NULL,
        embedding_json TEXT NOT NULL,
        recorded_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_vector_chunks_scope
        ON vector_chunks (app_name, user_id);
    """

    def __init__(
        self,
        *,
        sqlite_path: str = ":memory:",
        embedder: EmbeddingAdapter | None = None,
        top_k: int = 4,
        min_similarity: float = 0.18,
        index_model_turns: bool = True,
    ) -> None:
        self._sqlite_path = sqlite_path
        self._embedder = embedder or DeterministicCoastalEmbedder()
        self._top_k = top_k
        self._min_similarity = min_similarity
        self._index_model_turns = index_model_turns
        self._shared_mem_conn: aiosqlite.Connection | None = None
        self._initialized = False

    @asynccontextmanager
    async def _connection(self) -> AsyncIterator[aiosqlite.Connection]:
        if self._sqlite_path == ":memory:":
            if self._shared_mem_conn is None:
                self._shared_mem_conn = await aiosqlite.connect(":memory:")
            conn = self._shared_mem_conn
            if not self._initialized:
                await conn.executescript(self._SCHEMA_SQL)
                await conn.commit()
                self._initialized = True
            yield conn
        else:
            async with aiosqlite.connect(self._sqlite_path) as conn:
                if not self._initialized:
                    await conn.execute("PRAGMA journal_mode=WAL")
                    await conn.executescript(self._SCHEMA_SQL)
                    await conn.commit()
                    self._initialized = True
                yield conn

    async def close(self) -> None:
        if self._shared_mem_conn is not None:
            await self._shared_mem_conn.close()
            self._shared_mem_conn = None
            self._initialized = False

    @override
    async def add_session_to_memory(self, session: Session) -> None:
        scope = validate_tenant_scope(session.app_name, session.user_id)
        await self.add_events_to_memory(
            app_name=scope[0],
            user_id=scope[1],
            events=session.events,
            session_id=session.id,
        )

    @override
    async def add_events_to_memory(
        self,
        *,
        app_name: str,
        user_id: str,
        events: Sequence[Event],
        session_id: str | None = None,
        custom_metadata: Mapping[str, object] | None = None,
    ) -> None:
        _ = custom_metadata
        scope_app, scope_user = validate_tenant_scope(app_name, user_id)
        rows_to_embed: list[tuple[str, str, str, str, str]] = []
        for event in events:
            role = event.content.role if event.content and event.content.role else "user"
            if role != "user" and not self._index_model_turns:
                continue
            text = extract_plain_text(event.content)
            if not text:
                continue
            ts = (
                dt.datetime.fromtimestamp(event.timestamp, tz=dt.UTC).isoformat()
                if event.timestamp
                else dt.datetime.now(dt.UTC).isoformat()
            )
            chunk_id = event.id or str(uuid.uuid4())
            author = event.author or role
            rows_to_embed.append((chunk_id, author, role, text, ts))

        if not rows_to_embed:
            return

        vectors = await self._embedder.embed_texts([row[3] for row in rows_to_embed])
        async with self._connection() as conn:
            await conn.executemany(
                """
                INSERT OR REPLACE INTO vector_chunks
                    (chunk_id, app_name, user_id, session_id, author, role, chunk_text, embedding_json, recorded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk_id,
                        scope_app,
                        scope_user,
                        session_id,
                        author,
                        role,
                        text,
                        json.dumps(vec),
                        ts,
                    )
                    for (chunk_id, author, role, text, ts), vec in zip(
                        rows_to_embed, vectors, strict=True
                    )
                ],
            )
            await conn.commit()

    @override
    async def search_memory(
        self, *, app_name: str, user_id: str, query: str
    ) -> SearchMemoryResponse:
        scope_app, scope_user = validate_tenant_scope(app_name, user_id)
        if not query.strip():
            return SearchMemoryResponse(memories=[])
        query_vec = (await self._embedder.embed_texts([query]))[0]

        async with self._connection() as conn:
            cursor = await conn.execute(
                """
                SELECT chunk_id, author, role, chunk_text, embedding_json, recorded_at
                  FROM vector_chunks
                 WHERE app_name = ? AND user_id = ?
                """,
                (scope_app, scope_user),
            )
            rows = await cursor.fetchall()

        scored: list[tuple[float, MemoryEntry]] = []
        for chunk_id, author, role, chunk_text, emb_json, recorded_at in rows:
            emb = json.loads(emb_json)
            sim = cosine_similarity(query_vec, emb)
            if sim >= self._min_similarity:
                scored.append(
                    (
                        sim,
                        MemoryEntry(
                            id=chunk_id,
                            author=author,
                            content=types.Content(role=role, parts=[types.Part(text=chunk_text)]),
                            timestamp=recorded_at,
                            custom_metadata={"similarity": round(sim, 4)},
                        ),
                    )
                )

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return SearchMemoryResponse(memories=[entry for _, entry in scored[: self._top_k]])


class ConsolidatedMemoryService(BaseMemoryService):
    """Fact extraction, slot reconciliation, and profile lookup modeled on Vertex AI Memory Bank."""

    _SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS canonical_memories (
        memory_id TEXT PRIMARY KEY,
        app_name TEXT NOT NULL,
        user_id TEXT NOT NULL,
        category TEXT NOT NULL,
        slot_key TEXT NOT NULL,
        fact_text TEXT NOT NULL,
        embedding_json TEXT NOT NULL,
        revision INTEGER NOT NULL,
        source_session_id TEXT,
        updated_at TEXT NOT NULL,
        active INTEGER NOT NULL DEFAULT 1,
        superseded_by TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_canonical_active_scope
        ON canonical_memories (app_name, user_id, active, slot_key);
    """

    def __init__(
        self,
        *,
        sqlite_path: str = ":memory:",
        extractor: FactExtractor | None = None,
        embedder: EmbeddingAdapter | None = None,
        top_k: int = 4,
        min_similarity: float = 0.16,
    ) -> None:
        self._sqlite_path = sqlite_path
        self._extractor = extractor or HeuristicCoastalFactExtractor()
        self._embedder = embedder or DeterministicCoastalEmbedder()
        self._top_k = top_k
        self._min_similarity = min_similarity
        self._shared_mem_conn: aiosqlite.Connection | None = None
        self._initialized = False
        self.reconciliation_counts: dict[ReconcileAction, int] = {
            "ADD": 0,
            "UPDATE": 0,
            "NOOP": 0,
            "IGNORE_EPHEMERAL": 0,
        }

    @asynccontextmanager
    async def _connection(self) -> AsyncIterator[aiosqlite.Connection]:
        if self._sqlite_path == ":memory:":
            if self._shared_mem_conn is None:
                self._shared_mem_conn = await aiosqlite.connect(":memory:")
            conn = self._shared_mem_conn
            if not self._initialized:
                await conn.executescript(self._SCHEMA_SQL)
                await conn.commit()
                self._initialized = True
            yield conn
        else:
            async with aiosqlite.connect(self._sqlite_path) as conn:
                if not self._initialized:
                    await conn.execute("PRAGMA journal_mode=WAL")
                    await conn.executescript(self._SCHEMA_SQL)
                    await conn.commit()
                    self._initialized = True
                yield conn

    async def close(self) -> None:
        if self._shared_mem_conn is not None:
            await self._shared_mem_conn.close()
            self._shared_mem_conn = None
            self._initialized = False

    @override
    async def add_session_to_memory(self, session: Session) -> None:
        scope = validate_tenant_scope(session.app_name, session.user_id)
        await self.add_events_to_memory(
            app_name=scope[0],
            user_id=scope[1],
            events=session.events,
            session_id=session.id,
        )

    @override
    async def add_events_to_memory(
        self,
        *,
        app_name: str,
        user_id: str,
        events: Sequence[Event],
        session_id: str | None = None,
        custom_metadata: Mapping[str, object] | None = None,
    ) -> None:
        scope_app, scope_user = validate_tenant_scope(app_name, user_id)
        allowed_topics: set[str] | None = None
        if custom_metadata and "allowed_topics" in custom_metadata:
            raw_topics = custom_metadata["allowed_topics"]
            if isinstance(raw_topics, Sequence) and not isinstance(raw_topics, (str, bytes)):
                allowed_topics = {str(t) for t in raw_topics}

        turns: list[SessionTurn] = []
        latest_ts = dt.datetime.now(dt.UTC).isoformat()
        for event in events:
            text = extract_plain_text(event.content)
            if not text:
                continue
            role_raw = event.content.role if event.content and event.content.role else "user"
            role = "user" if role_raw == "user" else "model"
            turns.append(SessionTurn(role=role, text=text))
            if event.timestamp:
                latest_ts = dt.datetime.fromtimestamp(
                    event.timestamp, tz=dt.UTC
                ).isoformat()

        candidates = await self._extractor.extract_facts(turns)
        durable = []
        for cand in candidates:
            if cand.is_ephemeral or cand.category == "ephemeral_telemetry":
                self.reconciliation_counts["IGNORE_EPHEMERAL"] += 1
                continue
            if allowed_topics is not None and cand.category not in allowed_topics:
                continue
            durable.append(cand)

        if not durable:
            return

        vectors = await self._embedder.embed_texts([c.fact_text for c in durable])

        async with self._connection() as conn:
            for cand, vec in zip(durable, vectors, strict=True):
                cursor = await conn.execute(
                    """
                    SELECT memory_id, fact_text, revision
                      FROM canonical_memories
                     WHERE app_name = ? AND user_id = ? AND slot_key = ? AND active = 1
                     LIMIT 1
                    """,
                    (scope_app, scope_user, cand.slot_key),
                )
                existing = await cursor.fetchone()

                if existing is None:
                    new_id = f"mem-{uuid.uuid4().hex[:12]}"
                    await conn.execute(
                        """
                        INSERT INTO canonical_memories
                            (memory_id, app_name, user_id, category, slot_key, fact_text,
                             embedding_json, revision, source_session_id, updated_at, active, superseded_by)
                        VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?, 1, NULL)
                        """,
                        (
                            new_id,
                            scope_app,
                            scope_user,
                            cand.category,
                            cand.slot_key,
                            cand.fact_text,
                            json.dumps(vec),
                            session_id,
                            latest_ts,
                        ),
                    )
                    self.reconciliation_counts["ADD"] += 1
                    continue

                prev_id, prev_text, prev_rev = existing
                if prev_text.strip().lower() == cand.fact_text.strip().lower():
                    self.reconciliation_counts["NOOP"] += 1
                    continue

                new_id = f"mem-{uuid.uuid4().hex[:12]}"
                await conn.execute(
                    """
                    UPDATE canonical_memories
                       SET active = 0, superseded_by = ?
                     WHERE memory_id = ? AND app_name = ? AND user_id = ?
                    """,
                    (new_id, prev_id, scope_app, scope_user),
                )
                await conn.execute(
                    """
                    INSERT INTO canonical_memories
                        (memory_id, app_name, user_id, category, slot_key, fact_text,
                         embedding_json, revision, source_session_id, updated_at, active, superseded_by)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, NULL)
                    """,
                    (
                        new_id,
                        scope_app,
                        scope_user,
                        cand.category,
                        cand.slot_key,
                        cand.fact_text,
                        json.dumps(vec),
                        int(prev_rev) + 1,
                        session_id,
                        latest_ts,
                    ),
                )
                self.reconciliation_counts["UPDATE"] += 1

            await conn.commit()

    @override
    async def search_memory(
        self, *, app_name: str, user_id: str, query: str
    ) -> SearchMemoryResponse:
        scope_app, scope_user = validate_tenant_scope(app_name, user_id)
        if not query.strip():
            return SearchMemoryResponse(memories=[])
        query_vec = (await self._embedder.embed_texts([query]))[0]

        async with self._connection() as conn:
            cursor = await conn.execute(
                """
                SELECT memory_id, category, slot_key, fact_text, embedding_json, revision, updated_at
                  FROM canonical_memories
                 WHERE app_name = ? AND user_id = ? AND active = 1
                """,
                (scope_app, scope_user),
            )
            rows = await cursor.fetchall()

        scored: list[tuple[float, MemoryEntry]] = []
        for mem_id, category, slot_key, fact_text, emb_json, revision, updated_at in rows:
            emb = json.loads(emb_json)
            sim = cosine_similarity(query_vec, emb)
            if sim >= self._min_similarity:
                scored.append(
                    (
                        sim,
                        MemoryEntry(
                            id=mem_id,
                            author="user",
                            content=types.Content(
                                role="user", parts=[types.Part(text=fact_text)]
                            ),
                            timestamp=updated_at,
                            custom_metadata={
                                "category": category,
                                "slot_key": slot_key,
                                "revision": revision,
                                "similarity": round(sim, 4),
                            },
                        ),
                    )
                )

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return SearchMemoryResponse(memories=[entry for _, entry in scored[: self._top_k]])

    async def retrieve_profile(
        self, *, app_name: str, user_id: str
    ) -> list[CanonicalMemory]:
        """Deterministic scope-keyed profile lookup analogous to VertexAiMemoryBankService.retrieve_profiles."""
        scope_app, scope_user = validate_tenant_scope(app_name, user_id)
        async with self._connection() as conn:
            cursor = await conn.execute(
                """
                SELECT memory_id, app_name, user_id, category, slot_key, fact_text,
                       revision, source_session_id, updated_at, active, superseded_by
                  FROM canonical_memories
                 WHERE app_name = ? AND user_id = ? AND active = 1
                 ORDER BY slot_key ASC
                """,
                (scope_app, scope_user),
            )
            rows = await cursor.fetchall()
        return [
            CanonicalMemory(
                memory_id=r[0],
                app_name=r[1],
                user_id=r[2],
                category=r[3],
                slot_key=r[4],
                fact_text=r[5],
                revision=r[6],
                source_session_id=r[7],
                updated_at=r[8],
                active=bool(r[9]),
                superseded_by=r[10],
            )
            for r in rows
        ]

    async def list_revision_history(
        self, *, app_name: str, user_id: str
    ) -> list[CanonicalMemory]:
        """Inspect both active and superseded revisions for auditability and testing."""
        scope_app, scope_user = validate_tenant_scope(app_name, user_id)
        async with self._connection() as conn:
            cursor = await conn.execute(
                """
                SELECT memory_id, app_name, user_id, category, slot_key, fact_text,
                       revision, source_session_id, updated_at, active, superseded_by
                  FROM canonical_memories
                 WHERE app_name = ? AND user_id = ?
                 ORDER BY slot_key ASC, revision ASC
                """,
                (scope_app, scope_user),
            )
            rows = await cursor.fetchall()
        return [
            CanonicalMemory(
                memory_id=r[0],
                app_name=r[1],
                user_id=r[2],
                category=r[3],
                slot_key=r[4],
                fact_text=r[5],
                revision=r[6],
                source_session_id=r[7],
                updated_at=r[8],
                active=bool(r[9]),
                superseded_by=r[10],
            )
            for r in rows
        ]


def build_production_services(
    settings: MemoryBenchSettings,
) -> tuple[SqliteSessionService | VertexAiSessionService, BaseMemoryService]:
    """Return managed Vertex AI Agent Engine services when AGENT_ENGINE_ID is configured, else SQLite."""
    if settings.agent_engine_id:
        session_service = VertexAiSessionService(
            project=settings.google_cloud_project,
            location=settings.google_cloud_location,
            agent_engine_id=settings.agent_engine_id,
        )
        memory_service = VertexAiMemoryBankService(
            project=settings.google_cloud_project,
            location=settings.google_cloud_location,
            agent_engine_id=settings.agent_engine_id,
        )
        return session_service, memory_service

    db_path = "captain_sessions.db" if settings.sqlite_path == ":memory:" else settings.sqlite_path
    return (
        SqliteSessionService(db_path=db_path),
        ConsolidatedMemoryService(sqlite_path=settings.sqlite_path),
    )


async def auto_save_session_to_memory(
    callback_context: Context,
) -> types.Content | None:
    """ADK after_agent_callback that triggers background memory extraction after each turn."""
    await callback_context.add_session_to_memory()
    return None


def build_memory_aware_captain_agent(
    *,
    model: str | BaseLlm = "gemini-2.5-flash",
    retrieval_mode: str = "preload",
    auto_ingest_on_turn_end: bool = True,
) -> LlmAgent:
    """Build an ADK LlmAgent wired with PreloadMemoryTool (0 extra LLM turns) or LoadMemoryTool."""
    if retrieval_mode == "preload":
        tools = [PreloadMemoryTool()]
    elif retrieval_mode == "on_demand":
        tools = [LoadMemoryTool()]
    else:
        raise ValueError(f"Unsupported retrieval_mode: {retrieval_mode!r}")

    return LlmAgent(
        name="KeysMemoryCaptain",
        model=model,
        instruction=(
            "You are a Florida Keys marine captain assistant. Ground vessel clearance, "
            "departure marina, crew sea-state comfort rules, and tackle recommendations "
            "strictly in recalled user memories and current telemetry."
        ),
        tools=tools,
        after_agent_callback=auto_save_session_to_memory if auto_ingest_on_turn_end else None,
    )
