"""MemorySystem: the async orchestrator over the index, the SQL stores, and working memory.

The stores are synchronous SQLAlchemy; every store call here runs in a worker thread
(``asyncio.to_thread``) so the caller's event loop — a streaming assistant turn —
never blocks on database I/O.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from memory_core.types.smk_types import MemoryKind, ToolFlag, TopicBucket

from .models import AssistantMemoryTrace, MemoryCandidate, MemoryTrace, RecallResult

if TYPE_CHECKING:
    from collections.abc import Iterable

    from memory_core.core.interfaces import Embedder, MemoryIndex
    from memory_core.indexing.assistant_index import AssistantMemoryHit, AssistantMemoryIndex
    from memory_core.storage.ltm_store import LongTermStore
    from memory_core.storage.stm_store import ShortTermStore
    from memory_core.storage.working_mem import SqlWorkingMemory


class MemorySystem:
    """Coordinate memory ingest/recall workflows across all storage layers."""

    def __init__(
        self,
        *,
        memory_index: MemoryIndex,
        working_mem: SqlWorkingMemory,
        ltm_store: LongTermStore,
        embedder: Embedder | None = None,
        assistant_index: AssistantMemoryIndex | None = None,
        stm_store: ShortTermStore | None = None,
        stm_ttl_seconds: int = 900,
    ) -> None:
        self.memory_index = memory_index
        self.working_mem = working_mem
        self.ltm_store = ltm_store
        self.embedder = embedder
        self.assistant_index = assistant_index
        self.stm_store = stm_store
        self._stm_ttl_seconds = stm_ttl_seconds

    # --- Startup -----------------------------------------------------------

    async def hydrate(self, user_id: str) -> int:
        """Load a user's persisted traces from LTM into the in-memory index.

        Returns the number of traces ingested (traces persisted without an
        embedding are skipped — they cannot be vector-searched).
        """

        def _load() -> int:
            loaded = 0
            for trace, embedding in self.ltm_store.fetch_traces_for_user(user_id):
                if embedding:
                    self.memory_index.ingest(trace, embedding)
                    loaded += 1
            return loaded

        return await asyncio.to_thread(_load)

    # --- User memory -------------------------------------------------------

    async def remember(
        self,
        *,
        user_id: str,
        summary: str,
        importance: float,
        tags: list[str],
        content: str = "",
        embedding: Iterable[float] | None = None,
        extra: dict[str, Any] | None = None,
        also_working_mem: bool = True,
    ) -> MemoryTrace:
        """Persist a new trace across STM, LTM, the vector index, and working memory.

        Pass ``embedding`` explicitly, or construct the system with an ``Embedder``
        and it is computed here from the summary + content. ``extra`` carries
        provenance (source surface, session id, …) and round-trips through LTM.
        """
        trace = MemoryTrace(
            trace_uid=str(uuid.uuid4()),
            user_id=user_id,
            content=content,
            summary=summary,
            importance=importance,
            created_at=datetime.now(UTC),
            access_count=0,
            tags=set(tags),
            extra=dict(extra) if extra else {},
        )

        emb_list = await self._resolve_embedding(embedding, f"{summary}\n{content}".strip())

        def _persist() -> None:
            if self.stm_store is not None:
                expires_at = trace.created_at + timedelta(seconds=self._stm_ttl_seconds)
                self.stm_store.insert_trace(trace, expires_at=expires_at)
            self.ltm_store.upsert_trace(trace, embedding=emb_list)
            self.memory_index.ingest(trace, emb_list)
            if also_working_mem:
                self.working_mem.add_event(
                    user_id=user_id,
                    payload={
                        "kind": "ltm_trace",
                        "trace_uid": trace.trace_uid,
                        "summary": trace.summary,
                        "importance": trace.importance,
                        "tags": sorted(trace.tags),
                    },
                )

        await asyncio.to_thread(_persist)
        return trace

    async def recall(
        self,
        *,
        user_id: str,
        query_text: str,
        limit: int,
        tags: list[str] | None = None,
        query_embedding: Iterable[float] | None = None,
        include_working_mem: bool = True,
    ) -> RecallResult:
        """Recall candidates from the index plus recent STM traces and WM events."""
        emb_list = await self._resolve_embedding(query_embedding, query_text)

        def _recall() -> RecallResult:
            candidates: list[MemoryCandidate] = self.memory_index.search(
                user_id=user_id,
                text=query_text,
                tags=tags or [],
                limit=limit,
                query_embedding=emb_list,
            )
            for candidate in candidates:
                self.memory_index.mark_accessed(candidate.trace_uid)

            wm_events = (
                self.working_mem.get_recent(user_id=user_id, limit=20)
                if include_working_mem
                else []
            )
            stm_traces = (
                self.stm_store.fetch_recent_for_user(user_id=user_id, limit=limit)
                if self.stm_store is not None
                else []
            )
            return RecallResult(
                ltm_candidates=candidates,
                wm_events=wm_events,
                stm_traces=stm_traces,
            )

        return await asyncio.to_thread(_recall)

    async def forget(self, trace_uid: str) -> bool:
        """Remove a trace from the index and LTM. Returns True when LTM had the row."""

        def _forget() -> bool:
            self.memory_index.remove(trace_uid)
            return self.ltm_store.delete_trace(trace_uid)

        return await asyncio.to_thread(_forget)

    async def record_event(self, *, user_id: str, payload: dict[str, object]) -> None:
        """Append a raw working-memory event (a turn, a tool use, a session marker)."""
        await asyncio.to_thread(self.working_mem.add_event, user_id, dict(payload))

    # --- Assistant memory (SMK index) --------------------------------------

    async def remember_assistant(
        self,
        *,
        trace: AssistantMemoryTrace,
        embedding: Iterable[float] | None = None,
    ) -> None:
        """Ingest a new assistant-focused trace into the SMK index."""
        if self.assistant_index is None:
            return
        emb_list = await self._resolve_embedding(
            embedding, f"{trace.summary}\n{trace.rationale}".strip()
        )
        self.assistant_index.ingest(trace, emb_list)

    async def recall_assistant(
        self,
        *,
        k: int,
        query_text: str = "",
        query_embedding: Iterable[float] | None = None,
        topic: TopicBucket | None = None,
        required_tools: set[ToolFlag] | None = None,
        allowed_kinds: list[MemoryKind] | None = None,
        min_generality: int | None = None,
        min_importance: int | None = None,
    ) -> list[AssistantMemoryHit]:
        """Query the assistant index with SMK-aware filters."""
        if self.assistant_index is None:
            return []

        from memory_core.types.smk_types import Level2Bits

        emb_list = await self._resolve_embedding(query_embedding, query_text)
        return self.assistant_index.query(
            query_embedding=emb_list,
            k=k,
            topic=topic,
            required_tools=required_tools or set(),
            allowed_kinds=allowed_kinds or [MemoryKind.PATTERN, MemoryKind.ANTI_PATTERN],
            min_generality=(
                Level2Bits(min_generality) if min_generality is not None else Level2Bits.MEDIUM
            ),
            min_importance=(
                Level2Bits(min_importance) if min_importance is not None else Level2Bits.MEDIUM
            ),
        )

    # --- Internals ---------------------------------------------------------

    async def _resolve_embedding(
        self,
        embedding: Iterable[float] | None,
        text: str,
    ) -> list[float]:
        if embedding is not None:
            return list(embedding)
        if self.embedder is None:
            raise ValueError(
                "no embedding supplied and no Embedder configured on this MemorySystem"
            )
        vectors = await self.embedder.embed([text])
        return vectors[0]
