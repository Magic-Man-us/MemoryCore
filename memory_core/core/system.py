"""MemorySystem: the async orchestrator over the index, the SQL stores, and working memory.

The stores are synchronous SQLAlchemy and the Rust engines release the GIL; every
call here runs in a worker thread (``asyncio.to_thread``) so the caller's event loop
(a streaming assistant turn) never blocks on database I/O or vector search.

Consistency rule: LTM (and the assistant store) is the source of truth and the
in-RAM indexes mirror it. Writes validate first, persist second, and publish to the
index last: input the index would reject (a wrong embedding dimension) is never
persisted, a failed persist leaves the index untouched (including the previous
version of an updated trace), and a trace is never searchable before it is durable.
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from memory_core.types.smk_types import Level2Bits, MemoryKind, ToolFlag, TopicBucket

from .models import (
    AssistantMemoryTrace,
    MemoryCandidate,
    MemoryTrace,
    PurgeResult,
    RecallResult,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from memory_core.core.interfaces import Embedder, MemoryIndex
    from memory_core.indexing.assistant_index import (
        AssistantMemoryHit,
        AssistantMemoryIndex,
    )
    from memory_core.storage.assistant_store import AssistantStore
    from memory_core.storage.ltm_store import LongTermStore
    from memory_core.storage.stm_store import ShortTermStore
    from memory_core.storage.working_mem import SqlWorkingMemory

DEFAULT_ASSISTANT_KINDS: tuple[MemoryKind, ...] = (MemoryKind.PATTERN, MemoryKind.ANTI_PATTERN)


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
        assistant_store: AssistantStore | None = None,
        stm_store: ShortTermStore | None = None,
        stm_ttl_seconds: int = 900,
    ) -> None:
        self.memory_index = memory_index
        self.working_mem = working_mem
        self.ltm_store = ltm_store
        self.embedder = embedder
        self.assistant_index = assistant_index
        self.assistant_store = assistant_store
        self.stm_store = stm_store
        self._stm_ttl_seconds = stm_ttl_seconds
        # Serializes each assistant write (validate, persist, publish) so two
        # concurrent upserts of one uid cannot leave the index and store disagreeing.
        self._assistant_lock = threading.Lock()

    # --- Startup -----------------------------------------------------------

    async def hydrate(self, user_id: str) -> int:
        """Load a user's persisted traces from LTM into the in-memory index.

        Returns the number of traces ingested (traces persisted without an
        embedding are skipped: they cannot be vector-searched). Safe to repeat:
        ingest is an upsert.
        """

        def _load() -> int:
            loaded = 0
            for trace, embedding in self.ltm_store.fetch_traces_for_user(user_id):
                if embedding:
                    self.memory_index.ingest(trace, embedding)
                    loaded += 1
            return loaded

        return await asyncio.to_thread(_load)

    async def hydrate_assistant(self) -> int:
        """Load every persisted assistant trace into the SMK index.

        Returns the number ingested; 0 when the assistant index or store is not
        configured (like ``recall_assistant``, reads never raise for a disabled index).
        """
        index = self.assistant_index
        store = self.assistant_store
        if index is None or store is None:
            return 0

        def _load() -> int:
            loaded = 0
            with self._assistant_lock:
                for trace, embedding in store.fetch_all():
                    if embedding:
                        index.ingest(trace, embedding)
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
        """Persist a new trace to LTM (and STM if enabled), index it, and log a WM event.

        Pass ``embedding`` explicitly, or construct the system with an ``Embedder``
        and it is computed here from the summary + content. ``extra`` carries
        provenance (source surface, session id, ...) and round-trips through LTM.
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
            # Validate, persist, then publish: a rejected embedding never reaches
            # LTM (where it would break every later hydrate), and a concurrent
            # recall cannot see (and count an access on) a trace not yet durable.
            self.memory_index.check_embedding(emb_list)
            self.ltm_store.upsert_trace(trace, embedding=emb_list)
            try:
                self.memory_index.ingest(trace, emb_list)
            except BaseException:
                # e.g. two concurrent first writes with different dimensions
                self.ltm_store.delete_trace(trace.trace_uid)
                raise
            if self.stm_store is not None:
                expires_at = trace.created_at + timedelta(seconds=self._stm_ttl_seconds)
                self.stm_store.insert_trace(trace, expires_at=expires_at)
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
        """Recall index candidates plus recent STM traces and WM events.

        Every returned candidate is marked accessed, in the index and in LTM, so
        access counts survive restarts and can feed the access scoring weight.
        """
        emb_list = await self._resolve_embedding(query_embedding, query_text)

        def _recall() -> RecallResult:
            candidates: list[MemoryCandidate] = self.memory_index.search(
                user_id=user_id,
                text=query_text,
                tags=tags or [],
                limit=limit,
                query_embedding=emb_list,
            )
            uids = [candidate.trace_uid for candidate in candidates]
            for uid in uids:
                self.memory_index.mark_accessed(uid)
            if uids:
                self.ltm_store.increment_access(uids)

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
        """Remove a trace everywhere: index, LTM, STM, and linked WM events.

        Returns True when LTM had the row.
        """

        from memory_core.storage.stm_store import ShortTermStore

        # STM rows are deleted even when STM is disabled on this system: they may
        # have been written by a process that had it enabled, and would resurface
        # if STM were turned back on before they expire.
        stm_store = self.stm_store or ShortTermStore(self.ltm_store.db)

        def _forget() -> bool:
            self.memory_index.remove(trace_uid)
            deleted = self.ltm_store.delete_trace(trace_uid)
            stm_store.delete_trace(trace_uid)
            self.working_mem.delete_for_trace(trace_uid)
            return deleted

        return await asyncio.to_thread(_forget)

    async def record_event(self, *, user_id: str, payload: dict[str, object]) -> None:
        """Append a raw working-memory event (a turn, a tool use, a session marker)."""
        await asyncio.to_thread(self.working_mem.add_event, user_id, dict(payload))

    async def purge_expired(self) -> PurgeResult:
        """Delete expired STM traces and WM events for every user.

        WM already prunes a user's expired events on that user's writes; this
        catches users who went quiet. Run it periodically from a long-lived process.
        """

        def _purge() -> PurgeResult:
            return PurgeResult(
                stm_traces=self.stm_store.purge_expired() if self.stm_store is not None else 0,
                wm_events=self.working_mem.purge_expired(),
            )

        return await asyncio.to_thread(_purge)

    # --- Assistant memory (SMK index) --------------------------------------

    async def remember_assistant(
        self,
        *,
        trace: AssistantMemoryTrace,
        embedding: Iterable[float] | None = None,
    ) -> AssistantMemoryTrace:
        """Persist an assistant trace and index it (upsert by ``trace_uid``).

        Sets ``trace.smk_raw``. Raises RuntimeError when the assistant index is not
        enabled: silently dropping a write would lose what the assistant learned.
        If persisting fails, the index (and any previous version of the trace in
        it) is left untouched.
        """
        index = self._require_assistant_index()
        emb_list = await self._resolve_embedding(
            embedding, f"{trace.summary}\n{trace.rationale}".strip()
        )

        def _persist() -> None:
            with self._assistant_lock:
                validated = index.prepare(trace, emb_list)
                if self.assistant_store is not None:
                    self.assistant_store.upsert_trace(trace, embedding=validated)
                index.ingest(trace, validated)

        await asyncio.to_thread(_persist)
        return trace

    async def forget_assistant(self, trace_uid: str) -> bool:
        """Remove an assistant trace from the SMK index and the store."""
        index = self._require_assistant_index()

        def _forget() -> bool:
            with self._assistant_lock:
                removed = index.remove(trace_uid)
                if self.assistant_store is not None:
                    removed = self.assistant_store.delete_trace(trace_uid) or removed
                return removed

        return await asyncio.to_thread(_forget)

    async def recall_assistant(
        self,
        *,
        k: int,
        query_text: str = "",
        query_embedding: Iterable[float] | None = None,
        topic: TopicBucket | None = None,
        required_tools: set[ToolFlag] | None = None,
        allowed_kinds: list[MemoryKind] | None = None,
        min_generality: Level2Bits | None = Level2Bits.MEDIUM,
        min_importance: Level2Bits | None = Level2Bits.MEDIUM,
    ) -> list[AssistantMemoryHit]:
        """Query the assistant index with SMK-aware filters.

        Defaults favour reusable guidance: kinds PATTERN/ANTI_PATTERN and at least
        MEDIUM generality and importance. Pass ``allowed_kinds=list(MemoryKind)``
        and ``None`` for the minimums to search everything. Returns [] when the
        assistant index is not enabled.
        """
        if self.assistant_index is None:
            return []
        index = self.assistant_index
        emb_list = await self._resolve_embedding(query_embedding, query_text)
        kinds = list(allowed_kinds) if allowed_kinds else list(DEFAULT_ASSISTANT_KINDS)

        return await asyncio.to_thread(
            index.query,
            emb_list,
            k,
            topic=topic,
            required_tools=required_tools or set(),
            allowed_kinds=kinds,
            min_generality=min_generality,
            min_importance=min_importance,
        )

    # --- Internals ---------------------------------------------------------

    def _require_assistant_index(self) -> AssistantMemoryIndex:
        if self.assistant_index is None:
            raise RuntimeError(
                "assistant index is not enabled: build with enable_assistant_index=True "
                "and assistant_index_dim"
            )
        return self.assistant_index

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
