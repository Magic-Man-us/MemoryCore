# memory_core_py/system.py
from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from memory_core_py.types.smk_types import MemoryKind, ToolFlag, TopicBucket

from .models import AssistantMemoryTrace, MemoryCandidate, MemoryTrace

if TYPE_CHECKING:
    from collections.abc import Iterable

    from memory_core_py.core.interfaces import MemoryIndex
    from memory_core_py.indexing.assistant_index import AssistantMemoryIndex
    from memory_core_py.storage.ltm_store import LongTermStore
    from memory_core_py.storage.stm_store import ShortTermStore
    from memory_core_py.storage.working_mem import RedisWorkingMemory


class MemorySystem:
    """Coordinate memory ingest/recall workflows across all storage layers."""

    def __init__(
        self,
        *,
        memory_index: MemoryIndex,
        working_mem: RedisWorkingMemory,
        ltm_store: LongTermStore,
        assistant_index: AssistantMemoryIndex | None = None,
        stm_store: ShortTermStore | None = None,
        stm_ttl_seconds: int = 900,
    ) -> None:
        self.memory_index = memory_index
        self.working_mem = working_mem
        self.ltm_store = ltm_store
        self.assistant_index = assistant_index
        self.stm_store = stm_store
        self._stm_ttl_seconds = stm_ttl_seconds

    # --- User memory (what you already had) --------------------------------

    async def remember(
        self,
        *,
        user_id: str,
        content: str = "",
        summary: str,
        importance: float,
        tags: list[str],
        embedding: Iterable[float],
        also_working_mem: bool = True,
    ) -> MemoryTrace:
        """Persist a new trace across STM, LTM, the vector index, and working memory."""
        trace = MemoryTrace(
            trace_uid=str(uuid.uuid4()),
            user_id=user_id,
            content=content,
            summary=summary,
            importance=importance,
            created_at=datetime.now(UTC),
            access_count=0,
            tags=set(tags),
        )

        emb_list = list(embedding)

        if self.stm_store is not None:
            expires_at = trace.created_at + timedelta(seconds=self._stm_ttl_seconds)
            self.stm_store.insert_trace(trace, expires_at=expires_at)

        self.ltm_store.upsert_trace(trace, embedding=emb_list)
        self.memory_index.ingest(trace, emb_list)

        if also_working_mem:
            await self.working_mem.add_event(
                user_id=user_id,
                payload={
                    "kind": "ltm_trace",
                    "trace_uid": trace.trace_uid,
                    "summary": trace.summary,
                    "importance": trace.importance,
                    "tags": list(trace.tags),
                },
            )

        return trace

    async def recall(
        self,
        *,
        user_id: str,
        query_text: str,
        tags: list[str],
        limit: int,
        query_embedding: Iterable[float],
        include_working_mem: bool = True,
    ) -> dict[str, Any]:
        """Return recall candidates from LTM, STM, and working memory."""
        candidates: list[MemoryCandidate] = self.memory_index.search(
            user_id=user_id,
            text=query_text,
            tags=tags,
            limit=limit,
            query_embedding=list(query_embedding),
        )

        wm_events: list[dict[str, Any]] = []
        if include_working_mem:
            wm_events = await self.working_mem.get_recent(user_id=user_id, limit=20)

        stm_traces: list[MemoryTrace] = []
        if self.stm_store is not None:
            stm_traces = self.stm_store.fetch_recent_for_user(user_id=user_id, limit=limit)

        return {
            "ltm_candidates": candidates,
            "wm_events": wm_events,
            "stm_traces": stm_traces,
        }

    # --- Assistant memory (SMK index) --------------------------------------

    async def remember_assistant(
        self,
        *,
        trace: AssistantMemoryTrace,
        embedding: Iterable[float],
    ) -> None:
        """Ingest a new assistant-focused trace into the SMK index."""
        if self.assistant_index is None:
            return

        self.assistant_index.ingest(trace, embedding)

    async def recall_assistant(
        self,
        *,
        query_embedding: Iterable[float],
        k: int,
        topic: TopicBucket | None = None,
        required_tools: set[ToolFlag] | None = None,
        allowed_kinds: list[MemoryKind] | None = None,
    ):
        """Query the assistant index with SMK-aware filters."""
        if self.assistant_index is None:
            return []

        from memory_core_py.types.smk_types import Level2Bits

        return self.assistant_index.query(
            query_embedding=list(query_embedding),
            k=k,
            topic=topic,
            required_tools=required_tools or set(),
            allowed_kinds=allowed_kinds or [MemoryKind.PATTERN, MemoryKind.ANTI_PATTERN],
            min_generality=Level2Bits.MEDIUM,
            min_importance=Level2Bits.MEDIUM,
        )
