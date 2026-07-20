"""End-to-end MemorySystem tests: real Rust index, real temp SQLite, fake embedder."""

from __future__ import annotations

import pytest

from memory_core import MemorySystem, RustMemoryIndex
from memory_core.storage import LongTermStore, ShortTermStore, SqlWorkingMemory


@pytest.fixture
def system(db, embedder) -> MemorySystem:
    return MemorySystem(
        memory_index=RustMemoryIndex(),
        working_mem=SqlWorkingMemory(db, ttl_seconds=60),
        ltm_store=LongTermStore(db),
        embedder=embedder,
        stm_store=ShortTermStore(db),
        stm_ttl_seconds=60,
    )


async def test_remember_recall_roundtrip(system):
    trace = await system.remember(
        user_id="alice",
        summary="prefers uv over pip",
        content="alice said: always use uv",
        importance=0.9,
        tags=["preference", "python"],
    )
    assert trace.trace_uid

    result = await system.recall(user_id="alice", query_text="uv pip preference", limit=5)
    assert [c.trace_uid for c in result.ltm_candidates] == [trace.trace_uid]
    # the remember() call also left a working-memory event
    assert any(e.get("kind") == "ltm_trace" for e in result.wm_events)
    # and an STM trace
    assert [t.trace_uid for t in result.stm_traces] == [trace.trace_uid]


async def test_working_memory_can_be_skipped_on_remember(system):
    await system.remember(
        user_id="alice",
        summary="quiet write",
        importance=0.5,
        tags=[],
        also_working_mem=False,
    )
    result = await system.recall(user_id="alice", query_text="quiet write", limit=5)
    assert result.wm_events == []


async def test_working_memory_can_be_excluded_on_recall(system):
    await system.remember(user_id="alice", summary="noisy write", importance=0.5, tags=[])
    result = await system.recall(
        user_id="alice", query_text="noisy write", limit=5, include_working_mem=False
    )
    assert result.wm_events == []
    assert len(result.ltm_candidates) == 1


async def test_recall_requires_embedder_or_embedding(db):
    system = MemorySystem(
        memory_index=RustMemoryIndex(),
        working_mem=SqlWorkingMemory(db),
        ltm_store=LongTermStore(db),
    )
    with pytest.raises(ValueError, match="no Embedder configured"):
        await system.recall(user_id="alice", query_text="anything", limit=5)


async def test_explicit_embedding_still_works(db):
    system = MemorySystem(
        memory_index=RustMemoryIndex(),
        working_mem=SqlWorkingMemory(db),
        ltm_store=LongTermStore(db),
    )
    await system.remember(
        user_id="alice",
        summary="s",
        importance=0.5,
        tags=[],
        embedding=[0.1, 0.2, 0.3],
    )
    result = await system.recall(
        user_id="alice", query_text="s", limit=5, query_embedding=[0.1, 0.2, 0.3]
    )
    assert len(result.ltm_candidates) == 1


async def test_stm_is_optional(db, embedder):
    system = MemorySystem(
        memory_index=RustMemoryIndex(),
        working_mem=SqlWorkingMemory(db),
        ltm_store=LongTermStore(db),
        embedder=embedder,
    )
    await system.remember(user_id="alice", summary="no stm", importance=0.5, tags=[])
    result = await system.recall(user_id="alice", query_text="no stm", limit=5)
    assert result.stm_traces == []
    assert len(result.ltm_candidates) == 1


async def test_hydrate_restores_index_from_ltm(db, embedder):
    first = MemorySystem(
        memory_index=RustMemoryIndex(),
        working_mem=SqlWorkingMemory(db),
        ltm_store=LongTermStore(db),
        embedder=embedder,
    )
    kept = await first.remember(
        user_id="alice", summary="likes rust", importance=0.8, tags=["lang"]
    )

    # simulate a restart: fresh index, same database
    second = MemorySystem(
        memory_index=RustMemoryIndex(),
        working_mem=SqlWorkingMemory(db),
        ltm_store=LongTermStore(db),
        embedder=embedder,
    )
    loaded = await second.hydrate("alice")
    assert loaded == 1
    result = await second.recall(user_id="alice", query_text="likes rust", limit=5)
    assert [c.trace_uid for c in result.ltm_candidates] == [kept.trace_uid]


async def test_forget_removes_everywhere(system):
    trace = await system.remember(
        user_id="alice", summary="temporary fact", importance=0.5, tags=[]
    )
    assert await system.forget(trace.trace_uid) is True
    result = await system.recall(user_id="alice", query_text="temporary fact", limit=5)
    assert result.ltm_candidates == []
    # and it is gone from LTM, so a rehydrate cannot resurrect it
    fresh = MemorySystem(
        memory_index=RustMemoryIndex(),
        working_mem=system.working_mem,
        ltm_store=system.ltm_store,
        embedder=system.embedder,
    )
    assert await fresh.hydrate("alice") == 0


async def test_record_event(system):
    await system.record_event(user_id="alice", payload={"kind": "turn", "text": "hi"})
    result = await system.recall(user_id="alice", query_text="hi", limit=5)
    assert any(e.get("kind") == "turn" for e in result.wm_events)
