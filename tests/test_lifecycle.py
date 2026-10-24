"""Memory lifecycle: access tracking, forgetting, expiry, write ordering, scoring,
and persistence of the SMK assistant index."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from memory_core import (
    Level2Bits,
    MemoryKind,
    MemorySystem,
    RustMemoryIndex,
    ScoringWeights,
    TopicBucket,
    build_memory_system,
)
from memory_core.storage import LongTermStore, ShortTermStore, SqlWorkingMemory
from tests.conftest import make_assistant_trace, make_trace

EMB = [0.1, 0.2, 0.3]


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


def _restart(system: MemorySystem) -> MemorySystem:
    """A fresh in-RAM index over the same database."""
    return MemorySystem(
        memory_index=RustMemoryIndex(),
        working_mem=system.working_mem,
        ltm_store=system.ltm_store,
        embedder=system.embedder,
        stm_store=system.stm_store,
    )


class TestAccessTracking:
    async def test_recall_persists_access_count_across_restart(self, system):
        trace = await system.remember(user_id="alice", summary="likes tea", importance=0.5,
                                      tags=[])
        for _ in range(3):
            await system.recall(user_id="alice", query_text="tea", limit=5)

        [(stored, _)] = system.ltm_store.fetch_traces_for_user("alice")
        assert stored.access_count == 3

        fresh = _restart(system)
        await fresh.hydrate("alice")
        result = await fresh.recall(user_id="alice", query_text="tea", limit=5)
        [candidate] = result.ltm_candidates
        assert candidate.trace_uid == trace.trace_uid
        assert candidate.access_count == 3  # value before this recall's increment
        assert candidate.importance == 0.5


class TestForget:
    async def test_forget_clears_stm_and_linked_wm_events(self, system):
        keep = await system.remember(user_id="alice", summary="keep me", importance=0.5,
                                     tags=[])
        drop = await system.remember(user_id="alice", summary="secret", importance=0.5,
                                     tags=[])
        await system.record_event(user_id="alice", payload={"kind": "turn", "text": "hi"})

        assert await system.forget(drop.trace_uid) is True
        result = await system.recall(user_id="alice", query_text="secret", limit=5)

        assert [c.trace_uid for c in result.ltm_candidates] == [keep.trace_uid]
        assert [t.trace_uid for t in result.stm_traces] == [keep.trace_uid]
        assert all(e.get("trace_uid") != drop.trace_uid for e in result.wm_events)
        assert any(e.get("trace_uid") == keep.trace_uid for e in result.wm_events)
        assert any(e.get("kind") == "turn" for e in result.wm_events)
        assert len(system.memory_index) == 1  # really freed, not tombstoned

    async def test_forget_unknown_is_false(self, system):
        assert await system.forget("nope") is False


class TestWriteOrdering:
    async def test_rejected_embedding_is_never_persisted(self, db):
        system = MemorySystem(
            memory_index=RustMemoryIndex(),
            working_mem=SqlWorkingMemory(db),
            ltm_store=LongTermStore(db),
        )
        await system.remember(user_id="alice", summary="ok", importance=0.5, tags=[],
                              embedding=EMB)
        with pytest.raises(ValueError, match="dimension mismatch"):
            await system.remember(user_id="alice", summary="bad", importance=0.5, tags=[],
                                  embedding=[0.1, 0.2])

        assert [t.summary for t, _ in system.ltm_store.fetch_traces_for_user("alice")] == ["ok"]
        assert system.working_mem.get_recent("alice")[-1]["summary"] == "ok"
        # and a restart can still hydrate cleanly
        assert await _restart(system).hydrate("alice") == 1

    async def test_failed_persist_rolls_back_the_index(self, db, monkeypatch):
        system = MemorySystem(
            memory_index=RustMemoryIndex(),
            working_mem=SqlWorkingMemory(db),
            ltm_store=LongTermStore(db),
        )

        def boom(*_args, **_kwargs):
            raise RuntimeError("disk full")

        monkeypatch.setattr(system.ltm_store, "upsert_trace", boom)
        with pytest.raises(RuntimeError, match="disk full"):
            await system.remember(user_id="alice", summary="x", importance=0.5, tags=[],
                                  embedding=EMB)
        assert len(system.memory_index) == 0


class TestExpiry:
    async def test_purge_expired_covers_quiet_users(self, system, db):
        from sqlalchemy import update

        from memory_core.storage.database import (
            ShortTermTraceORM,
            WorkingMemoryEventORM,
        )

        await system.remember(user_id="alice", summary="a", importance=0.5, tags=[])
        await system.remember(user_id="bob", summary="b", importance=0.5, tags=[])
        past = datetime.now(UTC) - timedelta(seconds=5)
        with db.session_scope() as session:
            session.execute(update(ShortTermTraceORM).values(expires_at=past))
            session.execute(update(WorkingMemoryEventORM).values(expires_at=past))
            session.commit()

        purged = await system.purge_expired()
        assert (purged.stm_traces, purged.wm_events) == (2, 2)
        assert (await system.purge_expired()).wm_events == 0


class TestScoring:
    def test_recency_weight_reorders_equal_matches(self):
        now = datetime.now(UTC)
        index = RustMemoryIndex(weights=ScoringWeights(semantic=0.5, importance=0.0,
                                                       recency=0.5,
                                                       recency_half_life_days=7))
        old = make_trace(trace_uid="old", user_id="u", importance=0.5)
        old.created_at = now - timedelta(days=60)
        new = make_trace(trace_uid="new", user_id="u", importance=0.5)
        new.created_at = now - timedelta(hours=1)
        index.ingest(old, EMB)
        index.ingest(new, EMB)

        hits = index.search(user_id="u", text="", tags=[], limit=5, query_embedding=EMB,
                            now=now)
        assert [h.trace_uid for h in hits] == ["new", "old"]
        assert hits[0].score > hits[1].score

    async def test_scoring_overrides_reach_the_engine(self, tmp_path):
        system = build_memory_system(
            sqlite_path=tmp_path / "s.db",
            overrides={"scoring": {"semantic": 0.0, "importance": 0.0, "access": 1.0}},
        )
        hot = await system.remember(user_id="u", summary="hot", importance=0.1, tags=[],
                                    embedding=EMB)
        await system.remember(user_id="u", summary="cold", importance=0.9, tags=[],
                              embedding=EMB)
        for _ in range(5):
            system.memory_index.mark_accessed(hot.trace_uid)
        result = await system.recall(user_id="u", query_text="", limit=1,
                                     query_embedding=EMB)
        assert [c.trace_uid for c in result.ltm_candidates] == [hot.trace_uid]

    def test_invalid_weights_are_rejected(self):
        with pytest.raises(ValueError):
            ScoringWeights(recency=-0.1)
        with pytest.raises(ValueError):
            ScoringWeights(recency_half_life_days=0)
        with pytest.raises(ValueError):
            ScoringWeights(typo=1.0)


DIM = 8


def _assistant_system(tmp_path, **overrides) -> MemorySystem:
    return build_memory_system(
        sqlite_path=tmp_path / "assistant.db",
        overrides={"enable_assistant_index": True, "assistant_index_dim": DIM, **overrides},
    )


def _vec(i: int) -> list[float]:
    return [1.0 if j == i else 0.0 for j in range(DIM)]


class TestAssistantMemory:
    async def test_persists_across_restart(self, tmp_path):
        first = _assistant_system(tmp_path)
        trace = await first.remember_assistant(trace=make_assistant_trace(trace_uid="p-1"),
                                               embedding=_vec(0))
        assert trace.smk_raw is not None and trace.smk_raw & 0xFF == trace.context_topic

        second = _assistant_system(tmp_path)
        assert await second.recall_assistant(k=5, query_embedding=_vec(0)) == []
        assert await second.hydrate_assistant() == 1
        [hit] = await second.recall_assistant(k=5, query_embedding=_vec(0))
        assert hit.trace_uid == "p-1"
        assert hit.smk_raw == trace.smk_raw

        [(stored, emb)] = second.assistant_store.fetch_all()
        assert stored == trace
        assert emb == _vec(0)

    async def test_reingest_is_an_upsert(self, tmp_path):
        system = _assistant_system(tmp_path)
        for i in range(3):
            await system.remember_assistant(
                trace=make_assistant_trace(trace_uid="p-1", summary=f"v{i}"),
                embedding=_vec(i),
            )
        assert len(system.assistant_index) == 1
        assert len(system.assistant_store.fetch_all()) == 1
        hits = await system.recall_assistant(k=5, query_embedding=_vec(2))
        assert [h.trace_uid for h in hits] == ["p-1"]
        assert hits[0].score == pytest.approx(1.0, abs=1e-5)

    async def test_forget_assistant(self, tmp_path):
        system = _assistant_system(tmp_path)
        await system.remember_assistant(trace=make_assistant_trace(trace_uid="p-1"),
                                        embedding=_vec(0))
        assert await system.forget_assistant("p-1") is True
        assert await system.forget_assistant("p-1") is False
        assert await system.recall_assistant(k=5, query_embedding=_vec(0)) == []
        assert await _assistant_system(tmp_path).hydrate_assistant() == 0

    async def test_filters_and_widening_defaults(self, tmp_path):
        system = _assistant_system(tmp_path)
        await system.remember_assistant(
            trace=make_assistant_trace(trace_uid="insight", kind=MemoryKind.INSIGHT,
                                       generality=0.1),
            embedding=_vec(0),
        )
        # defaults: PATTERN/ANTI_PATTERN, >= MEDIUM generality
        assert await system.recall_assistant(k=5, query_embedding=_vec(0)) == []
        hits = await system.recall_assistant(
            k=5, query_embedding=_vec(0), allowed_kinds=list(MemoryKind),
            min_generality=None, min_importance=None,
        )
        assert [h.trace_uid for h in hits] == ["insight"]
        assert await system.recall_assistant(
            k=5, query_embedding=_vec(0), allowed_kinds=list(MemoryKind),
            min_generality=Level2Bits.LOW, topic=TopicBucket.DB_SCHEMA,
        ) == []

    async def test_disabled_index_raises_on_write(self, tmp_path):
        system = build_memory_system(sqlite_path=tmp_path / "x.db")
        with pytest.raises(RuntimeError, match="assistant index is not enabled"):
            await system.remember_assistant(trace=make_assistant_trace(), embedding=_vec(0))
        assert await system.recall_assistant(k=5, query_embedding=_vec(0)) == []

    async def test_dimension_mismatch_is_not_persisted(self, tmp_path):
        system = _assistant_system(tmp_path)
        with pytest.raises(ValueError, match="dimension mismatch"):
            await system.remember_assistant(trace=make_assistant_trace(), embedding=[1.0])
        assert system.assistant_store.fetch_all() == []


class TestReviewRegressions:
    """PR #17 review findings: write ordering, forget across config, concurrency."""

    async def test_trace_is_durable_before_it_is_searchable(self, system, monkeypatch):
        original = system.memory_index.ingest
        seen_in_ltm: list[bool] = []

        def spy(trace, embedding):
            rows = system.ltm_store.fetch_traces_for_user(trace.user_id)
            seen_in_ltm.append(any(t.trace_uid == trace.trace_uid for t, _ in rows))
            original(trace, embedding)

        monkeypatch.setattr(system.memory_index, "ingest", spy)
        await system.remember(user_id="alice", summary="x", importance=0.5, tags=[])
        assert seen_in_ltm == [True]

    async def test_index_rejection_after_persist_removes_the_row(self, system, monkeypatch):
        def reject(*_args, **_kwargs):
            raise ValueError("embedding dimension mismatch: raced")

        monkeypatch.setattr(system.memory_index, "ingest", reject)
        with pytest.raises(ValueError, match="raced"):
            await system.remember(user_id="alice", summary="x", importance=0.5, tags=[])
        assert system.ltm_store.fetch_traces_for_user("alice") == []

    async def test_forget_clears_stm_written_while_stm_was_enabled(self, system, db):
        trace = await system.remember(user_id="alice", summary="secret", importance=0.5,
                                      tags=[])
        stm_off = MemorySystem(
            memory_index=RustMemoryIndex(),
            working_mem=system.working_mem,
            ltm_store=system.ltm_store,
            embedder=system.embedder,
        )
        assert await stm_off.forget(trace.trace_uid) is True
        result = await system.recall(user_id="alice", query_text="secret", limit=5)
        assert result.stm_traces == []

    async def test_hydrate_assistant_is_zero_when_disabled(self, tmp_path):
        system = build_memory_system(sqlite_path=tmp_path / "x.db")
        assert await system.hydrate_assistant() == 0

    async def test_failed_store_write_keeps_previous_version_searchable(
        self, tmp_path, monkeypatch
    ):
        system = _assistant_system(tmp_path)
        await system.remember_assistant(trace=make_assistant_trace(trace_uid="p-1"),
                                        embedding=_vec(0))

        def boom(*_args, **_kwargs):
            raise RuntimeError("disk full")

        monkeypatch.setattr(system.assistant_store, "upsert_trace", boom)
        with pytest.raises(RuntimeError, match="disk full"):
            await system.remember_assistant(
                trace=make_assistant_trace(trace_uid="p-1", summary="v2"), embedding=_vec(1)
            )
        [hit] = await system.recall_assistant(k=5, query_embedding=_vec(0))
        assert hit.trace_uid == "p-1"
        assert hit.score == pytest.approx(1.0, abs=1e-5)

    async def test_concurrent_upserts_of_one_uid_leave_index_and_store_agreeing(
        self, tmp_path, monkeypatch
    ):
        import time

        system = _assistant_system(tmp_path)
        original = system.assistant_index.ingest

        def slow_for_v1(trace, embedding):
            if trace.summary == "v1":
                time.sleep(0.2)
            return original(trace, embedding)

        monkeypatch.setattr(system.assistant_index, "ingest", slow_for_v1)

        async def second():
            await asyncio.sleep(0.05)
            await system.remember_assistant(
                trace=make_assistant_trace(trace_uid="p-1", summary="v2"), embedding=_vec(2)
            )

        await asyncio.gather(
            system.remember_assistant(
                trace=make_assistant_trace(trace_uid="p-1", summary="v1"), embedding=_vec(1)
            ),
            second(),
        )
        [(stored, stored_emb)] = system.assistant_store.fetch_all()
        [hit] = await system.recall_assistant(k=1, query_embedding=stored_emb)
        assert stored.summary == "v2"
        assert hit.score == pytest.approx(1.0, abs=1e-5), "index holds the stored version"

    def test_query_drops_hits_removed_mid_query(self):
        from memory_core import AssistantMemoryIndex

        index = AssistantMemoryIndex(dim=DIM)
        index.ingest(make_assistant_trace(trace_uid="p-1"), _vec(0))
        index._id_to_uid.clear()  # what a concurrent remove() leaves behind
        assert index.query(_vec(0), k=5) == []
