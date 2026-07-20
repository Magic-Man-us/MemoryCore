"""MemorySystem orchestration across index, stores, and working memory."""

from __future__ import annotations

from datetime import timedelta

from conftest import make_assistant_trace
from memory_core_py.core.system import MemorySystem


def build_system(fake_index, fake_wm, fake_ltm, fake_stm=None, *, ttl=300) -> MemorySystem:
    return MemorySystem(
        memory_index=fake_index,
        working_mem=fake_wm,
        ltm_store=fake_ltm,
        stm_store=fake_stm,
        stm_ttl_seconds=ttl,
    )


EMBEDDING = [0.1, 0.2, 0.3]


class TestRemember:
    async def test_persists_across_all_layers(self, fake_index, fake_wm, fake_ltm, fake_stm):
        system = build_system(fake_index, fake_wm, fake_ltm, fake_stm, ttl=120)
        trace = await system.remember(
            user_id="u-1",
            content="full text",
            summary="short",
            importance=0.7,
            tags=["a", "b"],
            embedding=EMBEDDING,
        )

        assert trace.user_id == "u-1"
        assert trace.tags == {"a", "b"}
        assert trace.trace_uid  # generated

        # STM got the trace with created_at + ttl expiry.
        (stm_trace, expires_at), = fake_stm.inserts
        assert stm_trace.trace_uid == trace.trace_uid
        assert expires_at == trace.created_at + timedelta(seconds=120)

        # LTM and the vector index both received the embedding.
        (ltm_trace, ltm_emb), = fake_ltm.upserts
        assert ltm_trace.trace_uid == trace.trace_uid
        assert ltm_emb == EMBEDDING
        (idx_trace, idx_emb), = fake_index.ingested
        assert idx_trace.trace_uid == trace.trace_uid
        assert idx_emb == EMBEDDING

        # Working memory got an event describing the trace.
        (wm_user, payload), = fake_wm.events
        assert wm_user == "u-1"
        assert payload["kind"] == "ltm_trace"
        assert payload["trace_uid"] == trace.trace_uid

    async def test_working_memory_can_be_skipped(self, fake_index, fake_wm, fake_ltm):
        system = build_system(fake_index, fake_wm, fake_ltm)
        await system.remember(
            user_id="u-1",
            summary="short",
            importance=0.7,
            tags=[],
            embedding=EMBEDDING,
            also_working_mem=False,
        )
        assert fake_wm.events == []

    async def test_stm_is_optional(self, fake_index, fake_wm, fake_ltm):
        system = build_system(fake_index, fake_wm, fake_ltm, fake_stm=None)
        await system.remember(
            user_id="u-1", summary="s", importance=0.5, tags=[], embedding=EMBEDDING
        )
        assert len(fake_ltm.upserts) == 1  # no crash without STM


class TestRecall:
    async def test_aggregates_all_layers(self, fake_index, fake_wm, fake_ltm, fake_stm):
        from conftest import make_trace

        fake_wm.recent = [{"kind": "ltm_trace", "summary": "recent"}]
        fake_stm.recent = [make_trace(trace_uid="stm-1")]
        system = build_system(fake_index, fake_wm, fake_ltm, fake_stm)

        result = await system.recall(
            user_id="u-1",
            query_text="anything",
            tags=["a"],
            limit=5,
            query_embedding=EMBEDDING,
        )

        assert result["ltm_candidates"] == []
        assert result["wm_events"] == fake_wm.recent
        assert [t.trace_uid for t in result["stm_traces"]] == ["stm-1"]
        assert fake_index.searches == [
            {"user_id": "u-1", "text": "anything", "tags": ["a"], "limit": 5}
        ]

    async def test_working_memory_can_be_excluded(self, fake_index, fake_wm, fake_ltm):
        fake_wm.recent = [{"kind": "x"}]
        system = build_system(fake_index, fake_wm, fake_ltm)
        result = await system.recall(
            user_id="u-1", query_text="", tags=[], limit=3,
            query_embedding=EMBEDDING, include_working_mem=False,
        )
        assert result["wm_events"] == []
        assert result["stm_traces"] == []  # stm disabled


class TestAssistantPaths:
    async def test_noop_without_assistant_index(self, fake_index, fake_wm, fake_ltm):
        system = build_system(fake_index, fake_wm, fake_ltm)
        await system.remember_assistant(trace=make_assistant_trace(), embedding=EMBEDDING)
        hits = await system.recall_assistant(query_embedding=EMBEDDING, k=3)
        assert hits == []
