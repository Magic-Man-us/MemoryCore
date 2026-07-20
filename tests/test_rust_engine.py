"""Round trips through the real Rust engine (built via maturin)."""

from __future__ import annotations

import pytest

pytest.importorskip("memory_core", reason="Rust extension not built (run: uv sync)")

from conftest import make_assistant_trace, make_trace  # noqa: E402
from memory_core_py.indexing.assistant_index import AssistantMemoryIndex  # noqa: E402
from memory_core_py.indexing.rust_index import RustMemoryIndex  # noqa: E402
from memory_core_py.types.smk_types import (  # noqa: E402
    MemoryKind,
    ToolFlag,
    TopicBucket,
)

DIM = 8


def vec(*head: float) -> list[float]:
    values = list(head) + [0.0] * (DIM - len(head))
    return values[:DIM]


class TestRustMemoryIndex:
    def test_ingest_search_round_trip(self):
        index = RustMemoryIndex()
        index.ingest(make_trace(trace_uid="t-1", user_id="u-1", summary="rust rocks"),
                     vec(1.0))
        index.ingest(make_trace(trace_uid="t-2", user_id="u-1", summary="python glue"),
                     vec(0.0, 1.0))

        results = index.search(
            user_id="u-1", text="", tags=[], limit=2, query_embedding=vec(1.0)
        )
        assert [c.trace_uid for c in results][0] == "t-1"  # closest embedding wins
        assert results[0].summary == "rust rocks"
        assert results[0].score >= results[-1].score

    def test_users_are_isolated(self):
        index = RustMemoryIndex()
        index.ingest(make_trace(trace_uid="mine", user_id="u-1"), vec(1.0))
        index.ingest(make_trace(trace_uid="theirs", user_id="u-2"), vec(1.0))

        results = index.search(
            user_id="u-1", text="", tags=[], limit=10, query_embedding=vec(1.0)
        )
        assert [c.trace_uid for c in results] == ["mine"]

    def test_remove_and_mark_accessed(self):
        index = RustMemoryIndex()
        index.ingest(make_trace(trace_uid="t-1", user_id="u-1"), vec(1.0))
        index.mark_accessed("t-1")  # must not raise
        index.remove("t-1")
        results = index.search(
            user_id="u-1", text="", tags=[], limit=5, query_embedding=vec(1.0)
        )
        assert results == []


class TestAssistantIndex:
    def _ingest(self, index: AssistantMemoryIndex, uid: str, *, topic, tools, embedding):
        index.ingest(
            make_assistant_trace(trace_uid=uid, context_topic=topic, tools=tools),
            embedding,
        )

    def test_query_maps_ids_back_to_trace_uids(self):
        index = AssistantMemoryIndex(dim=DIM)
        self._ingest(
            index, "a-1",
            topic=TopicBucket.RUST_PYTHON_TOOLCHAIN,
            tools={ToolFlag.PY},
            embedding=vec(1.0),
        )
        hits = index.query(vec(1.0), k=3, allowed_kinds=[MemoryKind.PATTERN])
        assert [h.trace_uid for h in hits] == ["a-1"]
        assert hits[0].smk_raw > 0

    def test_topic_filter_excludes_other_topics(self):
        index = AssistantMemoryIndex(dim=DIM)
        self._ingest(
            index, "toolchain",
            topic=TopicBucket.RUST_PYTHON_TOOLCHAIN, tools={ToolFlag.PY},
            embedding=vec(1.0),
        )
        self._ingest(
            index, "schema",
            topic=TopicBucket.DB_SCHEMA, tools={ToolFlag.PY},
            embedding=vec(1.0),
        )

        hits = index.query(
            vec(1.0), k=5,
            topic=TopicBucket.DB_SCHEMA,
            allowed_kinds=[MemoryKind.PATTERN],
        )
        assert [h.trace_uid for h in hits] == ["schema"]

    def test_required_tools_filter(self):
        index = AssistantMemoryIndex(dim=DIM)
        self._ingest(
            index, "with-maturin",
            topic=TopicBucket.RUST_PYTHON_TOOLCHAIN,
            tools={ToolFlag.PY, ToolFlag.MATURIN},
            embedding=vec(1.0),
        )
        self._ingest(
            index, "py-only",
            topic=TopicBucket.RUST_PYTHON_TOOLCHAIN,
            tools={ToolFlag.PY},
            embedding=vec(1.0),
        )

        hits = index.query(
            vec(1.0), k=5,
            required_tools={ToolFlag.MATURIN},
            allowed_kinds=[MemoryKind.PATTERN],
        )
        assert [h.trace_uid for h in hits] == ["with-maturin"]
