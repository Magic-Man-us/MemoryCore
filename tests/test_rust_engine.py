"""Round trips through the real Rust engine (built via maturin)."""

from __future__ import annotations

import pytest

pytest.importorskip(
    "memory_core._native", reason="Rust extension not built (run: maturin develop)"
)

from memory_core.indexing.rust_index import RustMemoryIndex
from tests.conftest import make_trace

DIM = 8


def vec(*head: float) -> list[float]:
    values = list(head) + [0.0] * (DIM - len(head))
    return values[:DIM]


class TestRustMemoryIndex:
    def test_ingest_search_round_trip(self):
        index = RustMemoryIndex()
        index.ingest(
            make_trace(trace_uid="t-1", user_id="u-1", summary="rust rocks"), vec(1.0)
        )
        index.ingest(
            make_trace(trace_uid="t-2", user_id="u-1", summary="python glue"),
            vec(0.0, 1.0),
        )

        results = index.search(
            user_id="u-1", text="", tags=[], limit=2, query_embedding=vec(1.0)
        )
        assert next(c.trace_uid for c in results) == "t-1"  # closest embedding wins
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
