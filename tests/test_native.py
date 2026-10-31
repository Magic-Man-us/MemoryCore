"""Smoke tests for the Rust extension bindings (memory_core._native)."""

from __future__ import annotations

import pytest
from memory_core._native import PyMemoryEngine

DIM = 8


def emb(seed: int) -> list[float]:
    return [float((seed * (i + 1)) % 7) / 7.0 for i in range(DIM)]


def test_engine_kwargs_and_scoring():
    engine = PyMemoryEngine()
    engine.ingest_trace(
        "a1", "alice", "loves gardening", 0.9, 1_700_000_000, 0, ["hobby"], emb(3)
    )
    engine.ingest_trace(
        "a2", "alice", "works with terraform", 0.7, 1_700_000_100, 0, [], emb(5)
    )
    res = engine.search_candidates(
        user_id="alice", text="gardening", tags=[], limit=5, query_embedding=emb(3)
    )
    assert next(c.trace_uid for c in res) == "a1"
    assert res[0].score > res[1].score


def test_engine_user_isolation_at_scoring():
    engine = PyMemoryEngine()
    engine.ingest_trace("b1", "bob", "secret plans", 0.9, 1, 0, [], emb(1))
    assert (
        engine.search_candidates(
            user_id="alice",
            text="secret plans",
            tags=[],
            limit=10,
            query_embedding=emb(1),
        )
        == []
    )


def test_engine_keyword_flood_does_not_suppress_fallback():
    """Review P1-6: another user's keyword matches must not collapse recall to zero."""
    engine = PyMemoryEngine()
    for i in range(5):
        engine.ingest_trace(
            f"b{i}", "bob", f"python tip {i}", 0.5, 1 + i, 0, [], emb(i + 10)
        )
    engine.ingest_trace("a1", "alice", "prefers snake case", 0.9, 100, 0, [], emb(2))
    engine.ingest_trace(
        "a2", "alice", "dislikes long meetings", 0.8, 101, 0, [], emb(4)
    )

    hits = engine.search_candidates(
        user_id="alice", text="python", tags=[], limit=3, query_embedding=emb(2)
    )
    assert sorted(c.trace_uid for c in hits) == ["a1", "a2"]


def test_engine_query_dim_mismatch_raises_value_error():
    engine = PyMemoryEngine()
    engine.ingest_trace("a1", "alice", "s", 0.5, 1, 0, [], emb(1))
    with pytest.raises(ValueError, match="dimension mismatch"):
        engine.search_candidates(
            user_id="alice", text="", tags=[], limit=5, query_embedding=[0.1, 0.2]
        )


def test_engine_nan_score_sinks_to_the_bottom_deterministically():
    """PR review: unwrap_or(Equal) ties NaN with every finite score, so a naive fix
    that only avoids the panic can still scatter NaN hits through valid results.
    score_cmp_desc must give NaN a fixed, worst-ranked position."""
    engine = PyMemoryEngine()
    nan_emb = [float("nan")] * DIM
    engine.ingest_trace("good1", "alice", "s1", 0.9, 1, 0, [], emb(1))
    engine.ingest_trace("bad", "alice", "s2", 0.9, 2, 0, [], nan_emb)
    engine.ingest_trace("good2", "alice", "s3", 0.5, 3, 0, [], emb(4))

    orders = [
        [
            c.trace_uid
            for c in engine.search_candidates(
                user_id="alice", text="", tags=[], limit=10, query_embedding=emb(1)
            )
        ]
        for _ in range(5)
    ]
    assert all(order == orders[0] for order in orders), "sort must be deterministic"
    assert orders[0][-1] == "bad"  # NaN always ranks last
    assert set(orders[0][:-1]) == {"good1", "good2"}


def test_engine_rejects_invalid_weights():
    with pytest.raises(ValueError, match="semantic_weight"):
        PyMemoryEngine(semantic_weight=-1.0)
    with pytest.raises(ValueError, match="recency_half_life_days"):
        PyMemoryEngine(recency_half_life_days=0.0)
