"""Smoke tests for the Rust extension bindings (memory_core._native)."""

from __future__ import annotations

import pytest

from memory_core import (
    AssistantMemoryIndex,
    AssistantMemoryTrace,
    Level2Bits,
    MemoryKind,
    ToolFlag,
    TopicBucket,
)
from memory_core._native import PyMemoryEngine
from memory_core.indexing.assistant_index import stable_memory_id

DIM = 8


def emb(seed: int) -> list[float]:
    return [float((seed * (i + 1)) % 7) / 7.0 for i in range(DIM)]


def test_engine_kwargs_and_scoring():
    engine = PyMemoryEngine()
    engine.ingest_trace("a1", "alice", "loves gardening", 0.9, 1_700_000_000, 0, ["hobby"], emb(3))
    engine.ingest_trace("a2", "alice", "works with terraform", 0.7, 1_700_000_100, 0, [], emb(5))
    res = engine.search_candidates(
        user_id="alice", text="gardening", tags=[], limit=5, query_embedding=emb(3)
    )
    assert [c.trace_uid for c in res][0] == "a1"
    assert res[0].score > res[1].score


def test_engine_user_isolation_at_scoring():
    engine = PyMemoryEngine()
    engine.ingest_trace("b1", "bob", "secret plans", 0.9, 1, 0, [], emb(1))
    assert (
        engine.search_candidates(
            user_id="alice", text="secret plans", tags=[], limit=10, query_embedding=emb(1)
        )
        == []
    )


def test_engine_keyword_flood_does_not_suppress_fallback():
    """Review P1-6: another user's keyword matches must not collapse recall to zero."""
    engine = PyMemoryEngine()
    for i in range(5):
        engine.ingest_trace(f"b{i}", "bob", f"python tip {i}", 0.5, 1 + i, 0, [], emb(i + 10))
    engine.ingest_trace("a1", "alice", "prefers snake case", 0.9, 100, 0, [], emb(2))
    engine.ingest_trace("a2", "alice", "dislikes long meetings", 0.8, 101, 0, [], emb(4))

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


def test_smk_roundtrip_via_wrapper():
    index = AssistantMemoryIndex(dim=DIM)
    trace = AssistantMemoryTrace(
        trace_uid="pattern-1",
        context_topic=TopicBucket.RUST_PYTHON_TOOLCHAIN,
        context_activity="integrating",
        context_complexity=0.6,
        kind=MemoryKind.PATTERN,
        tools={ToolFlag.RS, ToolFlag.PY},
        summary="pyo3 mixed layout",
        rationale="module-name pkg._native",
        before_state_confusion=0.9,
        after_state_confidence=0.9,
        generality=0.8,
        importance=0.9,
    )
    index.ingest(trace, emb(1))

    hits = index.query(
        emb(1),
        k=5,
        topic=TopicBucket.RUST_PYTHON_TOOLCHAIN,
        required_tools={ToolFlag.RS},
        allowed_kinds=[MemoryKind.PATTERN],
        min_generality=Level2Bits.MEDIUM,
        min_importance=Level2Bits.MEDIUM,
    )
    assert [h.trace_uid for h in hits] == ["pattern-1"]

    # a topic filter that doesn't match prunes it
    assert (
        index.query(emb(1), k=5, topic=TopicBucket.DB_SCHEMA) == []
    )


def test_stable_memory_id_is_deterministic():
    assert stable_memory_id("trace-x") == stable_memory_id("trace-x")
    assert stable_memory_id("trace-x") != stable_memory_id("trace-y")
    assert stable_memory_id("trace-x") < 2**64


def test_smk_unknown_enum_discriminants_raise_value_error():
    """Review P1-9: bad discriminants are rejected, not silently coerced."""
    from memory_core._native import PyAssistantMemoryIndex, PySmkQuery

    index = PyAssistantMemoryIndex(DIM)
    with pytest.raises(ValueError, match="invalid topic"):
        index.add(
            id=1, topic=99, kind=1, tool_mask=0, difficulty=0, generality=0,
            importance=0, embedding=emb(1),
        )
    with pytest.raises(ValueError, match="invalid kind"):
        index.add(
            id=1, topic=1, kind=42, tool_mask=0, difficulty=0, generality=0,
            importance=0, embedding=emb(1),
        )
    with pytest.raises(ValueError, match="invalid topic"):
        PySmkQuery(topic=99)


def test_smk_dim_mismatch_raises_value_error_not_panic():
    """Review P1-8: dimension mismatches surface as ValueError, not PanicException."""
    from memory_core._native import PyAssistantMemoryIndex, PySmkQuery

    index = PyAssistantMemoryIndex(DIM)
    index.add(
        id=1, topic=1, kind=1, tool_mask=0, difficulty=0, generality=0,
        importance=0, embedding=emb(1),
    )
    with pytest.raises(ValueError, match="dimension mismatch"):
        index.add(
            id=2, topic=1, kind=1, tool_mask=0, difficulty=0, generality=0,
            importance=0, embedding=[0.1, 0.2],
        )
    with pytest.raises(ValueError, match="dimension mismatch"):
        index.query_top_k_filtered([0.1, 0.2], 5, PySmkQuery())
