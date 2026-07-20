"""Smoke tests for the Rust extension bindings (memory_core._native)."""

from __future__ import annotations

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
