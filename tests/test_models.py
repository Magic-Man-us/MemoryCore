"""Regression tests for the P0 model bugs (see docs/review-junovera-integration.md)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from memory_core import AssistantMemoryTrace, MemoryKind, MemoryTrace, ToolFlag, TopicBucket


def test_assistant_memory_trace_instantiates():
    """P0-2: the SMK enums must be importable at runtime for Pydantic to build the model."""
    trace = AssistantMemoryTrace(
        trace_uid="t1",
        context_topic=TopicBucket.RUST_PYTHON_TOOLCHAIN,
        context_activity="debugging",
        context_complexity=0.5,
        kind=MemoryKind.PATTERN,
        tools={ToolFlag.RS, ToolFlag.PY},
        summary="s",
        rationale="r",
        before_state_confusion=0.2,
        after_state_confidence=0.9,
        generality=0.5,
        importance=0.8,
    )
    assert trace.context_topic is TopicBucket.RUST_PYTHON_TOOLCHAIN
    assert ToolFlag.RS in trace.tools


def test_memory_trace_importance_bounded():
    """importance feeds the ranking score directly; out-of-range values must be rejected."""
    kwargs = dict(
        trace_uid="t",
        user_id="u",
        content="",
        summary="s",
        created_at=datetime.now(UTC),
    )
    with pytest.raises(ValidationError):
        MemoryTrace(importance=7.3, **kwargs)
    with pytest.raises(ValidationError):
        MemoryTrace(importance=-0.1, **kwargs)
    assert MemoryTrace(importance=1.0, **kwargs).importance == 1.0


def test_to_rust_args_order():
    trace = MemoryTrace(
        trace_uid="t",
        user_id="u",
        content="c",
        summary="s",
        importance=0.5,
        created_at=datetime.now(UTC),
        access_count=3,
        tags={"a"},
    )
    uid, user, summary, importance, ts, access, tags = trace.to_rust_args()
    assert (uid, user, summary, importance, access, tags) == ("t", "u", "s", 0.5, 3, ["a"])
    assert isinstance(ts, int)
