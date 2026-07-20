"""Pydantic model invariants, including the P0 regressions
(see docs/review-junovera-integration.md)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from memory_core import (
    AssistantMemoryTrace,
    MemoryCandidate,
    MemoryKind,
    MemoryTrace,
    ToolFlag,
    TopicBucket,
)
from tests.conftest import make_assistant_trace, make_trace


class TestAssistantMemoryTrace:
    def test_instantiates(self):
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

    @pytest.mark.parametrize(
        "field", ["context_complexity", "before_state_confusion", "generality", "importance"]
    )
    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_unit_interval_fields_are_bounded(self, field: str, bad: float):
        with pytest.raises(ValidationError):
            make_assistant_trace(**{field: bad})

    def test_defaults(self):
        trace = make_assistant_trace()
        assert trace.access_count == 0
        assert trace.smk_raw is None
        assert trace.created_at.tzinfo is not None


class TestMemoryTrace:
    def test_importance_bounded(self):
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

    def test_to_rust_args_order_is_stable(self):
        """The Rust FFI depends on this exact positional order — guard it."""
        created = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
        trace = make_trace(tags={"one"})
        trace.created_at = created
        trace.access_count = 7

        args = trace.to_rust_args()
        assert args == (
            trace.trace_uid,
            trace.user_id,
            trace.summary,
            0.5,
            int(created.timestamp()),
            7,
            ["one"],
        )
        assert isinstance(args[3], float)
        assert isinstance(args[4], int)
        assert isinstance(args[6], list)

    def test_tags_deduplicate_as_a_set(self):
        trace = make_trace(tags={"a", "a", "b"})
        assert trace.tags == {"a", "b"}


class TestMemoryCandidate:
    def test_is_frozen(self):
        candidate = MemoryCandidate(
            trace_uid="t",
            score=0.9,
            summary="s",
            tags=[],
            created_at=datetime.now(UTC),
        )
        with pytest.raises(ValidationError):
            candidate.score = 0.1  # type: ignore[misc]
