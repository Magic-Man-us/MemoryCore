"""Pydantic model invariants, including the P0 regressions
(see docs/review-junovera-integration.md)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from memory_core import MemoryCandidate, MemoryTrace
from tests.conftest import make_trace


class TestMemoryTrace:
    def test_importance_bounded(self):
        """importance feeds the ranking score directly; out-of-range values must be rejected."""
        kwargs = {
            "trace_uid": "t",
            "user_id": "u",
            "content": "",
            "summary": "s",
            "created_at": datetime.now(UTC),
        }
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
        trace = make_trace(tags={"a", "b"})
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
