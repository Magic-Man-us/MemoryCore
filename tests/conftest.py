"""Shared fixtures and factories: a temp SQLite database per test, trace builders, a fake embedder."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from memory_core import AssistantMemoryTrace, MemoryKind, MemoryTrace, ToolFlag, TopicBucket
from memory_core.storage import Database, DatabaseSettings


def make_assistant_trace(**overrides) -> AssistantMemoryTrace:
    defaults = dict(
        trace_uid="a-1",
        context_topic=TopicBucket.RUST_PYTHON_TOOLCHAIN,
        context_activity="debugging",
        context_complexity=0.5,
        kind=MemoryKind.PATTERN,
        summary="maturin builds need the venv python",
        rationale="PATH isolation",
        before_state_confusion=0.8,
        after_state_confidence=0.9,
        generality=0.6,
        tools={ToolFlag.PY, ToolFlag.MATURIN},
        importance=0.9,
    )
    defaults.update(overrides)
    return AssistantMemoryTrace(**defaults)


def make_trace(
    *,
    trace_uid: str = "t-1",
    user_id: str = "user-1",
    summary: str = "a summary",
    content: str = "full content",
    importance: float = 0.5,
    tags: set[str] | None = None,
) -> MemoryTrace:
    return MemoryTrace(
        trace_uid=trace_uid,
        user_id=user_id,
        content=content,
        summary=summary,
        importance=importance,
        created_at=datetime.now(UTC),
        access_count=0,
        tags=tags if tags is not None else {"tag-a"},
    )


@pytest.fixture
def db(tmp_path) -> Database:
    database = Database(settings=DatabaseSettings(url=f"sqlite:///{tmp_path}/test.db"))
    database.create_schema()
    return database


class FakeEmbedder:
    """Deterministic embedder: 8-dim vector derived from the text bytes."""

    dim = 8

    async def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            data = text.encode()
            out.append(
                [
                    ((data[i % max(len(data), 1)] if data else 0) % 97) / 97.0 + 0.01
                    for i in range(self.dim)
                ]
            )
        return out


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()
