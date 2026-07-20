"""Shared fakes for MemoryCore tests (no MariaDB/Redis required)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from memory_core_py.core.models import AssistantMemoryTrace, MemoryCandidate, MemoryTrace
from memory_core_py.types.smk_types import MemoryKind, ToolFlag, TopicBucket


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
        tags=tags or {"tag-a"},
    )


class FakeIndex:
    """MemoryIndex protocol fake recording every call."""

    def __init__(self) -> None:
        self.ingested: list[tuple[MemoryTrace, list[float]]] = []
        self.searches: list[dict[str, Any]] = []
        self.results: list[MemoryCandidate] = []

    def ingest(self, trace: MemoryTrace, embedding: list[float]) -> None:
        self.ingested.append((trace, list(embedding)))

    def search(self, *, user_id, text, tags, limit, query_embedding):
        self.searches.append(
            {"user_id": user_id, "text": text, "tags": tags, "limit": limit}
        )
        return list(self.results)

    def mark_accessed(self, trace_uid: str) -> None:  # pragma: no cover - protocol
        pass

    def remove(self, trace_uid: str) -> None:  # pragma: no cover - protocol
        pass


class FakeLtmStore:
    def __init__(self) -> None:
        self.upserts: list[tuple[MemoryTrace, list[float]]] = []

    def upsert_trace(self, trace: MemoryTrace, *, embedding: list[float]) -> None:
        self.upserts.append((trace, list(embedding)))


class FakeStmStore:
    def __init__(self) -> None:
        self.inserts: list[tuple[MemoryTrace, Any]] = []
        self.recent: list[MemoryTrace] = []

    def insert_trace(self, trace: MemoryTrace, *, expires_at) -> None:
        self.inserts.append((trace, expires_at))

    def fetch_recent_for_user(self, *, user_id: str, limit: int) -> list[MemoryTrace]:
        return list(self.recent)


class FakeRedisClient:
    """Just enough of redis.asyncio for RedisWorkingMemory."""

    def __init__(self) -> None:
        self.lists: dict[str, list[bytes]] = {}
        self.ttls: dict[str, int] = {}

    async def rpush(self, key: str, value: str) -> None:
        self.lists.setdefault(key, []).append(value.encode())

    async def expire(self, key: str, ttl: int) -> None:
        self.ttls[key] = ttl

    async def lrange(self, key: str, start: int, end: int) -> list[bytes]:
        items = self.lists.get(key, [])
        end = len(items) if end == -1 else end + 1
        return items[start if start >= 0 else max(0, len(items) + start) : end]


class FakeWorkingMemory:
    """Working-memory fake for MemorySystem orchestration tests."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.recent: list[dict[str, Any]] = []

    async def add_event(self, *, user_id: str, payload: dict[str, Any]) -> None:
        self.events.append((user_id, json.loads(json.dumps(payload))))

    async def get_recent(self, *, user_id: str, limit: int = 20) -> list[dict[str, Any]]:
        return list(self.recent)


@pytest.fixture
def fake_index() -> FakeIndex:
    return FakeIndex()


@pytest.fixture
def fake_ltm() -> FakeLtmStore:
    return FakeLtmStore()


@pytest.fixture
def fake_stm() -> FakeStmStore:
    return FakeStmStore()


@pytest.fixture
def fake_wm() -> FakeWorkingMemory:
    return FakeWorkingMemory()
