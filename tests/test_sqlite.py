"""The SQLite database: file paths, in-memory sharing across worker threads, isolation."""

from __future__ import annotations

import asyncio
import os

from memory_core import Database, EmbeddingSpec, MemoryService
from memory_core.memory.models import (
    ObservationMemory,
    RecallRequest,
    RememberRequest,
    RememberResponse,
)

SPEC = EmbeddingSpec(model_id="fixture", revision="1", dimensions=3)
EMB = [0.1, 0.2, 0.3]
OWNER = "alice"
WRITES = 20


def service(path: str | os.PathLike[str]) -> MemoryService:
    return MemoryService(Database(path), embedding_spec=SPEC)


async def remember(
    memory: MemoryService, text: str, owner: str = OWNER
) -> RememberResponse:
    return await memory.call(
        RememberRequest(
            memory=ObservationMemory(summary=text, observation=text), embedding=EMB
        ),
        owner_id=owner,
    )


async def stored_ids(memory: MemoryService, owner: str = OWNER) -> set[str]:
    response = await memory.call(
        RecallRequest(query="fact", embedding=EMB, limit=100), owner_id=owner
    )
    return {hit.memory.id for hit in response.hits}


async def concurrent_writes(memory: MemoryService) -> None:
    """Writes for one owner reconcile optimistically and may conflict; writes for
    different owners run in parallel worker threads against the same database."""
    owners = [f"owner-{i}" for i in range(WRITES)]
    written = await asyncio.gather(
        *(remember(memory, f"fact {i}", owner) for i, owner in enumerate(owners))
    )
    for owner, response in zip(owners, written, strict=True):
        assert await stored_ids(memory, owner) == {response.memory.id}


async def test_file_path_creates_parent_dirs(tmp_path):
    target = tmp_path / "nested" / "deeper" / "memory.db"
    await remember(service(target), "stored locally")
    assert target.exists()


async def test_file_path_expands_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    await remember(service("~/.assistant/memory.db"), "under home")
    assert (tmp_path / ".assistant" / "memory.db").exists()


async def test_in_memory_database_survives_worker_threads():
    """Store calls run in worker threads; without a shared connection each thread
    would see its own empty database ("no such table")."""
    memory = service(":memory:")
    assert memory.store.db.is_memory
    await concurrent_writes(memory)


async def test_in_memory_databases_are_isolated():
    first, second = service(":memory:"), service(":memory:")
    await remember(first, "only in the first")
    assert await stored_ids(second) == set()


async def test_file_database_concurrent_writes(tmp_path):
    await concurrent_writes(service(tmp_path / "busy.db"))
