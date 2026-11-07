"""Host-only hard delete: every stored copy goes, other instances follow, ids are reusable."""

from __future__ import annotations

import time
import zlib
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import text

from memory_core import Database, EmbeddingSpec, MemoryService
from memory_core.indexing.semantic import GraphManifest, IndexBackend, IndexPolicy
from memory_core.memory.models import (
    DreamPlanRequest,
    ObservationMemory,
    PreferenceMemory,
    RecallRequest,
    RememberRequest,
)

if TYPE_CHECKING:
    from pathlib import Path

SPEC = EmbeddingSpec(model_id="fixture", revision="1", dimensions=2)
VECTOR = [1.0, 0.0]
OWNER = "alice"
SECRET = "the vault code is 4417"


def service(path: Path, backend: IndexBackend = IndexBackend.EXACT) -> MemoryService:
    return MemoryService(
        Database(path),
        embedding_spec=SPEC,
        dream_after_writes=1000,
        index_policy=IndexPolicy(backend=backend),
    )


async def remember(memory: MemoryService, uid: str, content: str) -> None:
    await memory.call(
        RememberRequest(
            memory=ObservationMemory(id=uid, summary=content, observation=content),
            embedding=VECTOR,
        ),
        owner_id=OWNER,
    )


async def recalled(memory: MemoryService) -> set[str]:
    response = await memory.call(
        RecallRequest(query="vault", embedding=VECTOR, limit=100), owner_id=OWNER
    )
    return {hit.memory.id for hit in response.hits}


def stored_text(memory: MemoryService) -> str:
    """Every payload copy the database holds, decompressed."""
    chunks: list[str] = []
    with memory.store.db.session_scope() as session:
        for table in (
            "memories_v1",
            "memory_archives",
            "memory_dreams",
            "memory_dream_archives",
        ):
            for (payload,) in session.execute(text(f"SELECT payload FROM {table}")):
                raw = (
                    zlib.decompress(payload) if isinstance(payload, bytes) else payload
                )
                chunks.append(raw.decode() if isinstance(raw, bytes) else raw)
    return "\n".join(chunks)


async def test_forget_removes_the_record_and_reports_unknown_ids(tmp_path):
    memory = service(tmp_path / "memory.db")
    await remember(memory, "secret", SECRET)
    await remember(memory, "other", "an unrelated vault fact")
    assert await memory.forget("secret", owner_id=OWNER)
    assert await recalled(memory) == {"other"}
    assert not await memory.forget("secret", owner_id=OWNER)
    assert not await memory.forget("missing", owner_id=OWNER)


async def test_forget_removes_every_stored_copy_and_reaches_other_instances(tmp_path):
    path = tmp_path / "memory.db"
    first, second = service(path), service(path)
    await remember(first, "secret", SECRET)
    await remember(first, "other", "an unrelated vault fact")
    assert "secret" in await recalled(second)
    planned = await first.call(DreamPlanRequest(batch_size=8), owner_id=OWNER)
    assert any(group.memory.id == "secret" for group in planned.plan.groups)
    with first.store.db.session_scope() as session:
        payload = session.execute(
            text("SELECT payload FROM memories_v1 WHERE id='secret'")
        ).scalar_one()
        session.execute(
            text("INSERT INTO memory_archives VALUES (:o,'secret',0,:p,NULL,:t)"),
            {"o": OWNER, "p": zlib.compress(payload.encode()), "t": time.time()},
        )
        session.commit()
    assert SECRET in stored_text(first)

    assert await first.forget("secret", owner_id=OWNER)

    assert SECRET not in stored_text(first)
    assert await recalled(second) == {"other"}


async def test_forgotten_id_can_be_remembered_again(tmp_path):
    path = tmp_path / "memory.db"
    first, second = service(path), service(path)
    await remember(first, "secret", SECRET)
    assert "secret" in await recalled(second)
    await first.forget("secret", owner_id=OWNER)
    await remember(first, "secret", "a new vault fact")
    assert "secret" in await recalled(first)
    assert "secret" in await recalled(second)


async def test_forget_keeps_plans_whose_json_merely_contains_the_id(tmp_path):
    memory = service(tmp_path / "memory.db")
    await memory.call(
        RememberRequest(
            memory=PreferenceMemory(
                id="pref",
                summary="vault preference",
                preference_key="vault",
                value={"id": "secret"},
            ),
            embedding=VECTOR,
        ),
        owner_id=OWNER,
    )
    planned = await memory.call(DreamPlanRequest(batch_size=8), owner_id=OWNER)
    await remember(memory, "secret", SECRET)
    assert await memory.forget("secret", owner_id=OWNER)
    with memory.store.db.session_scope() as session:
        kept = session.execute(text("SELECT id FROM memory_dreams")).scalars().all()
    assert kept == [planned.plan.id]


def saved_labels(memory: MemoryService) -> set[str]:
    with memory.store.db.session_scope() as session:
        manifest = session.execute(
            text("SELECT manifest FROM memory_vector_snapshots")
        ).scalar_one()
    return set(GraphManifest.model_validate_json(manifest).labels)


async def test_checkpoint_after_forget_drops_the_vector_in_every_instance(tmp_path):
    pytest.importorskip("hnswlib")
    path = tmp_path / "memory.db"
    first, second = service(path, IndexBackend.HNSW), service(path, IndexBackend.HNSW)
    await remember(first, "secret", SECRET)
    await remember(first, "other", "an unrelated vault fact")
    assert "secret" in await recalled(second)
    assert await first.forget("secret", owner_id=OWNER)
    for memory in (first, second):
        assert await memory.checkpoint_indexes(force=True) == 1
        assert saved_labels(memory) == {"other"}


async def test_checkpoint_read_before_a_forget_is_not_saved(tmp_path, monkeypatch):
    pytest.importorskip("hnswlib")
    path = tmp_path / "memory.db"
    first, second = service(path, IndexBackend.HNSW), service(path, IndexBackend.HNSW)
    await remember(first, "secret", SECRET)
    assert "secret" in await recalled(second)
    vectors = second._caches[OWNER].vectors
    snapshot = vectors.snapshot

    def snapshot_then_forget() -> tuple[str, bytes]:
        saved = snapshot()
        with first.store.transaction(OWNER) as session:
            first.store.forget(session, OWNER, "secret")
        return saved

    monkeypatch.setattr(vectors, "snapshot", snapshot_then_forget)
    assert await second.checkpoint_indexes(force=True) == 0
    with second.store.db.session_scope() as session:
        assert not session.execute(
            text("SELECT 1 FROM memory_vector_snapshots")
        ).first()


async def test_record_recreated_after_forget_survives_a_restored_checkpoint(tmp_path):
    pytest.importorskip("hnswlib")
    path = tmp_path / "memory.db"
    memory = service(path, IndexBackend.HNSW)
    await remember(memory, "secret", SECRET)
    await memory.forget("secret", owner_id=OWNER)
    await remember(memory, "secret", "a new vault fact")
    assert await memory.checkpoint_indexes(force=True) == 1
    reopened = service(path, IndexBackend.HNSW)
    assert "secret" in await recalled(reopened)
    assert reopened._caches[OWNER].vectors.active == {"secret"}
