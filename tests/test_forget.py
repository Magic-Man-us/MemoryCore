"""Host-only hard delete: every stored copy goes, other instances follow, ids are reusable."""

from __future__ import annotations

import time
import zlib
from typing import TYPE_CHECKING

from sqlalchemy import text

from memory_core import Database, EmbeddingSpec, MemoryService
from memory_core.memory.models import (
    DreamPlanRequest,
    ObservationMemory,
    RecallRequest,
    RememberRequest,
)

if TYPE_CHECKING:
    from pathlib import Path

SPEC = EmbeddingSpec(model_id="fixture", revision="1", dimensions=2)
VECTOR = [1.0, 0.0]
OWNER = "alice"
SECRET = "the vault code is 4417"


def service(path: Path) -> MemoryService:
    return MemoryService(Database(path), embedding_spec=SPEC, dream_after_writes=1000)


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
