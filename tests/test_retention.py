"""Retention must preserve provenance, ownership and explicit replay semantics."""

import time
import zlib

import pytest
from sqlalchemy import text

from memory_core import EmbeddingSpec, MemoryCore
from memory_core.indexing.semantic import IndexPolicy
from memory_core.memory.errors import ConflictError
from memory_core.memory.models import (
    ObservationMemory,
    RecallRequest,
    RememberRequest,
    Status,
    Supersede,
    TaskContext,
)
from memory_core.runtime import (
    Evidence,
    Finding,
    MaintenancePolicy,
    RetentionPolicy,
    TaskOutcome,
)

pytest.importorskip("hnswlib")


class Embedder:
    async def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


def runtime(path):
    return MemoryCore.local(
        path,
        embedding_spec=EmbeddingSpec(model_id="test", revision="1", dimensions=2),
        embedder=Embedder(),
        index_policy=IndexPolicy(backend="hnsw"),
        maintenance=MaintenancePolicy(background=False),
        retention=RetentionPolicy(
            enabled=True,
            superseded_after_days=0,
            completed_job_days=1,
            event_replay_days=1,
            batch_size=2,
        ),
    )


async def test_archive_preserves_history_and_rejects_id_reuse(tmp_path):
    path = tmp_path / "memory.db"
    memory = runtime(path)
    old = ObservationMemory(id="old", summary="Old fact", observation="Old fact")
    created = await memory.call(RememberRequest(memory=old), owner_id="alice")
    await memory.service.checkpoint_indexes(force=True)
    await memory.call(
        RememberRequest(
            memory=ObservationMemory(
                id="new", summary="New fact", observation="New fact"
            ),
            decision=Supersede(target_id="old", expected_revision=1),
            expected_generation=created.generation,
        ),
        owner_id="alice",
    )
    assert (await memory.maintain(owner_id="bob")).archived_memories == 0
    assert (await memory.maintain(owner_id="alice")).archived_memories == 1
    assert not (await memory.maintain(owner_id="alice")).archived_memories
    with memory.queue.transaction("alice") as s:
        raw = s.execute(
            text("SELECT payload FROM memory_archives WHERE owner='alice'")
        ).scalar_one()
        assert "Old fact" in zlib.decompress(raw).decode()
    memory.service.store.db.engine.dispose()
    memory = runtime(path)
    assert [
        h.memory.id
        for h in (await memory.call(RecallRequest(query="fact"), owner_id="alice")).hits
    ] == ["new"]
    historical = await memory.call(
        RecallRequest(query="fact", statuses=[Status.SUPERSEDED]), owner_id="alice"
    )
    assert historical.hits[0].memory.id == "old"
    with pytest.raises(ConflictError):
        await memory.call(RememberRequest(memory=old), owner_id="alice")
    assert (await memory.measurements(owner_id="alice")).archived_records == 1
    memory.service.store.db.engine.dispose()


async def test_replay_window_and_job_compaction(tmp_path):
    memory = runtime(tmp_path / "events.db")
    task = memory.task(
        owner_id="alice", context=TaskContext(task_id="t"), objective="Learn a fact"
    )
    await task.__aenter__()
    await task.prepare()
    finding = Finding(
        statement="A supported fact",
        evidence=[
            Evidence(id="source", uri="local:source", excerpt="A supported fact")
        ],
    )
    await task.record_finding(finding, event_id="event")
    await task.complete(TaskOutcome(status="succeeded", summary="Learned"))
    while await memory.worker.run_once():
        pass
    with memory.queue.transaction("alice") as s:
        s.execute(
            text("UPDATE memory_job_retention SET completed_at=:t"),
            {"t": time.time() - 3 * 86400},
        )
    result = await memory.maintain(owner_id="alice")
    assert result.compacted_jobs > 0 and result.archived_events > 0
    with pytest.raises(ConflictError, match="replay window expired"):
        await task.record_finding(finding, event_id="event")
    with memory.queue.transaction("alice") as s:
        assert (
            s.execute(text("SELECT COUNT(*) FROM memory_event_archives")).scalar() > 0
        )
        assert not s.execute(
            text("SELECT COUNT(*) FROM memory_jobs WHERE payload!='{}'")
        ).scalar()
    memory.service.store.db.engine.dispose()


async def test_retention_disabled_and_bounded_batches(tmp_path):
    memory = runtime(tmp_path / "bounded.db")
    for i in range(5):
        created = await memory.call(
            RememberRequest(
                memory=ObservationMemory(
                    id=f"old{i}", summary=f"Old {i}", observation=f"Old {i}"
                )
            ),
            owner_id="alice",
        )
        await memory.call(
            RememberRequest(
                memory=ObservationMemory(
                    id=f"new{i}", summary=f"New {i}", observation=f"New {i}"
                ),
                decision=Supersede(target_id=f"old{i}", expected_revision=1),
                expected_generation=created.generation,
            ),
            owner_id="alice",
        )
    enabled = memory.retention
    memory.retention = RetentionPolicy()
    assert (await memory.maintain(owner_id="alice")).archived_memories == 0
    memory.retention = enabled
    assert (await memory.maintain(owner_id="alice")).archived_memories == 2
    assert (await memory.maintain(owner_id="alice")).archived_memories == 2
    assert (await memory.maintain(owner_id="alice")).archived_memories == 1
    memory.service.store.db.engine.dispose()


async def test_applied_plan_archive_preserves_replay_guard(tmp_path):
    memory = runtime(tmp_path / "plans.db")
    await memory.call(
        RememberRequest(memory=ObservationMemory(summary="Fact", observation="Fact")),
        owner_id="alice",
    )
    with memory.queue.transaction("alice") as s:
        memory.queue.enqueue(s, "alice", "dream", {})
    while await memory.worker.run_once():
        pass
    with memory.queue.transaction("alice") as s:
        plan_id = s.execute(text("SELECT id FROM memory_dreams")).scalar_one()
        s.execute(
            text("UPDATE memory_job_retention SET completed_at=:t"),
            {"t": time.time() - 3 * 86400},
        )
    assert (await memory.maintain(owner_id="alice")).archived_plans == 1
    with pytest.raises(ConflictError, match="already applied"):
        await memory.service.apply_dream(plan_id, owner_id="alice")
    memory.service.store.db.engine.dispose()
