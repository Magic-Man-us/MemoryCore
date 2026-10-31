"""Growth-loop regressions: finite scans, restart, review, and warm cache reuse."""

import pytest
from sqlalchemy import text

from memory_core import EmbeddingSpec, MemoryCore
from memory_core.memory.models import (
    DreamChange,
    DreamProposal,
    ObservationMemory,
    RecallRequest,
    RememberRequest,
    Supersede,
)
from memory_core.runtime import MaintenancePolicy


class Embedder:
    async def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


def runtime(path, planner=None):
    return MemoryCore.local(
        path,
        embedding_spec=EmbeddingSpec(model_id="growth", revision="1", dimensions=2),
        embedder=Embedder(),
        dream_planner=planner,
        maintenance=MaintenancePolicy(
            background=False, dream_after_writes=100, retry_seconds=0
        ),
    )


async def write(memory, start, count):
    for i in range(start, start + count):
        await memory.call(
            RememberRequest(
                memory=ObservationMemory(
                    id=f"m{i:06}",
                    summary=f"Observation {i}",
                    observation=f"Fact {i}",
                )
            ),
            owner_id="alice",
        )


async def drain(memory, limit=30):
    passes = 0
    while await memory.worker.run_once():
        passes += 1
        assert passes < limit, "maintenance failed to reach idle"
    return passes


async def test_scan_drains_after_writes_stop_and_retains_cache(tmp_path):
    seen = []

    async def planner(groups):
        seen.extend(g.memory.id for g in groups)
        return DreamProposal()

    memory = runtime(tmp_path / "memory.db", planner)
    try:
        await write(memory, 0, 100)
        cache = memory.service._caches["alice"]
        assert await drain(memory) == 7
        assert len(seen) == len(set(seen)) == 100
        assert memory.service._caches["alice"] is cache
        await memory.call(RecallRequest(query="fact"), owner_id="alice")
        assert memory.service._caches["alice"] is cache
    finally:
        memory.service.store.db.engine.dispose()


async def test_scan_resumes_after_restart_without_revisiting_batch(tmp_path):
    seen = []

    async def planner(groups):
        seen.extend(g.memory.id for g in groups)
        return DreamProposal()

    path = tmp_path / "restart.db"
    first = runtime(path, planner)
    await write(first, 0, 100)
    assert await first.worker.run_once()
    first.service.store.db.engine.dispose()
    second = runtime(path, planner)
    try:
        assert await drain(second) == 6
        assert len(seen) == len(set(seen)) == 100
    finally:
        second.service.store.db.engine.dispose()


async def test_review_handoff_does_not_block_remaining_batches(tmp_path):
    seen = []

    async def planner(groups):
        seen.extend(g.memory.id for g in groups)
        g = groups[0]
        replacement = ObservationMemory(
            summary="Proposed summary", observation="Proposed fact"
        )
        return DreamProposal(
            changes=[
                DreamChange(
                    memory=replacement,
                    decision=Supersede(
                        target_id=g.memory.id, expected_revision=g.memory.revision
                    ),
                )
            ]
        )

    memory = runtime(tmp_path / "review.db", planner)
    try:
        await write(memory, 0, 100)
        assert await drain(memory) == 7
        assert len(set(seen)) == 100
        with memory.queue.transaction("alice") as session:
            assert (
                session.execute(
                    text("SELECT COUNT(*) FROM memory_jobs WHERE status='proposed'")
                ).scalar()
                == 7
            )
            assert (
                session.execute(
                    text(
                        "SELECT COUNT(*) FROM memories_v1 WHERE json_extract(payload,'$.status')='active'"
                    )
                ).scalar()
                == 100
            )
    finally:
        memory.service.store.db.engine.dispose()


async def test_writes_during_pass_trigger_another_finite_scan(tmp_path):
    seen = []

    async def planner(groups):
        seen.extend(g.memory.id for g in groups)
        return DreamProposal()

    memory = runtime(tmp_path / "writes.db", planner)
    try:
        await write(memory, 0, 100)
        assert await memory.worker.run_once()
        await write(memory, 100, 100)
        assert await drain(memory) == 13
        assert len(seen) == len(set(seen)) == 200
    finally:
        memory.service.store.db.engine.dispose()


async def test_retired_records_leave_cache_but_remain_queryable(tmp_path):
    from memory_core.memory.errors import ConflictError
    from memory_core.memory.models import Status

    memory = runtime(tmp_path / "history.db")
    try:
        await write(memory, 0, 1)
        original = (
            (await memory.call(RecallRequest(query="fact"), owner_id="alice"))
            .hits[0]
            .memory
        )
        state = await memory.measurements(owner_id="alice")
        assert state.cached_records == 1
        with memory.queue.transaction("alice") as session:
            generation = memory.service.store.state(session, "alice")["generation"]
        await memory.call(
            RememberRequest(
                memory=ObservationMemory(
                    id="replacement", summary="Updated fact", observation="Updated"
                ),
                decision=Supersede(
                    target_id=original.id, expected_revision=original.revision
                ),
                expected_generation=generation,
            ),
            owner_id="alice",
        )
        active = await memory.call(RecallRequest(query="fact"), owner_id="alice")
        assert [h.memory.id for h in active.hits] == ["replacement"]
        assert original.id not in memory.service._caches["alice"].memories
        history = await memory.call(
            RecallRequest(query="fact", statuses=[Status.SUPERSEDED]), owner_id="alice"
        )
        assert [h.memory.id for h in history.hits] == [original.id]
        assert original.id not in memory.service._caches["alice"].memories
        with pytest.raises(ConflictError):
            await memory.call(RememberRequest(memory=original), owner_id="alice")
        measured = await memory.measurements(owner_id="alice")
        assert measured.memories_by_status == {"active": 1, "superseded": 1}
        assert measured.cached_records == 1
        assert measured.allocated_database_bytes > 0
        other = await memory.measurements(owner_id="bob")
        assert not other.memories_by_status and not other.jobs_by_status
    finally:
        memory.service.store.db.engine.dispose()


async def test_measurements_observe_backlog_without_hydration(tmp_path):
    memory = runtime(tmp_path / "metrics.db")
    try:
        await write(memory, 0, 100)
        memory.service._caches.clear()
        before = await memory.measurements(owner_id="alice")
        assert before.jobs_by_status == {"pending": 1}
        assert before.cached_records == 0
        assert before.oldest_ready_wait_seconds is not None
        assert before.records_after_dream_cursor == 100
        assert not memory.service._caches
        await drain(memory)
        after = await memory.measurements(owner_id="alice")
        assert after.jobs_by_status == {"done": 7}
        assert after.records_after_dream_cursor == 0
    finally:
        memory.service.store.db.engine.dispose()


async def test_continuation_failure_rolls_back_ack_and_cursor(tmp_path, monkeypatch):
    memory = runtime(tmp_path / "atomic.db")
    try:
        await write(memory, 0, 100)
        original = memory.queue.enqueue

        def fail(*args, **kwargs):
            raise RuntimeError("simulated continuation failure")

        monkeypatch.setattr(memory.queue, "enqueue", fail)
        assert await memory.worker.run_once()
        with memory.queue.transaction("alice") as session:
            assert memory.service.store.state(session, "alice")["dream_cursor"] == 0
            assert (
                session.execute(text("SELECT status FROM memory_jobs")).scalar_one()
                == "pending"
            )
        monkeypatch.setattr(memory.queue, "enqueue", original)
        assert await drain(memory) == 7
        measured = await memory.measurements(owner_id="alice")
        assert measured.records_after_dream_cursor == 0
        assert measured.jobs_by_status == {"done": 7}
    finally:
        memory.service.store.db.engine.dispose()


def test_binary_embeddings_preserve_float_values_and_read_legacy_json():
    import json

    from memory_core.storage.codec import decode_embedding, encode_embedding

    values = [0.1, -1.23456789012345, 1e-200, 1e100]
    encoded = encode_embedding(values)
    assert encoded.startswith(b"MCV1")
    assert len(encoded) == 4 + 8 * len(values)
    assert decode_embedding(encoded) == values
    assert decode_embedding(json.dumps(values).encode()) == values
    assert decode_embedding(b"MCV1bad") is None
    assert decode_embedding(b'["not a float"]') is None
    assert decode_embedding(b"[[1,2]]") is None


async def test_history_streams_multiple_batches_and_survives_restart(tmp_path):
    from memory_core.memory.models import Status
    from memory_core.storage.codec import encode_embedding

    path = tmp_path / "retired.db"
    memory = runtime(path)
    with memory.queue.transaction("alice") as session:
        for i in range(300):
            record = ObservationMemory(
                id=f"retired-{i:03}",
                summary=f"Retired fact {i}",
                observation="Historical",
                status=Status.SUPERSEDED,
            )
            memory.service.store.put(
                session, "alice", record, encode_embedding([1, 0]), 1
            )
        session.execute(
            text("UPDATE memory_owners SET generation=1 WHERE owner='alice'")
        )
    memory.service.store.db.engine.dispose()
    memory = runtime(path)
    try:
        result = await memory.call(
            RecallRequest(query="299", statuses=[Status.SUPERSEDED]), owner_id="alice"
        )
        assert result.hits[0].memory.id == "retired-299"
        assert not memory.service._caches["alice"].memories
        assert len(result.hits) == 10
    finally:
        memory.service.store.db.engine.dispose()


async def test_late_approval_does_not_rewind_completed_scan(tmp_path):
    from memory_core.runtime.models import ReviewJobRequest

    async def planner(groups):
        g = groups[0]
        return DreamProposal(
            changes=[
                DreamChange(
                    memory=ObservationMemory(
                        summary="Replacement", observation="Replacement"
                    ),
                    decision=Supersede(
                        target_id=g.memory.id, expected_revision=g.memory.revision
                    ),
                )
            ]
        )

    memory = runtime(tmp_path / "late-review.db", planner)
    try:
        await write(memory, 0, 100)
        await drain(memory)
        with memory.queue.transaction("alice") as session:
            before = memory.service.store.state(session, "alice")["dream_cursor"]
            job = session.execute(
                text(
                    "SELECT id FROM memory_jobs WHERE json_extract(payload,'$.scan.cursor')=0"
                )
            ).scalar_one()
        await memory.call(ReviewJobRequest(job_id=job, accept=True), owner_id="alice")
        assert await memory.worker.run_once()
        with memory.queue.transaction("alice") as session:
            assert (
                memory.service.store.state(session, "alice")["dream_cursor"] == before
            )
            assert (
                session.execute(
                    text("SELECT status FROM memory_jobs WHERE id=:id"), {"id": job}
                ).scalar_one()
                == "done"
            )
        assert not await memory.worker.run_once()
    finally:
        memory.service.store.db.engine.dispose()
