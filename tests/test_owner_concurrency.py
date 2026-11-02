import asyncio
import threading

import pytest
from sqlalchemy import event

from memory_core import EmbeddingSpec, build_memory_system
from memory_core.memory.errors import EmbeddingError
from memory_core.memory.models import (
    PreferenceMemory,
    RecallRequest,
    RememberRequest,
    Supersede,
    TaskContext,
)

SPEC = EmbeddingSpec(model_id="synthetic-local", revision="1", dimensions=2)
VECTOR = [1.0, 0.0]


def make_service(path, **options):
    return build_memory_system(sqlite_path=path, embedding_spec=SPEC, **options).service


async def add(service, owner, value="old", **kwargs):
    return await service.remember_reconciled(
        RememberRequest(
            memory=PreferenceMemory(
                id=f"{owner}-{value}",
                summary=f"Synthetic {owner} preference {value}",
                preference_key="answer.style",
                value=value,
            ),
            embedding=VECTOR,
            **kwargs,
        ),
        owner_id=owner,
    )


async def read(service, owner, operation):
    if operation == "recall":
        return await service.recall(
            RecallRequest(query="Synthetic", embedding=VECTOR, limit=5), owner_id=owner
        )
    if operation == "preferences":
        return await service.resolve_prefs(
            TaskContext(task_id="synthetic-task"), owner_id=owner
        )
    return await service.prepare_task_context(
        TaskContext(task_id="synthetic-task"), owner_id=owner
    )


@pytest.mark.parametrize("operation", ["recall", "preferences", "prepare"])
async def test_warm_cross_owner_reads_overlap_inside_sql_snapshots(
    tmp_path, monkeypatch, operation
):
    service = make_service(tmp_path / "concurrent.db")
    try:
        for owner in ("alice", "bob"):
            await add(service, owner)
        barrier = threading.Barrier(2, timeout=3)
        original = service._snapshot

        def synchronized(session, owner):
            result = original(session, owner)
            barrier.wait()
            return result

        monkeypatch.setattr(service, "_snapshot", synchronized)
        responses = await asyncio.wait_for(
            asyncio.gather(
                read(service, "alice", operation), read(service, "bob", operation)
            ),
            5,
        )
        assert len(responses) == 2
    finally:
        service.store.db.dispose()


async def test_owner_lock_survives_active_cache_eviction_and_serializes_writes(
    tmp_path, monkeypatch
):
    service = make_service(tmp_path / "evict.db", max_cached_owners=1)
    entered, release, writer_entered = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    try:
        original_value = await add(service, "alice")
        await add(service, "bob")
        original_snapshot, original_lock = service._snapshot, service._owner_lock
        once = True

        def paused(session, owner):
            nonlocal once
            result = original_snapshot(session, owner)
            if owner == "alice" and once:
                once = False
                entered.set()
                assert release.wait(3)
            return result

        def observed_lock(owner):
            if owner == "alice" and entered.is_set():
                writer_entered.set()
            return original_lock(owner)

        monkeypatch.setattr(service, "_snapshot", paused)
        monkeypatch.setattr(service, "_owner_lock", observed_lock)
        first = asyncio.create_task(read(service, "alice", "preferences"))
        assert await asyncio.to_thread(entered.wait, 3)
        writer = asyncio.create_task(
            add(
                service,
                "alice",
                "new",
                decision=Supersede(
                    target_id=original_value.memory.id, expected_revision=1
                ),
                expected_generation=original_value.generation,
            )
        )
        assert await asyncio.to_thread(writer_entered.wait, 3)
        assert not writer.done()
        other = await asyncio.wait_for(read(service, "bob", "preferences"), 2)
        assert other.effective[0].value == "old"
        assert list(service._caches) == ["bob"]
        release.set()
        assert (await first).effective[0].value == "old"
        await writer
        assert (await read(service, "alice", "preferences")).effective[0].value == "new"
        assert len(service._caches) == 1
    finally:
        release.set()
        service.store.db.dispose()


async def test_read_snapshot_keeps_generation_and_lazy_payload_consistent_during_external_write(
    tmp_path, monkeypatch
):
    path = tmp_path / "snapshot.db"
    from memory_core.indexing.semantic import IndexPolicy

    first = make_service(path, index_policy=IndexPolicy(payload_cache_size=0))
    second = make_service(path)
    entered, release = threading.Event(), threading.Event()
    try:
        old = await add(first, "alice")
        original = first._snapshot

        def paused(session, owner):
            result = original(session, owner)
            entered.set()
            assert release.wait(3)
            return result

        monkeypatch.setattr(first, "_snapshot", paused)
        pending = asyncio.create_task(read(first, "alice", "preferences"))
        assert await asyncio.to_thread(entered.wait, 3)
        changed = await asyncio.wait_for(
            add(
                second,
                "alice",
                "new",
                decision=Supersede(target_id=old.memory.id, expected_revision=1),
                expected_generation=old.generation,
            ),
            2,
        )
        release.set()
        snapshot = await pending
        assert snapshot.generation == old.generation
        assert [(p.id, p.value) for p in snapshot.effective] == [(old.memory.id, "old")]
        monkeypatch.setattr(first, "_snapshot", original)
        refreshed = await read(first, "alice", "preferences")
        assert refreshed.generation == changed.generation
        assert [(p.id, p.value) for p in refreshed.effective] == [
            (changed.memory.id, "new")
        ]
    finally:
        release.set()
        first.store.db.dispose()
        second.store.db.dispose()


async def test_warmed_read_apis_emit_no_write_transactions_or_owner_seed_writes(
    tmp_path,
):
    service = make_service(tmp_path / "readonly.db")
    try:
        await add(service, "alice")
        statements = []

        def trace(conn, cursor, statement, parameters, context, executemany):
            statements.append(statement.strip().upper())

        event.listen(service.store.db.engine, "before_cursor_execute", trace)
        try:
            await read(service, "alice", "recall")
            await read(service, "alice", "preferences")
            prepared = await read(service, "alice", "prepare")
            await service.validate_context(
                prepared.receipt,
                TaskContext(task_id="synthetic-task"),
                owner_id="alice",
            )
        finally:
            event.remove(service.store.db.engine, "before_cursor_execute", trace)
        assert statements.count("BEGIN") == 4
        assert not any(
            statement.startswith(("INSERT", "UPDATE", "DELETE", "BEGIN IMMEDIATE"))
            for statement in statements
        )
    finally:
        service.store.db.dispose()


async def test_concurrent_owner_bootstrap_preserves_embedding_identity(tmp_path):
    path = tmp_path / "bootstrap.db"
    first = make_service(path)
    second = build_memory_system(
        sqlite_path=path,
        embedding_spec=EmbeddingSpec(
            model_id="different-model", revision="1", dimensions=2
        ),
    ).service
    try:
        results = await asyncio.gather(
            read(first, "new-owner", "preferences"),
            read(second, "new-owner", "preferences"),
            return_exceptions=True,
        )
        assert sum(isinstance(result, EmbeddingError) for result in results) == 1
        assert sum(not isinstance(result, BaseException) for result in results) == 1
    finally:
        first.store.db.dispose()
        second.store.db.dispose()


async def test_registry_reuses_current_generation_and_invalidates_after_field_registration(
    tmp_path, monkeypatch
):
    from memory_core.memory.models import (
        DescribeSchemaRequest,
        FieldDefinition,
        FieldType,
        RegisterFieldRequest,
    )

    service = make_service(tmp_path / "registry.db")
    try:
        await add(service, "alice")
        await read(service, "alice", "preferences")
        loads = []
        original = service.store.fields

        def fields(session, owner):
            loads.append(owner)
            return original(session, owner)

        monkeypatch.setattr(service.store, "fields", fields)
        for _ in range(5):
            await read(service, "alice", "preferences")
        assert loads == []
        await service.call(
            RegisterFieldRequest(
                definition=FieldDefinition(
                    name="editor",
                    type=FieldType.KEYWORD,
                    aliases=["preferred_editor"],
                )
            ),
            owner_id="alice",
        )
        schema = await service.call(DescribeSchemaRequest(), owner_id="alice")
        assert loads == ["alice"]
        exposed = next(field for field in schema.fields if field.name == "editor")
        exposed.aliases.append("unpersisted_alias")
        another = await service.call(DescribeSchemaRequest(), owner_id="alice")
        assert next(
            field for field in another.fields if field.name == "editor"
        ).aliases == ["preferred_editor"]
        assert loads == ["alice"]
    finally:
        service.store.db.dispose()


async def test_in_memory_reclaim_waits_for_database_owned_transaction_and_preserves_commit(
    monkeypatch,
):
    from contextlib import contextmanager

    from sqlalchemy import text

    from memory_core import MemoryCore

    service = make_service(":memory:")
    runtime = MemoryCore(service)
    entered, release, reclaim_entered = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    original_scope = service.store.db.session_scope

    def write():
        with service.store.transaction("alice") as session:
            session.execute(
                text("UPDATE memory_owners SET writes=99 WHERE owner='alice'")
            )
            entered.set()
            assert release.wait(3)

    @contextmanager
    def observed_scope():
        reclaim_entered.set()
        with original_scope() as session:
            yield session

    writer = asyncio.create_task(asyncio.to_thread(write))
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        monkeypatch.setattr(service.store.db, "session_scope", observed_scope)
        reclamation = asyncio.create_task(runtime.reclaim_storage())
        assert await asyncio.to_thread(reclaim_entered.wait, 3)
        assert not reclamation.done()
        release.set()
        await writer
        await reclamation
        with original_scope() as session:
            assert (
                session.execute(
                    text("SELECT writes FROM memory_owners WHERE owner='alice'")
                ).scalar_one()
                == 99
            )
    finally:
        release.set()
        await asyncio.gather(writer, return_exceptions=True)
        service.store.db.dispose()
