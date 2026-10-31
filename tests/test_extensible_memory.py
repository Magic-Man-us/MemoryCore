"""Behavior tests for the extensible local memory service and automatic context hook."""

import asyncio
import json

import pytest
from memory_core._native import PyMemoryEngine, PyVectorIndex
from pydantic import ValidationError
from sqlalchemy import text

from memory_core import EmbeddingSpec, build_memory_system
from memory_core.memory.errors import (
    ConflictError,
    ContextRequiredError,
    EmbeddingError,
    SchemaError,
)
from memory_core.memory.models import (
    REQUEST_ADAPTER,
    RESPONSE_ADAPTER,
    All,
    AnyOf,
    AttachEvidence,
    ContainsAll,
    ContainsAny,
    Create,
    DescribeSchemaRequest,
    DreamApplyRequest,
    DreamPlanRequest,
    Eq,
    Exists,
    FieldDefinition,
    FieldType,
    FlagConflict,
    In,
    ListValuesRequest,
    ObservationMemory,
    PreferenceMemory,
    PreserveException,
    ProjectScope,
    Range,
    RecallRequest,
    RegisterFieldRequest,
    RememberRequest,
    Status,
    Supersede,
    TaskContext,
)

SPEC = EmbeddingSpec(model_id="test-local", revision="1", dimensions=2)
V = [1.0, 0.0]


class LocalEmbedder:
    async def embed(self, texts):
        return [V[:] for _ in texts]


@pytest.fixture
def service(tmp_path):
    return build_memory_system(
        sqlite_path=tmp_path / "memory.db",
        embedding_spec=SPEC,
        embedder=LocalEmbedder(),
    ).service


def preference(**overrides):
    return PreferenceMemory(
        **dict(
            {
                "summary": "Use uv",
                "preference_key": "python.package_manager",
                "value": "uv",
                "applicability": {"languages": ["python"]},
                "metadata": {"languages": ["python"]},
            },
            **overrides,
        )
    )


async def remember(service, memory, owner="alice", **kw):
    return await service.call(
        RememberRequest(memory=memory, embedding=V, **kw), owner_id=owner
    )


async def recall(service, owner="alice", **kw):
    return await service.call(
        RecallRequest(query="dependencies", embedding=V, **kw), owner_id=owner
    )


async def register(service, name, type=FieldType.KEYWORD, aliases=None):
    return await service.call(
        RegisterFieldRequest(
            definition=FieldDefinition(name=name, type=type, aliases=aliases or [])
        ),
        owner_id="alice",
    )


async def test_union_roundtrip_schema_and_no_domain_enum(service):
    req = REQUEST_ADAPTER.validate_json(
        '{"operation":"resolve_prefs","context":{"task_id":"a","attributes":{"languages":["brand_new_language"]}}}'
    )
    result = await service.call(req, owner_id="alice")
    assert result.effective == [] and result.complete
    assert RESPONSE_ADAPTER.validate_json(result.model_dump_json()) == result
    schema = await service.call(DescribeSchemaRequest(), owner_id="alice")
    assert schema.request_schema["discriminator"]["propertyName"] == "operation"
    assert schema.memory_schema["discriminator"]["propertyName"] == "kind"
    with pytest.raises(ValidationError):
        REQUEST_ADAPTER.validate_python(
            {"operation": "recall", "query": "q", "owner_id": "bob"}
        )


async def test_unlimited_domain_values_and_aliases(service):
    await register(service, "editor", aliases=["ide"])
    result = await remember(
        service,
        preference(
            metadata={"tools": [f"new_tool_{i}" for i in range(300)], "ide": "custom"}
        ),
    )
    assert result.memory.metadata["editor"] == "custom"
    hits = await recall(
        service, filter=ContainsAny(field="tools", values=["new_tool_299"])
    )
    assert len(hits.hits) == 1
    values = await service.call(
        ListValuesRequest(field="tools", prefix="new_tool_", limit=100),
        owner_id="alice",
    )
    assert len(values.values) == 100 and values.next_offset == 100
    with pytest.raises(SchemaError):
        await register(service, "editor", FieldType.NUMBER)
    with pytest.raises(SchemaError):
        await register(service, "other", aliases=["ide"])


async def test_filter_types_empty_missing_and_boolean_semantics(service):
    await register(service, "priority", FieldType.NUMBER)
    await register(service, "enabled", FieldType.BOOLEAN)
    await remember(
        service,
        preference(id="a", metadata={"priority": 2, "enabled": True, "tools": []}),
    )
    await remember(
        service,
        preference(
            id="b",
            value="pip",
            metadata={"priority": 5, "enabled": False, "tools": ["x"]},
        ),
    )
    await remember(service, preference(id="c", value="other", metadata={}))

    async def ids(f):
        return {h.memory.id for h in (await recall(service, filter=f)).hits}

    assert await ids(Range(field="priority", minimum=3)) == {"b"}
    assert await ids(Eq(field="enabled", value=True)) == {"a"}
    assert await ids(ContainsAll(field="tools", values=[])) == {"a", "b"}
    assert await ids(ContainsAny(field="tools", values=[])) == set()
    assert await ids(Exists(field="priority")) == {"a", "b"}
    assert await ids(In(field="priority", values=[2, 5])) == {"a", "b"}
    assert await ids(All(conditions=[])) == {"a", "b", "c"}
    assert await ids(AnyOf(conditions=[])) == set()
    assert await ids(
        All(conditions=[Exists(field="tools"), Range(field="priority", maximum=3)])
    ) == {"a"}
    for bad in [
        Eq(field="enabled", value=1),
        ContainsAny(field="priority", values=["2"]),
        Eq(field="missing", value="x"),
    ]:
        with pytest.raises(SchemaError):
            await ids(bad)


async def test_owner_scope_restart_and_model_identity(service):
    await remember(service, preference(id="shared"))
    await remember(service, preference(id="shared", value="bob value"), owner="bob")
    assert (await recall(service)).hits[0].memory.value == "uv"
    assert (await recall(service, "bob")).hits[0].memory.value == "bob value"
    from memory_core.memory.service import MemoryService

    restarted = MemoryService(service.store.db, embedding_spec=SPEC)
    assert (await recall(restarted)).hits[0].memory.id == "shared"
    incompatible = MemoryService(
        service.store.db, embedding_spec=SPEC.model_copy(update={"revision": "2"})
    )
    with pytest.raises(EmbeddingError):
        await recall(incompatible)
    with pytest.raises(SchemaError):
        await remember(service, preference(metadata={"unregistered": "x"}))


async def test_preferences_complete_project_exception_and_conflict(service):
    for i in range(12):
        await remember(service, preference(preference_key=f"rule_{i}", value=i))
    context = TaskContext(task_id="coding", attributes={"languages": ["python"]})
    assert len((await service.resolve_prefs(context, owner_id="alice")).effective) == 12
    assert not (
        await service.resolve_prefs(
            context.model_copy(update={"attributes": {}}), owner_id="alice"
        )
    ).effective
    global_rule = await remember(service, preference())
    special = preference(value="pip", scope=ProjectScope(project_id="legacy"))
    await remember(
        service,
        special,
        decision=PreserveException(
            target_id=global_rule.memory.id, expected_revision=1
        ),
        expected_generation=global_rule.generation,
    )
    selected = await service.resolve_prefs(
        context.model_copy(update={"project_id": "legacy"}), owner_id="alice"
    )
    assert [
        m.value
        for m in selected.effective
        if m.preference_key == "python.package_manager"
    ] == ["pip"]
    await remember(service, preference(value="poetry"))
    assert len((await service.resolve_prefs(context, owner_id="alice")).conflicts) == 1


async def test_reconcile_duplicate_supersede_and_stale_revision(service):
    first = await remember(service, preference(source_refs=["one"]))
    duplicate = await remember(service, preference(source_refs=["two"]))
    assert isinstance(duplicate.decision, AttachEvidence)
    assert duplicate.memory.id == first.memory.id
    assert duplicate.memory.source_refs == ["one", "two"]
    replacement = await remember(
        service,
        preference(value="pip"),
        decision=Supersede(target_id=first.memory.id, expected_revision=2),
        expected_generation=duplicate.generation,
    )
    assert replacement.memory.supersedes_id == first.memory.id
    assert len((await recall(service)).hits) == 1
    with pytest.raises(ConflictError):
        await remember(
            service,
            preference(value="other"),
            decision=Supersede(target_id=replacement.memory.id, expected_revision=1),
            expected_generation=0,
        )
    with pytest.raises(ConflictError):
        await remember(
            service,
            preference(value="other"),
            decision=Supersede(target_id="bob-id", expected_revision=1),
            expected_generation=replacement.generation,
        )


async def test_flag_conflict_and_scope_guard(service):
    a = await remember(service, preference())
    with pytest.raises(SchemaError):
        await remember(
            service,
            preference(scope=ProjectScope(project_id="p")),
            decision=Supersede(target_id=a.memory.id, expected_revision=1),
            expected_generation=a.generation,
        )
    b = await remember(
        service,
        preference(value="pip"),
        decision=FlagConflict(target_id=a.memory.id, expected_revision=1),
        expected_generation=a.generation,
    )
    assert b.memory.status == Status.DISPUTED
    result = await service.resolve_prefs(
        TaskContext(task_id="x", attributes={"languages": ["python"]}), owner_id="alice"
    )
    assert len(result.conflicts[0].memories) == 2


async def test_context_hook_and_refresh(service):
    await remember(service, preference())
    ctx = TaskContext(task_id="code", attributes={"languages": ["python"]})

    async def action(prepared):
        assert prepared.preferences.effective[0].value == "uv"
        await service.validate_context(prepared.receipt, ctx, owner_id="alice")
        return prepared

    prepared = await service.run_with_context(ctx, action, owner_id="alice")
    with pytest.raises(ContextRequiredError):
        await service.validate_context(prepared.receipt, ctx, owner_id="bob")
    with pytest.raises(ContextRequiredError):
        await service.validate_context(
            prepared.receipt,
            ctx.model_copy(update={"attributes": {"languages": ["rust"]}}),
            owner_id="alice",
        )
    await remember(service, preference(preference_key="other", value=1))
    with pytest.raises(ContextRequiredError):
        await service.validate_context(prepared.receipt, ctx, owner_id="alice")
    fresh = await service.refresh_task_context(ctx, owner_id="alice")
    await service.validate_context(fresh.receipt, ctx, owner_id="alice")


async def test_cross_instance_updates_and_reconcile_race(service):
    from memory_core.memory.service import MemoryService

    other = MemoryService(service.store.db, embedding_spec=SPEC)
    assert not (await recall(other)).hits
    await remember(service, preference())
    assert len((await recall(other)).hits) == 1
    entered = asyncio.Event()
    release = asyncio.Event()

    async def reconcile(memory, hits):
        entered.set()
        await release.wait()
        return {"action": "create"}

    service.reconciler = reconcile
    pending = asyncio.create_task(remember(service, preference(value="new")))
    await entered.wait()
    await remember(other, preference(preference_key="separate", value=1))
    release.set()
    with pytest.raises(ConflictError):
        await pending
    assert len((await recall(other)).hits) == 2


async def test_failed_write_rolls_back_audit_and_record(service, monkeypatch):
    first = await remember(service, preference())

    def fail(*args):
        raise OSError("disk full")

    monkeypatch.setattr(service.store, "put", fail)
    with pytest.raises(OSError):
        await remember(service, preference(value="pip"))
    recalled = await recall(service)
    assert recalled.generation == first.generation and len(recalled.hits) == 1


async def test_embedding_failures_leave_no_records(service):
    for vector in [[1.0], [0.0, 0.0], [float("inf"), 1.0], [1e100, 1.0]]:
        with pytest.raises((EmbeddingError, ValidationError)):
            await service.call(
                RememberRequest(memory=preference(), embedding=vector), owner_id="alice"
            )
    assert not (await recall(service)).hits


async def test_dream_threshold_plan_apply_and_stale_guard(service):
    service.dream_after_writes = 2
    first = await remember(service, preference(id="a"))
    # Explicit create preserves duplicate for a maintenance pass.
    second = await remember(
        service,
        preference(id="b"),
        decision=Create(),
        expected_generation=first.generation,
    )
    assert second.dream_plan_id is not None
    assert not await service.dream_due("alice")
    planned = await service.call(DreamPlanRequest(batch_size=2), owner_id="alice")
    assert len(planned.plan.proposal.changes) == 1
    assert len((await recall(service)).hits) == 2  # planning never changes memories
    applied = await service.call(
        DreamApplyRequest(plan_id=planned.plan.id), owner_id="alice"
    )
    assert applied.plan.applied and len((await recall(service)).hits) == 1
    with pytest.raises(ConflictError):
        await service.apply_dream(planned.plan.id, owner_id="alice")
    plan2 = await service.dream(owner_id="alice")
    await remember(service, preference(value="other"))
    with pytest.raises(ConflictError):
        await service.apply_dream(plan2.plan.id, owner_id="alice")


async def test_automatic_dream_is_bounded_and_non_mutating(service):
    calls = []

    async def planner(groups):
        calls.append(groups)
        return {
            "recommendations": [
                "Evaluate a different local embedding model against the recall fixture."
            ]
        }

    service.dream_planner = planner
    service.dream_after_writes = 1
    await remember(service, preference())
    assert len(calls) == 1 and len(calls[0]) <= 16
    assert len((await recall(service)).hits) == 1
    assert not await service.dream_due("alice")


async def test_dream_failure_keeps_successful_write_explicit(service):
    async def fail(groups):
        raise RuntimeError("planner unavailable")

    service.dream_planner = fail
    service.dream_after_writes = 1
    result = await remember(service, preference())
    assert result.maintenance_error == "RuntimeError"
    assert len((await recall(service)).hits) == 1
    assert await service.dream_due("alice")


async def test_legacy_migration_preserves_data_without_guessing_ownership(service):
    from memory_core.memory.migration import import_legacy

    db = service.store.db
    payload = {
        "trace_uid": "old",
        "context_topic": 99,
        "context_activity": "coding",
        "summary": "old lesson",
        "rationale": "why",
        "tools": [32],
        "languages": ["python"],
        "extra": {"custom": "keep"},
    }
    with db.session_scope() as session:
        session.execute(
            text(
                "INSERT INTO assistant_traces(trace_uid,created_at,payload,embedding) VALUES (:id,:date,:p,:e)"
            ),
            {
                "id": "old",
                "date": "2026-01-01",
                "p": json.dumps(payload),
                "e": json.dumps(V).encode(),
            },
        )
        session.commit()
    assert await import_legacy(service, owner_id="alice") == 0
    assert (
        await import_legacy(service, owner_id="alice", include_unscoped_assistant=True)
        == 1
    )
    assert (
        await import_legacy(service, owner_id="alice", include_unscoped_assistant=True)
        == 0
    )
    hit = (await recall(service)).hits[0].memory
    assert hit.metadata["topics"] == ["legacy_topic_99"]
    with db.session_scope() as session:
        archived = session.execute(
            text("SELECT payload FROM memory_legacy_archive")
        ).scalar_one()
        assert json.loads(archived) == payload
    assert not (await recall(service, "bob")).hits


def test_semantic_match_survives_keyword_flood():
    engine = PyMemoryEngine()
    for i in range(20):
        engine.ingest_trace(
            str(i), "alice", "python dependencies", 0.5, 1, 0, [], [0.0, 1.0]
        )
    engine.ingest_trace("best", "alice", "Use uv", 0.5, 1, 0, [], V)
    hits = engine.search_candidates("alice", "python dependencies", [], 5, V)
    assert hits[0].trace_uid == "best"


def test_generic_native_index_validation_and_selected_ids():
    engine = PyVectorIndex(2)
    engine.upsert("a", V)
    engine.upsert("b", [0.0, 1.0])
    assert [uid for uid, _ in engine.search(V, ["b"], 5)] == ["b"]
    engine.upsert("b", V)
    assert engine.search(V, ["b"], 1)[0][1] == pytest.approx(1)
    assert engine.remove("b") and not engine.remove("b")
    with pytest.raises(ValueError):
        engine.upsert("bad", [0.0, 0.0])


async def test_metadata_readonly_status_and_deep_copy(service):
    first = await remember(service, preference())
    with pytest.raises(SchemaError):
        await remember(service, preference(metadata={"status": "active"}))
    response = await recall(service)
    response.hits[0].memory.metadata["languages"].append("bad")
    assert (await recall(service)).hits[0].memory.metadata["languages"] == ["python"]
    await remember(
        service,
        preference(value="pip"),
        decision=Supersede(target_id=first.memory.id, expected_revision=1),
        expected_generation=first.generation,
    )
    old = await recall(
        service,
        statuses=[Status.SUPERSEDED],
        filter=Eq(field="status", value="superseded"),
    )
    assert [h.memory.id for h in old.hits] == [first.memory.id]


async def test_dream_malicious_plan_is_atomic(service):
    a = await remember(service, preference(id="a"))
    await remember(
        service,
        preference(id="b", preference_key="other"),
        decision=Create(),
        expected_generation=a.generation,
    )
    before = await recall(service)

    async def planner(groups):
        return {
            "changes": [
                {
                    "memory": preference(id="new").model_dump(mode="json"),
                    "decision": {
                        "action": "supersede",
                        "target_id": "other-owner-id",
                        "expected_revision": 1,
                    },
                }
            ]
        }

    service.dream_planner = planner
    planned = await service.dream(owner_id="alice")
    with pytest.raises(ConflictError):
        await service.apply_dream(planned.plan.id, owner_id="alice")
    after = await recall(service)
    assert after.generation == before.generation and len(after.hits) == 2


async def test_dream_cursor_visits_records_with_equal_generations(service):
    # Bulk migration and consolidation can put multiple rows in one generation.
    for i in range(5):
        await remember(service, preference(id=f"id{i}", preference_key=f"key_{i}"))
    with service.store.transaction("alice") as session:
        session.execute(
            text("UPDATE memories_v1 SET generation=5 WHERE owner=:o"), {"o": "alice"}
        )
    seen = set()
    for _ in range(3):
        planned = await service.dream(
            owner_id="alice", request=DreamPlanRequest(batch_size=2)
        )
        seen.update(g.memory.id for g in planned.plan.groups)
        await service.apply_dream(planned.plan.id, owner_id="alice")
    assert seen == {f"id{i}" for i in range(5)}


async def test_dream_generated_replacement_and_failure_rollback(service):
    await remember(service, preference(id="source"))

    async def planner(groups):
        return {
            "changes": [
                {
                    "memory": preference(id="replacement", value="pip").model_dump(
                        mode="json"
                    ),
                    "decision": {
                        "action": "supersede",
                        "target_id": "source",
                        "expected_revision": 1,
                    },
                }
            ]
        }

    service.dream_planner = planner
    planned = await service.dream(owner_id="alice")
    await service.apply_dream(planned.plan.id, owner_id="alice")
    assert [h.memory.id for h in (await recall(service)).hits] == ["replacement"]


async def test_empty_filter_array_and_unknown_fields_do_not_widen(service):
    await remember(service, preference())
    assert not (await recall(service, statuses=[])).hits
    with pytest.raises(SchemaError):
        await recall(
            service,
            filter=AnyOf(conditions=[Exists(field="languages"), Exists(field="typo")]),
        )


async def test_same_dimension_model_change_is_rejected_before_write(service):
    from memory_core.memory.service import MemoryService

    await remember(service, preference())
    changed = MemoryService(
        service.store.db,
        embedding_spec=SPEC.model_copy(update={"model_id": "different"}),
    )
    with pytest.raises(EmbeddingError):
        await remember(changed, preference(value="pip"))


async def test_evaluation_semantic_candidate_survives_lexical_flood(service):
    from memory_core.memory.evaluation import RetrievalCase, evaluate_retrieval

    for i in range(20):
        await service.call(
            RememberRequest(
                memory=ObservationMemory(
                    summary="python dependency installation", observation=str(i)
                ),
                embedding=[0.0, 1.0],
            ),
            owner_id="alice",
        )
    await remember(service, preference(id="relevant", summary="Use uv"))
    measured = await evaluate_retrieval(
        service,
        [
            RetrievalCase(
                query="python dependency installation",
                relevant_ids={"relevant"},
                embedding=V,
            )
        ],
        owner_id="alice",
        k=5,
    )
    assert measured.mean_recall_at_k == 1 and measured.mean_reciprocal_rank == 1


async def test_boolean_and_numeric_preference_values_are_not_duplicates(service):
    first = await remember(service, preference(value=True))
    second = await remember(service, preference(value=1))
    assert isinstance(second.decision, Create)
    with pytest.raises(SchemaError):
        await remember(
            service,
            preference(value=False),
            decision=AttachEvidence(target_id=first.memory.id, expected_revision=1),
            expected_generation=second.generation,
        )


async def test_dream_multi_change_failure_rolls_back_earlier_change(service):
    await remember(service, preference(id="source"))

    async def planner(groups):
        return {
            "changes": [
                {
                    "memory": preference(id="replacement", value="pip").model_dump(
                        mode="json"
                    ),
                    "decision": {
                        "action": "supersede",
                        "target_id": "source",
                        "expected_revision": 1,
                    },
                },
                {
                    "memory": preference(id="unrelated", value="other").model_dump(
                        mode="json"
                    ),
                    "decision": {
                        "action": "supersede",
                        "target_id": "missing",
                        "expected_revision": 1,
                    },
                },
            ]
        }

    service.dream_planner = planner
    before = await recall(service)
    planned = await service.dream(owner_id="alice")
    with pytest.raises(ConflictError):
        await service.apply_dream(planned.plan.id, owner_id="alice")
    after = await recall(service)
    assert after.generation == before.generation
    assert [h.memory.id for h in after.hits] == ["source"]


async def test_local_embedder_never_downloads_or_executes_remote_code(
    tmp_path, monkeypatch
):
    import sys
    from types import SimpleNamespace

    from memory_core.memory.embedding import LocalSentenceEmbedder

    seen = {}

    class LocalModel:
        def __init__(self, path, **options):
            seen.update(options)
            assert path == str(tmp_path)

        def get_sentence_embedding_dimension(self):
            return 2

        def encode(self, texts, **options):
            return SimpleNamespace(tolist=lambda: [V for _ in texts])

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=LocalModel),
    )
    embedder = LocalSentenceEmbedder(
        tmp_path, model_id="local", revision="1", dimensions=2
    )
    assert await embedder.embed(["test"]) == [V]
    assert seen == {
        "local_files_only": True,
        "trust_remote_code": False,
        "device": "cpu",
    }
    with pytest.raises(ValueError):
        LocalSentenceEmbedder(
            tmp_path / "missing", model_id="local", revision="1", dimensions=2
        )


async def test_exact_duplicate_outside_semantic_candidates(service):
    original = preference(id="z-original", source_refs=["first"])
    await remember(service, original)
    # Opposite vectors and no lexical overlap exclude the duplicate from search.
    duplicate = original.model_copy(update={"id": "new", "source_refs": ["second"]})
    service.retrieval_policy = service.retrieval_policy.model_copy(
        update={"minimum_score": 1.0}
    )
    service._caches.clear()
    result = await service.call(
        RememberRequest(memory=duplicate, embedding=[-1.0, 0.0]), owner_id="alice"
    )
    assert result.decision.action == "attach_evidence"
    assert result.memory.id == original.id
    assert result.memory.source_refs == ["first", "second"]
    assert result.memory.revision == 2


async def test_indexed_preferences_match_all_fields_once_and_respect_scope(service):
    await register(service, "editor")
    await remember(
        service,
        preference(
            id="conditional",
            applicability={"languages": ["python", "rust"], "editor": ["vim"]},
        ),
    )
    await remember(
        service, preference(id="global", preference_key="global", applicability={})
    )
    await remember(
        service,
        preference(
            id="project",
            preference_key="project",
            scope=ProjectScope(project_id="other"),
            applicability={},
        ),
    )
    context = TaskContext(task_id="task", attributes={"languages": ["python", "rust"]})
    result = await service.resolve_prefs(context, owner_id="alice")
    assert {m.id for m in result.effective} == {"global"}
    context = context.model_copy(
        update={"attributes": {"languages": ["python", "rust"], "editor": "vim"}}
    )
    result = await service.resolve_prefs(context, owner_id="alice")
    assert {m.id for m in result.effective} == {"global", "conditional"}
    assert result.complete
    # Incremental sync removes old applicability postings and exact-content entries.
    replacement = preference(
        id="replacement",
        value="pip",
        applicability={"languages": ["python", "rust"], "editor": ["vim"]},
    )
    await remember(
        service,
        replacement,
        decision=Supersede(target_id="conditional", expected_revision=1),
        expected_generation=result.generation,
    )
    result = await service.resolve_prefs(context, owner_id="alice")
    assert {m.id for m in result.effective} == {"global", "replacement"}
    cache = service._caches["alice"]
    assert "conditional" not in cache.preference_fields
    assert all("conditional" not in ids for ids in cache.equivalents.values())


async def test_preference_index_avoids_reading_unrelated_payloads(service):
    for i in range(30):
        await remember(
            service,
            preference(
                id=f"unrelated-{i}",
                preference_key=f"rule_{i}",
                applicability={"languages": [f"language_{i}"]},
            ),
        )
    await remember(service, preference(id="wanted"))
    context = TaskContext(task_id="task", attributes={"languages": ["python"]})
    await service.resolve_prefs(context, owner_id="alice")
    cache = service._caches["alice"]

    class GuardedMemories(dict):
        def __getitem__(self, key):
            assert key == "wanted", "resolver loaded an unrelated payload"
            return super().__getitem__(key)

    cache.memories = GuardedMemories(cache.memories)
    result = await service.resolve_prefs(context, owner_id="alice")
    assert [m.id for m in result.effective] == ["wanted"]


async def test_dream_finds_exact_duplicates_without_semantic_hits(service):
    first = await remember(service, preference(id="a", source_refs=["one"]))
    second = await remember(
        service,
        preference(id="b", source_refs=["two"]),
        decision=Create(),
        expected_generation=first.generation,
    )
    service.retrieval_policy = service.retrieval_policy.model_copy(
        update={"minimum_score": 1.0}
    )
    service._caches.clear()
    planned = await service.dream(
        owner_id="alice", request=DreamPlanRequest(batch_size=2, neighbors_per_memory=1)
    )
    assert len(planned.plan.proposal.changes) == 1
    assert all(len(group.neighbors) <= 1 for group in planned.plan.groups)
    await service.apply_dream(planned.plan.id, owner_id="alice")
    result = await service.resolve_prefs(
        TaskContext(task_id="task", attributes={"languages": ["python"]}),
        owner_id="alice",
    )
    assert len(result.effective) == 1
    assert result.effective[0].source_refs == ["one", "two"]
    assert result.effective[0].updated_at > second.memory.updated_at
