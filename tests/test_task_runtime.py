"""Behavioral acceptance for durable task memory and bounded knowledge resolution."""

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError
from sqlalchemy import text

from memory_core import EmbeddingSpec, MemoryCore, build_memory_system
from memory_core.memory.errors import ConflictError, ContextRequiredError, SchemaError
from memory_core.memory.models import (
    AttachEvidence,
    DescribeSchemaRequest,
    RecallRequest,
    TaskContext,
)
from memory_core.runtime import (
    Evidence,
    Finding,
    KnowledgeGap,
    MaintenancePolicy,
    MemoryTools,
    ResolvedKnowledge,
    TaskOutcome,
    TaskRunResult,
    UnresolvedKnowledge,
)
from memory_core.runtime.models import (
    BeginTaskRequest,
    CompleteTaskRequest,
    ContextContribution,
    MaintenanceStatusRequest,
    RecordFindingRequest,
    RequireKnowledgeRequest,
    ReviewJobRequest,
)

SPEC = EmbeddingSpec(model_id="fixture", revision="1", dimensions=2)


class Embedder:
    async def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


def runtime_at(path, **kwargs):
    service = build_memory_system(
        sqlite_path=path, embedding_spec=SPEC, embedder=Embedder()
    ).service
    return MemoryCore(
        service,
        maintenance=MaintenancePolicy(
            background=False, retry_seconds=0, dream_after_writes=1000
        ),
        **kwargs,
    )


@pytest.fixture
def runtime(tmp_path):
    return runtime_at(tmp_path / "tasks.db")


def finding(statement="The project retries idempotent requests only"):
    return Finding(
        statement=statement,
        evidence=[Evidence(id="repo", uri="repo:policy.py:10", excerpt=statement)],
    )


async def begin(runtime, owner="alice", task_id="task"):
    return await runtime.call(
        BeginTaskRequest(
            context=TaskContext(
                task_id=task_id,
                project_id="project",
                attributes={"languages": ["python"]},
            ),
            objective="Implement retry behavior",
        ),
        owner_id=owner,
    )


async def capture(runtime, value=None, event_id="event", owner="alice"):
    return await runtime.call(
        RecordFindingRequest(
            task_id="task", event_id=event_id, finding=value or finding()
        ),
        owner_id=owner,
    )


async def status(runtime, owner="alice"):
    return (await runtime.call(MaintenanceStatusRequest(), owner_id=owner)).jobs


async def test_capture_survives_restart_and_completion_replays(tmp_path):
    path = tmp_path / "memory.db"
    first = runtime_at(path)
    await begin(first)
    item = finding()
    saved = await capture(first, item)
    assert (await capture(first, item)).job_ids == saved.job_ids
    with pytest.raises(ConflictError):
        await capture(first, finding("different"))
    second = runtime_at(path)
    assert await second.worker.run_once()
    assert not await second.worker.run_once()
    recalled = await second.call(
        RecallRequest(query="retry", embedding=[1, 0], project_id="project"),
        owner_id="alice",
    )
    assert len(recalled.hits) == 1
    completion = CompleteTaskRequest(
        task_id="task", outcome=TaskOutcome(status="succeeded", summary="Done")
    )
    a = await second.call(completion, owner_id="alice")
    b = await second.call(completion, owner_id="alice")
    assert a.job_ids == b.job_ids
    assert len(await status(second)) == 2
    with pytest.raises(ConflictError):
        await second.call(
            completion.model_copy(
                update={"outcome": TaskOutcome(status="failed", summary="No")}
            ),
            owner_id="alice",
        )
    assert await status(second, "bob") == []


async def test_unfinished_exit_is_not_success_and_run_captures(runtime):
    async with runtime.task(
        owner_id="alice", context=TaskContext(task_id="task"), objective="Code"
    ) as task:
        await task.prepare()
    assert (
        await runtime.call(
            BeginTaskRequest(context=TaskContext(task_id="task"), objective="Code"),
            owner_id="alice",
        )
    ).outcome is None
    async with runtime.task(
        owner_id="alice", context=TaskContext(task_id="task"), objective="Code"
    ) as task:

        async def action(prepared):
            assert prepared.receipt
            return TaskRunResult(
                outcome=TaskOutcome(status="succeeded", summary="Test passed"),
                findings=[finding()],
            )

        await task.run(action)
    assert len(await status(runtime)) == 2
    with pytest.raises(ConflictError):
        await task.before_action()


async def test_memory_first_research_capture_and_cache(runtime):
    calls = []

    async def evaluate(request):
        if request.evidence:
            return ResolvedKnowledge(
                answer="Retry only idempotent requests", evidence=[request.evidence[0]]
            )
        return UnresolvedKnowledge(reason="Need project documentation")

    async def research(request):
        calls.append(request)
        return finding().evidence

    runtime.knowledge_evaluator, runtime.researcher = evaluate, research
    await begin(runtime)
    req = RequireKnowledgeRequest(
        task_id="task", gap=KnowledgeGap(id="retry", question="Which retry policy?")
    )
    result = await runtime.call(req, owner_id="alice")
    assert result.result.status == "resolved" and result.capture.job_ids
    again = await runtime.call(req, owner_id="alice")
    assert again.result == result.result and len(calls) == 1
    assert len(await status(runtime)) == 1
    await runtime.worker.run_once()
    await begin(runtime, task_id="another")
    await runtime.call(req.model_copy(update={"task_id": "another"}), owner_id="alice")
    assert len(calls) == 1  # resolved using memory, without another research call


async def test_unresolved_gap_blocks_action_and_success_but_allows_failure(runtime):
    async with runtime.task(
        owner_id="alice", context=TaskContext(task_id="task"), objective="Code"
    ) as task:
        response = await task.require_knowledge(
            KnowledgeGap(id="unknown", question="Unknown?")
        )
        assert response.result.status == "unresolved"
        with pytest.raises(ContextRequiredError):
            await task.before_action()
        with pytest.raises(ContextRequiredError):
            await task.complete(TaskOutcome(status="succeeded", summary="Pretend"))
        await task.complete(TaskOutcome(status="failed", summary="Missing information"))


async def test_fabricated_evidence_never_becomes_memory(runtime):
    async def evaluator(request):
        return ResolvedKnowledge(answer="Invented", evidence=finding().evidence)

    runtime.knowledge_evaluator = evaluator
    await begin(runtime)
    result = await runtime.call(
        RequireKnowledgeRequest(
            task_id="task", gap=KnowledgeGap(id="g", question="Unknown?")
        ),
        owner_id="alice",
    )
    assert result.result.status == "unresolved"
    assert not await status(runtime)


async def test_expired_evidence_and_provider_timeout(runtime):
    runtime.policy = runtime.policy.model_copy(
        update={"provider_timeout_seconds": 0.01}
    )

    async def slow(request):
        await asyncio.sleep(1)

    runtime.knowledge_evaluator = slow
    await begin(runtime)
    request = RequireKnowledgeRequest(
        task_id="task", gap=KnowledgeGap(id="g", question="Unknown?")
    )
    result = await runtime.call(request, owner_id="alice")
    assert "TimeoutError" in result.result.reason

    async def evaluator(request):
        return (
            ResolvedKnowledge(answer="Old", evidence=request.evidence)
            if request.evidence
            else UnresolvedKnowledge(reason="No fresh evidence")
        )

    async def researcher(request):
        return [
            finding()
            .evidence[0]
            .model_copy(update={"observed_at": datetime.now(UTC) - timedelta(days=3)})
        ]

    runtime.knowledge_evaluator, runtime.researcher = evaluator, researcher
    result = await runtime.call(request, owner_id="alice")
    assert result.result.status == "unresolved"


async def test_context_resolver_explicit_precedence_and_schema(runtime):
    async def resolver(request):
        return ContextContribution(
            source="project detector",
            attributes={"languages": ["rust"], "activities": ["coding"]},
        )

    runtime.context_resolvers = (resolver,)
    task = await begin(runtime)
    assert task.context.attributes == {
        "languages": ["python"],
        "activities": ["coding"],
    }
    assert task.sources["languages"] == "caller"
    assert task.sources["activities"] == "project detector"

    async def invalid(request):
        return ContextContribution(source="bad", attributes={"not_registered": True})

    runtime.context_resolvers = (invalid,)
    with pytest.raises(SchemaError):
        await begin(runtime, task_id="other")
    schema = await runtime.call(DescribeSchemaRequest(), owner_id="alice")
    assert "require_knowledge" in str(schema.request_schema)


async def test_lease_replacement_cannot_commit_old_worker(runtime):
    await begin(runtime)
    await capture(runtime)
    first = runtime.queue.claim()
    assert runtime.queue.claim() is None
    with runtime.queue.transaction("alice") as session:
        session.execute(text("UPDATE memory_jobs SET lease_until=0"))
    second = runtime.queue.claim()
    assert second.id == first.id and second.token != first.token
    with pytest.raises(ConflictError):
        await runtime.worker._capture(first)
    assert not (
        await runtime.call(
            RecallRequest(query="retry", embedding=[1, 0], project_id="project"),
            owner_id="alice",
        )
    ).hits
    await runtime.worker._capture(second)
    assert (await status(runtime))[0].status == "done"


async def test_commit_ack_failure_rolls_back_memory(runtime, monkeypatch):
    await begin(runtime)
    await capture(runtime)
    claim = runtime.queue.claim()

    def broken(*args, **kwargs):
        raise OSError("ack failed")

    monkeypatch.setattr(runtime.queue, "finish", broken)
    with pytest.raises(OSError):
        await runtime.worker._capture(claim)
    recalled = await runtime.call(
        RecallRequest(query="retry", embedding=[1, 0], project_id="project"),
        owner_id="alice",
    )
    assert not recalled.hits and recalled.generation == 0


async def test_semantic_proposal_requires_review(runtime):
    await begin(runtime)
    await capture(runtime)
    await runtime.worker.run_once()

    async def reconciler(memory, neighbors):
        return AttachEvidence(
            target_id=neighbors[0].memory.id,
            expected_revision=neighbors[0].memory.revision,
        )

    runtime.service.reconciler = reconciler
    await capture(
        runtime, finding("Different supported observation"), event_id="second"
    )
    await runtime.worker.run_once()
    proposal = next(j for j in await status(runtime) if j.status == "proposed")
    await runtime.call(
        ReviewJobRequest(job_id=proposal.id, accept=True), owner_id="alice"
    )
    await runtime.worker.run_once()
    assert all(j.status == "done" for j in await status(runtime))
    with pytest.raises(ConflictError):
        await runtime.call(
            ReviewJobRequest(job_id=proposal.id, accept=True), owner_id="bob"
        )


async def test_threshold_schedules_dream_off_write_path(runtime):
    runtime.queue.policy = runtime.policy.model_copy(update={"dream_after_writes": 1})
    await begin(runtime)
    await capture(runtime)
    await runtime.worker.run_once()
    jobs = await status(runtime)
    assert {j.kind for j in jobs} == {"capture", "dream"}
    assert next(j for j in jobs if j.kind == "dream").status == "pending"
    await runtime.worker.run_once()
    assert all(j.status == "done" for j in await status(runtime))


async def test_background_worker_and_tool_adapter(runtime):
    tools = MemoryTools(runtime, owner_id="alice")
    await tools.invoke(
        BeginTaskRequest(
            context=TaskContext(task_id="task"), objective="Code"
        ).model_dump()
    )
    await capture(runtime)
    await runtime.worker.start()
    async with asyncio.timeout(3):
        while (await status(runtime))[0].status != "done":
            await asyncio.sleep(0.01)
    await runtime.worker.stop()
    assert runtime.worker._runner is None
    with pytest.raises(ValidationError):
        await tools.invoke({"operation": "maintenance_status", "owner_id": "bob"})


async def test_retries_are_bounded(runtime):
    await begin(runtime)
    await capture(runtime)

    class Broken:
        async def embed(self, texts):
            raise OSError("offline")

    runtime.service.embedder = Broken()
    for _ in range(runtime.policy.max_attempts):
        assert await runtime.worker.run_once()
    assert not await runtime.worker.run_once()
    job = (await status(runtime))[0]
    assert job.status == "failed" and job.error == "OSError"


async def test_retry_failed_job_and_owner_isolation(runtime):
    from memory_core.runtime.models import RetryJobRequest

    await begin(runtime)
    await capture(runtime)
    claim = runtime.queue.claim()
    runtime.worker._finish(claim, "failed", None)
    with pytest.raises(ConflictError):
        await runtime.call(RetryJobRequest(job_id=claim.id), owner_id="bob")
    retried = await runtime.call(RetryJobRequest(job_id=claim.id), owner_id="alice")
    assert retried.job.attempts == 0 and retried.job.status == "pending"
    await runtime.worker.run_once()
    with pytest.raises(ConflictError):
        await runtime.call(RetryJobRequest(job_id=claim.id), owner_id="alice")


async def test_concurrent_workers_claim_once(tmp_path):
    path = tmp_path / "concurrent.db"
    a, b = runtime_at(path), runtime_at(path)
    await begin(a)
    await capture(a)
    claims = await asyncio.gather(
        asyncio.to_thread(a.queue.claim), asyncio.to_thread(b.queue.claim)
    )
    assert sum(c is not None for c in claims) == 1


async def test_cancelled_provider_leaves_retryable_job(runtime):
    await begin(runtime)
    await capture(runtime)
    entered = asyncio.Event()

    async def slow(memory, neighbors):
        entered.set()
        await asyncio.Event().wait()

    runtime.service.reconciler = slow
    pending = asyncio.create_task(runtime.worker.run_once())
    await entered.wait()
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert (await status(runtime))[0].status == "pending"
    runtime.service.reconciler = None
    await runtime.worker.run_once()
    assert (await status(runtime))[0].status == "done"


async def test_event_and_job_insertion_roll_back_together(runtime, monkeypatch):
    await begin(runtime)
    original = runtime.queue.enqueue

    def fail_after_insert(*args, **kwargs):
        original(*args, **kwargs)
        raise OSError("simulated crash")

    monkeypatch.setattr(runtime.queue, "enqueue", fail_after_insert)
    with pytest.raises(OSError):
        await capture(runtime)
    assert not await status(runtime)
    with runtime.queue.transaction("alice") as session:
        assert (
            session.execute(text("SELECT COUNT(*) FROM memory_events")).scalar_one()
            == 0
        )


async def test_completed_failed_task_preserves_discovery(runtime):
    await begin(runtime)
    await capture(runtime)
    await runtime.call(
        CompleteTaskRequest(
            task_id="task",
            outcome=TaskOutcome(
                status="failed",
                summary="Implementation failed but discovery was useful",
            ),
        ),
        owner_id="alice",
    )
    while await runtime.worker.run_once():
        pass
    result = await runtime.call(
        RecallRequest(query="retry", embedding=[1, 0], project_id="project"),
        owner_id="alice",
    )
    assert len(result.hits) == 2
    assert all(h.memory.kind == "observation" for h in result.hits)
    assert all(h.memory.metadata["languages"] == ["python"] for h in result.hits)


async def test_expired_answer_blocks_later_action(runtime):
    import json

    async def evaluate(request):
        return (
            ResolvedKnowledge(answer="policy", evidence=request.evidence)
            if request.evidence
            else UnresolvedKnowledge(reason="research")
        )

    async def research(request):
        return finding().evidence

    runtime.knowledge_evaluator, runtime.researcher = evaluate, research
    async with runtime.task(
        owner_id="alice", context=TaskContext(task_id="task"), objective="Code"
    ) as task:
        await task.require_knowledge(KnowledgeGap(id="g", question="Policy?"))
        await task.before_action()
        with runtime.queue.transaction("alice") as session:
            raw = json.loads(
                session.execute(text("SELECT result FROM memory_gaps")).scalar_one()
            )
            raw["evidence"][0]["observed_at"] = (
                datetime.now(UTC) - timedelta(days=3)
            ).isoformat()
            session.execute(
                text("UPDATE memory_gaps SET result=:r"), {"r": json.dumps(raw)}
            )
        with pytest.raises(ContextRequiredError):
            await task.before_action()


async def test_stale_semantic_proposal_requires_replan(runtime):
    from memory_core.runtime.models import RetryJobRequest

    await begin(runtime)
    await capture(runtime)
    await runtime.worker.run_once()

    async def reconciler(memory, neighbors):
        return AttachEvidence(
            target_id=neighbors[0].memory.id,
            expected_revision=neighbors[0].memory.revision,
        )

    runtime.service.reconciler = reconciler
    await capture(runtime, finding("New fact"), event_id="second")
    await runtime.worker.run_once()
    proposal = next(j for j in await status(runtime) if j.status == "proposed")
    assert proposal.result.incoming.summary == "New fact"
    with runtime.queue.transaction("alice") as session:
        runtime.service.store.bump(session, "alice", "test_change", {})
    with pytest.raises(ConflictError):
        await runtime.call(
            ReviewJobRequest(job_id=proposal.id, accept=True), owner_id="alice"
        )
    await runtime.call(RetryJobRequest(job_id=proposal.id), owner_id="alice")
    await runtime.worker.run_once()
    renewed = next(j for j in await status(runtime) if j.id == proposal.id)
    assert renewed.status == "proposed"
    assert renewed.result.generation > proposal.result.generation


async def test_prepared_task_injects_knowledge_and_rejects_changed_evidence(runtime):
    from memory_core.runtime.models import PrepareTaskRequest

    async def evaluate(request):
        if request.evidence:
            return ResolvedKnowledge(answer="Policy", evidence=[request.evidence[0]])
        return UnresolvedKnowledge(reason="Missing")

    runtime.knowledge_evaluator = evaluate
    await begin(runtime)
    await capture(runtime)
    await runtime.worker.run_once()
    req = RequireKnowledgeRequest(
        task_id="task", gap=KnowledgeGap(id="policy", question="Policy?")
    )
    await runtime.call(req, owner_id="alice")
    prepared = await runtime.call(
        PrepareTaskRequest(task_id="task", guard=True), owner_id="alice"
    )
    assert prepared.knowledge["policy"].status == "resolved"
    # Simulate a legitimate record revision; previously resolved evidence is stale.
    with runtime.queue.transaction("alice") as session:
        raw = session.execute(text("SELECT payload FROM memories_v1")).scalar_one()
        from memory_core.memory.models import MEMORY_ADAPTER

        memory = MEMORY_ADAPTER.validate_json(raw)
        changed = memory.model_copy(update={"revision": memory.revision + 1})
        generation = runtime.service.store.bump(session, "alice", "test_revision", {})
        embedding = session.execute(
            text("SELECT embedding FROM memories_v1")
        ).scalar_one()
        runtime.service.store.put(session, "alice", changed, embedding, generation)
    with pytest.raises(ContextRequiredError):
        await runtime.call(
            PrepareTaskRequest(task_id="task", guard=True), owner_id="alice"
        )
    refreshed = await runtime.call(req, owner_id="alice")
    assert refreshed.result.status == "resolved"
    assert refreshed.result.evidence[0].id.endswith(":2")


async def test_provider_model_copy_cannot_bypass_validation(runtime):
    async def research(request):
        return finding().evidence

    async def evaluate(request):
        if not request.evidence:
            return UnresolvedKnowledge(reason="research")
        return ResolvedKnowledge(answer="valid", evidence=request.evidence).model_copy(
            update={"answer": ""}
        )

    runtime.researcher, runtime.knowledge_evaluator = research, evaluate
    await begin(runtime)
    response = await runtime.call(
        RequireKnowledgeRequest(
            task_id="task", gap=KnowledgeGap(id="g", question="Policy?")
        ),
        owner_id="alice",
    )
    assert response.result.status == "unresolved"
    assert "ValidationError" in response.result.reason
    assert not await status(runtime)


async def test_structured_reasoner_supplies_schemas_and_validates_json():
    from memory_core.memory.models import ObservationMemory
    from memory_core.runtime.reasoning import StructuredReasoner

    calls = []

    async def generate(request):
        calls.append(request)
        return '{"action":"create"}'

    reasoner = StructuredReasoner(generate)
    result = await reasoner.reconcile(
        ObservationMemory(summary="Fact", observation="Fact"), []
    )
    assert result.action == "create"
    assert calls[0].operation == "reconcile"
    assert calls[0].output_schema["discriminator"]["propertyName"] == "action"
    assert calls[0].input["memory"]["kind"] == "observation"

    async def invalid(request):
        return '{"action":"supersede","target_id":"x"}'

    with pytest.raises(ValidationError):
        await StructuredReasoner(invalid).reconcile(
            ObservationMemory(summary="Fact", observation="Fact"), []
        )


async def test_owner_index_cache_evicts_lru_and_reloads_memories(tmp_path):
    memory = MemoryCore.local(
        tmp_path / "lru.db",
        embedding_spec=SPEC,
        embedder=Embedder(),
        max_cached_owners=2,
        maintenance=MaintenancePolicy(background=False),
    )
    try:
        await begin(memory)
        await capture(memory, finding())
        assert await memory.worker.run_once()
        request = RecallRequest(query="retry", embedding=[1, 0], project_id="project")
        expected = await memory.call(request, owner_id="alice")
        await memory.call(request, owner_id="bob")
        await memory.call(request, owner_id="alice")
        await memory.call(request, owner_id="carol")
        assert list(memory.service._caches) == ["alice", "carol"]
        await memory.call(request, owner_id="bob")
        assert list(memory.service._caches) == ["carol", "bob"]
        reloaded = await memory.call(request, owner_id="alice")
        assert [hit.memory for hit in reloaded.hits] == [
            hit.memory for hit in expected.hits
        ]
        assert len(reloaded.hits) == 1
        assert len(memory.service._caches) == 2
    finally:
        memory.service.store.db.engine.dispose()


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_owner_cache_limit_requires_positive_integer(tmp_path, limit):
    with pytest.raises(ValueError, match="max_cached_owners"):
        MemoryCore.local(
            tmp_path / "invalid.db",
            embedding_spec=SPEC,
            embedder=Embedder(),
            max_cached_owners=limit,
        )
