"""Offline SDK integration acceptance; no model calls or credentials are required."""

from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("claude_agent_sdk")
pytest.importorskip("mcp")

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookMatcher,
    ResultMessage,
)
from claude_agent_sdk._internal.transport import Transport
from mcp import Client

from memory_core import EmbeddingSpec, MemoryCore
from memory_core.integrations.claude import (
    AgentCapacityError,
    AgentRunConflict,
    ClaudeMemoryAgent,
    ClaudeMemoryPolicy,
    ContextBudgetError,
)
from memory_core.integrations.claude.binding import TaskBinding
from memory_core.integrations.claude.hooks import MemoryHooks
from memory_core.memory.models import TaskContext
from memory_core.runtime import (
    Evidence,
    Finding,
    KnowledgeGap,
    MaintenancePolicy,
    TaskOutcome,
    TaskRunResult,
)
from memory_core.runtime.models import (
    BeginTaskRequest,
    CompleteTaskRequest,
)


class Embedder:
    calls = 0

    async def embed(self, texts):
        self.calls += 1
        return [[1.0, 0.0] for _ in texts]


@pytest.fixture
def memory(tmp_path):
    value = MemoryCore.local(
        tmp_path / "agents.db",
        embedding_spec=EmbeddingSpec(model_id="test", revision="1", dimensions=2),
        embedder=Embedder(),
        maintenance=MaintenancePolicy(background=False),
    )
    yield value
    value.service.store.db.dispose()


def output(summary="Finished", **kwargs):
    return ResultMessage(
        subtype="success",
        duration_ms=0,
        duration_api_ms=0,
        is_error=False,
        num_turns=1,
        session_id="sdk-session",
        structured_output=TaskRunResult(
            outcome=TaskOutcome(status="succeeded", summary=summary), **kwargs
        ).model_dump(mode="json"),
    )


class FakeClient:
    def __init__(self, options):
        self.options = options
        self.closed = False
        self.prompt = None
        self.message = output()

    async def connect(self):
        pass

    async def query(self, prompt):
        self.prompt = prompt

    async def receive_response(self):
        yield self.message

    async def disconnect(self):
        self.closed = True


async def run(agent, task="task", owner="alice", **kwargs):
    return await agent.run(
        owner_id=owner,
        context=TaskContext(task_id=task, project_id=owner),
        objective="Implement retry behavior",
        **kwargs,
    )


async def test_structured_completion_and_original_options_are_preserved(memory):
    clients = []

    def factory(options):
        client = FakeClient(options)
        clients.append(client)
        return client

    async def host_hook(*args):
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
            }
        }

    matcher = HookMatcher(hooks=[host_hook])
    source = ClaudeAgentOptions(
        allowed_tools=["Read"],
        permission_mode="plan",
        hooks={"PreToolUse": [matcher]},
        system_prompt="Existing host instructions",
    )
    agent = ClaudeMemoryAgent(memory, client_factory=factory)
    report = await run(agent, options=source)
    assert report.result.outcome.status == "succeeded" and report.capture.job_ids
    assert report.execution_error is None and report.capture_error is None
    assert clients[0].closed and agent.admitted_runs == 0
    configured = clients[0].options
    assert configured.permission_mode == "plan"
    assert configured.system_prompt == source.system_prompt
    assert source.allowed_tools == ["Read"] and source.mcp_servers == {}
    assert len(source.hooks["PreToolUse"]) == 1
    assert configured.hooks["PreToolUse"][0] is matcher
    assert configured.output_format["schema"]["properties"]["outcome"]
    assert "MemoryCore task context" in clients[0].prompt


async def test_task_bound_mcp_schema_and_no_admin_tools(memory):
    binding = TaskBinding(
        memory,
        "alice",
        TaskContext(task_id="task", project_id="p"),
        "Code",
        ClaudeMemoryPolicy(),
    )
    await binding.start()
    server = binding.sdk_server()
    async with Client(server["instance"]) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
        assert (
            "complete_task" not in tools
            and "review_job" not in tools
            and "begin_task" not in tools
        )
        assert "task_id" not in tools["require_knowledge"].input_schema["properties"]
        assert "project_id" not in tools["recall"].input_schema["properties"]
        wrong = await client.call_tool(
            "require_knowledge",
            {"task_id": "other", "gap": {"id": "g", "question": "Q?"}},
        )
        assert wrong.is_error
        wrong = await client.call_tool("recall", {"project_id": "other", "query": "q"})
        assert wrong.is_error
        prepared = await client.call_tool("prepare_task", {})
        assert prepared.structured_content["context"]["task_id"] == "task"
    assert memory.worker._runner is None


async def test_guards_block_gaps_but_leave_recovery_and_structured_output_available(
    memory,
):
    binding = TaskBinding(
        memory, "alice", TaskContext(task_id="task"), "Code", ClaudeMemoryPolicy()
    )
    await binding.start()
    await binding.session.require_knowledge(KnowledgeGap(id="g", question="Unknown?"))
    hooks = MemoryHooks(binding)
    blocked = await hooks.before_tool({"tool_name": "Write"}, "use", {})
    assert blocked["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert (
        await hooks.before_tool(
            {"tool_name": "mcp__memorycore__require_knowledge"}, "use", {}
        )
        == {}
    )
    assert await hooks.before_tool({"tool_name": "StructuredOutput"}, "use", {}) == {}
    agent = ClaudeMemoryAgent(memory, client_factory=FakeClient)
    report = await agent.run(
        owner_id="alice", context=binding.context, objective="Code"
    )
    assert (
        report.result.outcome.status == "failed"
    )  # Cannot claim success with an unresolved gap.


async def test_cached_context_avoids_reembedding_but_rechecks_new_gaps(memory):
    binding = TaskBinding(
        memory, "alice", TaskContext(task_id="task"), "Code", ClaudeMemoryPolicy()
    )
    await binding.start()
    initial = memory.service.embedder.calls
    for _ in range(5):
        await binding.prepare(guard=True)
    assert memory.service.embedder.calls == initial
    await binding.session.require_knowledge(KnowledgeGap(id="new", question="Unknown?"))
    blocked = await MemoryHooks(binding).before_tool({"tool_name": "Write"}, None, {})
    assert blocked["hookSpecificOutput"]["permissionDecision"] == "deny"


async def test_capacity_queue_and_duplicate_task(memory):
    entered = asyncio.Event()
    release = asyncio.Event()
    clients = []

    class Slow(FakeClient):
        async def receive_response(self):
            entered.set()
            await release.wait()
            yield self.message

    def factory(options):
        c = Slow(options)
        clients.append(c)
        return c

    agent = ClaudeMemoryAgent(
        memory,
        policy=ClaudeMemoryPolicy(max_concurrent_runs=1, max_pending_runs=1),
        client_factory=factory,
    )
    first = asyncio.create_task(run(agent, "one"))
    await entered.wait()
    with pytest.raises(AgentRunConflict):
        await run(agent, "one")
    second = asyncio.create_task(run(agent, "two"))
    await asyncio.sleep(0)
    with pytest.raises(AgentCapacityError):
        await run(agent, "three")
    assert len(clients) == 1
    release.set()
    reports = await asyncio.gather(first, second)
    assert all(r.capture for r in reports) and len(clients) == 2
    assert agent.admitted_runs == 0


async def test_cancellation_disconnects_and_leaves_task_resumable(memory):
    entered = asyncio.Event()
    clients = []

    class Slow(FakeClient):
        async def receive_response(self):
            entered.set()
            await asyncio.Event().wait()
            yield self.message

    def factory(options):
        c = Slow(options)
        clients.append(c)
        return c

    agent = ClaudeMemoryAgent(memory, client_factory=factory)
    pending = asyncio.create_task(run(agent))
    await entered.wait()
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert clients[0].closed and agent.admitted_runs == 0
    task = await memory.call(
        BeginTaskRequest(
            context=TaskContext(task_id="task", project_id="alice"),
            objective="Implement retry behavior",
        ),
        owner_id="alice",
    )
    assert task.outcome is None


async def test_session_errors_do_not_become_success(memory):
    class NoStructured(FakeClient):
        async def receive_response(self):
            self.message.structured_output = None
            yield self.message

    report = await run(ClaudeMemoryAgent(memory, client_factory=NoStructured))
    assert (
        report.result.outcome.status == "failed"
        and report.execution_error == "ValidationError"
    )
    assert report.capture


async def test_capture_failure_is_separate_from_successful_execution(
    memory, monkeypatch
):
    original = memory.call

    async def call(request, **kwargs):
        if isinstance(request, CompleteTaskRequest):
            raise OSError("storage unavailable")
        return await original(request, **kwargs)

    monkeypatch.setattr(memory, "call", call)
    report = await run(ClaudeMemoryAgent(memory, client_factory=FakeClient))
    assert report.result.outcome.status == "succeeded"
    assert report.capture is None and report.capture_error == "OSError"
    assert report.execution_error is None


async def test_tool_evidence_and_fabricated_findings(memory):
    binding = TaskBinding(
        memory, "alice", TaskContext(task_id="task"), "Code", ClaudeMemoryPolicy()
    )
    await binding.start()
    evidence = Evidence(id="test-run", uri="tool:pytest", excerpt="3 passed")
    value = TaskRunResult(
        outcome=TaskOutcome(status="succeeded", summary="Done"),
        findings=[Finding(statement="Tests passed", evidence=[evidence])],
    )
    with pytest.raises(Exception, match="observed by this run"):
        binding.validate_result(value.model_dump(mode="json"))

    async def extractor(input):
        return [evidence]

    response = await MemoryHooks(binding, extractor).after_tool(
        {"tool_name": "Bash"}, "t", {}
    )
    assert "test-run" in response["hookSpecificOutput"]["additionalContext"]
    assert binding.validate_result(value.model_dump(mode="json")) == value


async def test_mandatory_context_budget_fails_instead_of_dropping_preferences(memory):
    binding = TaskBinding(
        memory,
        "alice",
        TaskContext(task_id="task"),
        "x" * 1500,
        ClaudeMemoryPolicy(max_context_chars=1024),
    )
    prepared = await binding.start()
    with pytest.raises(ContextBudgetError):
        binding.render(prepared)


class ScriptedTransport(Transport):
    """Exercise actual SDK serialization and parsing without launching a model."""

    def __init__(self):
        self.queue = asyncio.Queue()
        self.closed = False
        self.writes = []

    async def connect(self):
        pass

    async def write(self, data):
        message = json.loads(data)
        self.writes.append(message)
        if message["type"] == "control_request":
            await self.queue.put(
                {
                    "type": "control_response",
                    "response": {
                        "subtype": "success",
                        "request_id": message["request_id"],
                        "response": {},
                    },
                }
            )
        elif message["type"] == "user":
            await self.queue.put(
                {
                    "type": "result",
                    "subtype": "success",
                    "duration_ms": 0,
                    "duration_api_ms": 0,
                    "is_error": False,
                    "num_turns": 1,
                    "session_id": "actual-sdk-session",
                    "structured_output": {
                        "outcome": {
                            "status": "succeeded",
                            "summary": "Done",
                            "evidence": [],
                        },
                        "findings": [],
                    },
                }
            )

    async def read_messages(self):
        while not self.closed:
            value = await self.queue.get()
            if value is None:
                break
            yield value

    async def close(self):
        self.closed = True
        await self.queue.put(None)

    def is_ready(self):
        return not self.closed

    async def end_input(self):
        pass


async def test_actual_sdk_client_transport_protocol(memory):
    transport = ScriptedTransport()

    def factory(options):
        return ClaudeSDKClient(options=options, transport=transport)

    report = await run(ClaudeMemoryAgent(memory, client_factory=factory))
    assert report.sdk_session_id == "actual-sdk-session" and report.capture
    assert transport.closed
    assert any(m["type"] == "user" for m in transport.writes)


async def test_cross_adapter_execution_lease_and_expired_token_fencing(memory):
    from sqlalchemy import text

    from memory_core.integrations.claude.leases import RunLeases

    leases_a, leases_b = RunLeases(memory), RunLeases(memory)
    first = leases_a.claim("alice", "same", 60)
    with pytest.raises(AgentRunConflict):
        leases_b.claim("alice", "same", 60)
    with memory.queue.transaction("alice") as session:
        session.execute(
            text("UPDATE memory_agent_leases SET expires=0 WHERE owner='alice'")
        )
    second = leases_b.claim("alice", "same", 60)
    with pytest.raises(AgentRunConflict):
        first.validate()
    first.release()  # An old cleanup must not delete the new owner's token.
    second.validate()
    second.release()


async def test_atomic_capture_rejects_expired_execution_lease(memory):
    from sqlalchemy import text

    from memory_core.integrations.claude.leases import RunLeases

    binding = TaskBinding(
        memory, "alice", TaskContext(task_id="task"), "Code", ClaudeMemoryPolicy()
    )
    await binding.start()
    binding.lease = RunLeases(memory).claim("alice", "task", 60)
    with memory.queue.transaction("alice") as session:
        session.execute(text("UPDATE memory_agent_leases SET expires=0"))
    with pytest.raises(AgentRunConflict):
        await binding.complete(
            TaskRunResult(outcome=TaskOutcome(status="succeeded", summary="stale"))
        )
    with memory.queue.transaction("alice") as session:
        assert memory.queue.task(session, "alice", "task").outcome is None
        assert (
            session.execute(text("SELECT COUNT(*) FROM memory_jobs")).scalar_one() == 0
        )


async def test_parallel_owners_are_isolated_and_use_distinct_clients(memory):
    clients = []
    all_connected = asyncio.Event()

    class Concurrent(FakeClient):
        async def connect(self):
            if len(clients) == 2:
                all_connected.set()
            await all_connected.wait()

    def factory(options):
        client = Concurrent(options)
        clients.append(client)
        return client

    agent = ClaudeMemoryAgent(
        memory, client_factory=factory, policy=ClaudeMemoryPolicy(max_concurrent_runs=2)
    )
    async with asyncio.timeout(5):
        reports = await asyncio.gather(
            run(agent, owner="alice"), run(agent, owner="bob")
        )
    assert len(clients) == 2 and all(r.capture for r in reports)
    assert sum('"project_id": "alice"' in c.prompt for c in clients) == 1
    assert sum('"project_id": "bob"' in c.prompt for c in clients) == 1
    assert all(
        not ('"project_id": "alice"' in c.prompt and '"project_id": "bob"' in c.prompt)
        for c in clients
    )
    assert clients[0].options.mcp_servers is not clients[1].options.mcp_servers
    assert clients[0].options.hooks is not clients[1].options.hooks


async def test_admission_timeout_does_not_start_second_client(memory):
    entered = asyncio.Event()
    clients = []

    class Waiting(FakeClient):
        async def receive_response(self):
            entered.set()
            await asyncio.Event().wait()
            yield self.message

    def factory(options):
        value = Waiting(options)
        clients.append(value)
        return value

    agent = ClaudeMemoryAgent(
        memory,
        client_factory=factory,
        policy=ClaudeMemoryPolicy(
            max_concurrent_runs=1, max_pending_runs=1, admission_timeout_seconds=0.02
        ),
    )
    first = asyncio.create_task(run(agent, "first"))
    await entered.wait()
    with pytest.raises(AgentCapacityError):
        await run(agent, "second")
    assert len(clients) == 1
    await agent.aclose()
    assert first.cancelled() and clients[0].closed and agent.admitted_runs == 0


async def test_overall_timeout_disconnects_and_frees_capacity(memory):
    clients = []

    class Waiting(FakeClient):
        async def receive_response(self):
            await asyncio.Event().wait()
            yield self.message

    def factory(options):
        value = Waiting(options)
        clients.append(value)
        return value

    agent = ClaudeMemoryAgent(
        memory,
        client_factory=factory,
        policy=ClaudeMemoryPolicy(run_timeout_seconds=0.1),
    )
    with pytest.raises(TimeoutError):
        await run(agent)
    assert clients[0].closed and agent.admitted_runs == 0


async def test_invalid_sdk_options_rejected_without_starting_task(memory):
    agent = ClaudeMemoryAgent(memory, client_factory=FakeClient)
    with pytest.raises(ValueError, match="fresh SDK"):
        await run(agent, options=ClaudeAgentOptions(resume="other-owner-session"))
    assert agent.admitted_runs == 0
    with memory.queue.transaction("alice") as session:
        from sqlalchemy import text

        assert (
            session.execute(text("SELECT COUNT(*) FROM memory_tasks")).scalar_one() == 0
        )


async def test_shared_worker_is_not_stopped_by_per_agent_mcp_session(memory):
    await memory.worker.start()
    runner = memory.worker._runner
    try:
        binding = TaskBinding(
            memory, "alice", TaskContext(task_id="task"), "Code", ClaudeMemoryPolicy()
        )
        await binding.start()
        async with Client(binding.sdk_server()["instance"]) as client:
            await client.list_tools()
        assert memory.worker._runner is runner and not runner.done()
    finally:
        await memory.worker.stop()


async def test_cached_context_invalidates_on_new_memory_generation(memory):
    binding = TaskBinding(
        memory, "alice", TaskContext(task_id="task"), "Code", ClaudeMemoryPolicy()
    )
    first = await binding.start()
    initial = memory.service.embedder.calls
    with memory.queue.transaction("alice") as session:
        memory.service.store.bump(session, "alice", "test_change", {})
    prepared = await binding.prepare(guard=True)
    assert prepared.receipt != first.receipt
    assert memory.service.embedder.calls == initial + 1


async def test_task_tool_limits_reject_excessive_output(memory):
    from memory_core.memory.errors import SchemaError

    binding = TaskBinding(
        memory,
        "alice",
        TaskContext(task_id="task"),
        "Code",
        ClaudeMemoryPolicy(max_recall_hits=2, max_tool_output_chars=1024),
    )
    await binding.start()
    with pytest.raises(SchemaError, match="recall limit"):
        await binding.invoke({"operation": "recall", "query": "q", "limit": 3})
    with pytest.raises(ContextBudgetError, match="output budget"):
        await binding.invoke({"operation": "describe_schema"})


async def test_disallowed_memory_tools_do_not_get_silently_reenabled(memory):
    agent = ClaudeMemoryAgent(memory, client_factory=FakeClient)
    with pytest.raises(ValueError, match="disallowed_tools"):
        await run(
            agent, options=ClaudeAgentOptions(disallowed_tools=["mcp__memorycore__*"])
        )
    assert agent.admitted_runs == 0


async def test_lease_cleanup_failure_does_not_leak_capacity(memory, monkeypatch):
    from memory_core.integrations.claude.leases import RunLease

    def fail(self):
        raise OSError("lease cleanup unavailable")

    monkeypatch.setattr(RunLease, "release", fail)
    agent = ClaudeMemoryAgent(memory, client_factory=FakeClient)
    report = await run(agent)
    assert report.result.outcome.status == "succeeded" and report.capture
    assert report.cleanup_error == "OSError" and agent.admitted_runs == 0


async def test_many_tasks_stay_within_client_capacity_and_release_all_leases(memory):
    from sqlalchemy import text

    active = 0
    peak = 0

    class Counted(FakeClient):
        async def connect(self):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)

        async def receive_response(self):
            await asyncio.sleep(0.01)
            yield self.message

        async def disconnect(self):
            nonlocal active
            active -= 1
            await super().disconnect()

    agent = ClaudeMemoryAgent(
        memory,
        client_factory=Counted,
        policy=ClaudeMemoryPolicy(max_concurrent_runs=4, max_pending_runs=32),
    )
    async with asyncio.timeout(15):
        reports = await asyncio.gather(
            *(run(agent, task=f"task-{i}") for i in range(32))
        )
    assert all(r.capture and not r.execution_error for r in reports)
    assert 1 < peak <= 4 and active == 0 and agent.admitted_runs == 0
    with memory.queue.transaction("alice") as session:
        assert (
            session.execute(
                text("SELECT COUNT(*) FROM memory_agent_leases")
            ).scalar_one()
            == 0
        )
