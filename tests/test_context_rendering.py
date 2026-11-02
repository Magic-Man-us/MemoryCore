import json

import pytest

from memory_core.memory.models import (
    MemoryHit,
    ObservationMemory,
    PreferencesResponse,
    TaskContext,
)
from memory_core.runtime import ContextBudgetError, render_task_context
from memory_core.runtime.context import CONTEXT_PREFIX
from memory_core.runtime.models import PreparedTask, UnresolvedKnowledge


def prepared():
    return PreparedTask(
        context=TaskContext(task_id="task", attributes={"languages": ["python"]}),
        objective="Implement a client",
        preferences=PreferencesResponse(effective=[], conflicts=[], generation=1),
        receipt="receipt",
        knowledge={"gap": UnresolvedKnowledge(reason="Need evidence")},
        supporting=[
            MemoryHit(
                memory=ObservationMemory(
                    id=str(i),
                    summary='Quotes: " and Unicode: café 🐍',
                    observation="Fact",
                ),
                score=1.0,
            )
            for i in range(3)
        ],
    )


def test_context_is_valid_json_preserving_mandatory_fields_and_order():
    value = prepared()
    rendered = render_task_context(value)
    assert json.loads(rendered.removeprefix(CONTEXT_PREFIX)) == value.model_dump(
        mode="json"
    )
    assert [hit.memory.id for hit in value.supporting] == ["0", "1", "2"]


def test_budget_keeps_ranked_prefix_and_never_drops_mandatory_context():
    value = prepared()
    one = value.model_copy(update={"supporting": value.supporting[:1]})
    budget = len(render_task_context(one))
    rendered = render_task_context(value, max_chars=budget)
    assert len(rendered) == budget
    result = json.loads(rendered.removeprefix(CONTEXT_PREFIX))
    assert [hit["memory"]["id"] for hit in result["supporting"]] == ["0"]
    assert result["knowledge"]["gap"]["status"] == "unresolved"
    mandatory = len(render_task_context(value.model_copy(update={"supporting": []})))
    with pytest.raises(ContextBudgetError):
        render_task_context(value, max_chars=mandatory - 1)
    for invalid in [0, -1, True, 1.5]:
        with pytest.raises(ValueError):
            render_task_context(value, max_chars=invalid)


def test_claude_budget_error_remains_same_public_exception():
    pytest.importorskip("claude_agent_sdk")
    from memory_core.integrations.claude import ContextBudgetError as ClaudeError

    assert ClaudeError is ContextBudgetError


async def test_generic_task_reuses_retrieval_and_still_checks_changes(tmp_path):
    from memory_core import EmbeddingSpec, MemoryCore
    from memory_core.memory.errors import ContextRequiredError
    from memory_core.runtime import KnowledgeGap, MaintenancePolicy

    class Embedder:
        calls = 0

        async def embed(self, texts):
            self.calls += 1
            return [[1.0, 0.0] for _ in texts]

    embedder = Embedder()
    memory = MemoryCore.local(
        tmp_path / "context.db",
        embedding_spec=EmbeddingSpec(model_id="test", revision="1", dimensions=2),
        embedder=embedder,
        maintenance=MaintenancePolicy(background=False),
    )
    try:
        async with memory.task(
            owner_id="alice", context=TaskContext(task_id="chat-turn"), objective="Code"
        ) as task:
            first = await task.prepare()
            for _ in range(4):
                assert (await task.before_action()).receipt == first.receipt
            assert embedder.calls == 1
            with memory.queue.transaction("alice") as session:
                memory.service.store.bump(session, "alice", "test_change", {})
            assert (await task.before_action()).receipt != first.receipt
            assert embedder.calls == 2
            await task.require_knowledge(
                KnowledgeGap(id="gap", question="Missing information?")
            )
            before = embedder.calls
            with pytest.raises(ContextRequiredError):
                await task.before_action()
            assert embedder.calls == before
    finally:
        memory.service.store.db.dispose()
