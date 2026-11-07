from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from memory_core import Database, EmbeddingSpec, MemoryCore, MemoryService
from memory_core.integrations import (
    ContextUpdate,
    SessionContext,
    SessionContextRequest,
)
from memory_core.memory.errors import ContextBudgetError, ContextRequiredError
from memory_core.memory.models import (
    Create,
    DescribeSchemaRequest,
    ObservationMemory,
    PreferenceMemory,
    ProjectScope,
    RememberRequest,
    Supersede,
    TaskContext,
)
from memory_core.runtime import MaintenancePolicy


class Embedder:
    def __init__(self):
        self.calls = 0

    async def embed(self, texts):
        self.calls += 1
        return [[1.0, 0.0] for _ in texts]


@pytest.fixture
def memory(tmp_path):
    embedder = Embedder()
    runtime = MemoryCore(
        MemoryService(
            Database(tmp_path / "sessions.db"),
            embedding_spec=EmbeddingSpec(
                model_id="fixture", revision="1", dimensions=2
            ),
            embedder=embedder,
        ),
        maintenance=MaintenancePolicy(background=False, dream_after_writes=10000),
    )
    return runtime, embedder


async def save(memory, item, *, owner="alice", decision=None):
    schema = await memory.service.call(DescribeSchemaRequest(), owner_id=owner)
    return await memory.service.call(
        RememberRequest(
            memory=item,
            embedding=[1.0, 0.0],
            decision=decision,
            expected_generation=schema.generation if decision is not None else None,
        ),
        owner_id=owner,
    )


def pref(value="uv", **kwargs):
    return PreferenceMemory(
        summary=f"Use {value}", preference_key="package_manager", value=value, **kwargs
    )


def request(*, task="session", project="app", query=None, budget=20000, limit=5):
    return SessionContextRequest(
        context=TaskContext(task_id=task, project_id=project),
        query=query,
        max_tokens=budget,
        limit=limit,
    )


@dataclass(slots=True)
class PersistentClient:
    provider: str
    system_prompt: str = "Existing host policy"
    session_id: str = "existing-session"
    messages: list[str] = field(default_factory=list)

    async def send(self, text):
        self.messages.append(text)
        return f"{self.provider} answer"


@pytest.mark.parametrize("provider", ["claude", "openai", "local"])
async def test_persistent_clients_get_revisions_and_withdrawals_without_recreation(
    memory, provider
):
    runtime, embedder = memory
    first = await save(runtime, pref())
    session = SessionContext(runtime, owner_id="alice", count_tokens=len)
    client = PersistentClient(provider)
    identity = id(client)
    initial = await session.prepare(request())
    assert initial.update is ContextUpdate.INITIAL and '"value":"uv"' in initial.text
    await client.send(initial.text + "\nInstall the project")
    session.commit(initial)
    await session.before_action(initial)
    unchanged = await session.prepare(request())
    assert unchanged.update is ContextUpdate.UNCHANGED and unchanged.text == ""
    session.commit(unchanged)
    changed = await save(
        runtime,
        pref("pip"),
        decision=Supersede(target_id=first.memory.id, expected_revision=1),
    )
    updated = await session.prepare(request())
    assert updated.update is ContextUpdate.UPDATED
    assert [(item.id, item.revision) for item in updated.withdrawn] == [
        (first.memory.id, 1)
    ]
    assert '"value":"pip"' in updated.text
    assert '"value":"uv"' not in updated.text
    await client.send(updated.text + "\nTry again")
    session.commit(updated)
    await session.before_action(updated)
    assert id(client) == identity and client.session_id == "existing-session"
    assert client.system_prompt == "Existing host policy"
    assert len(client.messages) == 2 and embedder.calls == 0
    assert updated.preferences.effective[0].id == changed.memory.id


async def test_project_switch_withdraws_prior_project_preferences(memory):
    runtime, _ = memory
    old = await save(runtime, pref(scope=ProjectScope(project_id="a")))
    await save(runtime, pref("pip", scope=ProjectScope(project_id="b")))
    session = SessionContext(runtime, owner_id="alice", count_tokens=len)
    one = await session.prepare(request(project="a"))
    session.commit(one)
    two = await session.prepare(request(project="b"))
    assert two.update is ContextUpdate.UPDATED
    assert [item.id for item in two.withdrawn] == [old.memory.id]
    assert [item.value for item in two.preferences.effective] == ["pip"]
    assert '"value":"uv"' not in two.text


async def test_owner_is_host_bound_and_never_taken_from_turn_input(memory):
    runtime, _ = memory
    await save(runtime, pref("alice-only"))
    await save(runtime, pref("bob-only"), owner="bob")
    session = SessionContext(runtime, owner_id="bob", count_tokens=len)
    result = await session.prepare(request())
    assert "bob-only" in result.text and "alice-only" not in result.text


async def test_failed_injection_is_retried_and_foreign_results_cannot_commit(memory):
    runtime, _ = memory
    await save(runtime, pref())
    session = SessionContext(runtime, owner_id="alice", count_tokens=len)
    first = await session.prepare(request())
    second = await session.prepare(request())
    assert first.text and second.text
    assert second.update is ContextUpdate.INITIAL
    with pytest.raises(ContextRequiredError):
        session.commit(first)
    other = SessionContext(runtime, owner_id="alice", count_tokens=len)
    with pytest.raises(ContextRequiredError):
        other.commit(second)
    session.commit(second)
    assert (await session.prepare(request())).update is ContextUpdate.UNCHANGED


async def test_conflicts_and_revisions_are_checked_before_actions(memory):
    runtime, _ = memory
    await save(runtime, pref())
    session = SessionContext(runtime, owner_id="alice", count_tokens=len)
    initial = await session.prepare(request())
    session.commit(initial)
    await save(runtime, pref("pip"), decision=Create())
    with pytest.raises(ContextRequiredError):
        await session.before_action(initial)
    conflict = await session.prepare(request())
    assert conflict.preferences.conflicts
    session.commit(conflict)
    # Response DTOs are not permission authority: mutation cannot clear the trusted snapshot.
    conflict.preferences.conflicts.clear()
    with pytest.raises(ContextRequiredError, match="conflicts"):
        await session.before_action(conflict)


async def test_independent_tasks_share_preferences_but_shared_task_receipts_contend(
    memory,
):
    runtime, _ = memory
    await save(runtime, pref())
    one = SessionContext(runtime, owner_id="alice", count_tokens=len)
    two = SessionContext(runtime, owner_id="alice", count_tokens=len)
    a = await one.prepare(request(task="claude-session"))
    one.commit(a)
    b = await two.prepare(request(task="openai-session"))
    two.commit(b)
    await one.before_action(a)
    await two.before_action(b)
    shared_a = await one.prepare(request(task="shared"))
    one.commit(shared_a)
    shared_b = await two.prepare(request(task="shared"))
    two.commit(shared_b)
    with pytest.raises(ContextRequiredError):
        await one.before_action(shared_a)
    await two.before_action(shared_b)


async def test_token_budget_uses_caller_counter_and_never_drops_mandatory_preferences(
    memory,
):
    runtime, embedder = memory
    await save(runtime, pref("x" * 500))
    measurements = []

    def count(text):
        measured = len(text.encode("utf-8")) * 2
        measurements.append(measured)
        return measured

    session = SessionContext(runtime, owner_id="alice", count_tokens=count)
    with pytest.raises(ContextBudgetError):
        await session.prepare(request(budget=100))
    result = await session.prepare(request(budget=20000))
    assert result.used_tokens == count(result.text)
    assert result.used_tokens <= 20000 and measurements and embedder.calls == 0


async def test_supporting_drop_metadata_and_changed_budget_reinject_correct_subset(
    memory,
):
    runtime, embedder = memory
    await save(runtime, pref())
    await save(runtime, ObservationMemory(summary="Supporting", observation="z" * 2500))
    session = SessionContext(runtime, owner_id="alice", count_tokens=len)
    full = await session.prepare(request(query="task", budget=20000))
    assert full.supporting
    session.commit(full)
    mandatory = await session.prepare(request(budget=20000))
    # This budget fits mandatory data but cannot fit the long supporting observation.
    small_budget = len(mandatory.text) + 100
    small = await session.prepare(request(query="task", budget=small_budget))
    assert small.update is ContextUpdate.UPDATED
    assert small.omitted_supporting > 0
    assert not any(
        isinstance(hit.memory, ObservationMemory) for hit in small.supporting
    )
    assert small.used_tokens <= small_budget
    session.commit(small)
    expanded = await session.prepare(request(query="task", budget=20000))
    assert expanded.update is ContextUpdate.UPDATED
    assert expanded.supporting and expanded.omitted_supporting == 0
    assert embedder.calls > 0


async def test_revised_and_removed_support_is_withdrawn_but_never_omitted_hits(memory):
    runtime, _ = memory
    original = await save(
        runtime, ObservationMemory(summary="Relevant", observation="OLD_SUPPORT")
    )
    await save(runtime, ObservationMemory(summary="Long", observation="z" * 3000))
    session = SessionContext(runtime, owner_id="alice", count_tokens=len)
    small = await session.prepare(request(query="task", budget=1600))
    assert [hit.memory.id for hit in small.supporting] == [original.memory.id]
    assert small.omitted_supporting == 1
    session.commit(small)
    repeated = await session.prepare(request(query="task", budget=1600))
    assert repeated.update is ContextUpdate.UNCHANGED
    session.commit(repeated)
    replacement = await save(
        runtime,
        ObservationMemory(summary="Relevant", observation="NEW_SUPPORT"),
        decision=Supersede(target_id=original.memory.id, expected_revision=1),
    )
    changed = await session.prepare(request(query="task", budget=1600))
    assert changed.update is ContextUpdate.UPDATED
    assert [(item.id, item.revision) for item in changed.withdrawn] == [
        (original.memory.id, 1)
    ]
    assert "OLD_SUPPORT" not in changed.text and "NEW_SUPPORT" in changed.text
    session.commit(changed)
    removed = await session.prepare(request())
    assert [item.id for item in removed.withdrawn] == [replacement.memory.id]
    assert "NEW_SUPPORT" not in removed.text
    assert not removed.supporting


async def test_returned_model_mutation_cannot_change_private_snapshot_or_receipt_context(
    memory,
):
    runtime, _ = memory
    await save(runtime, pref())
    session = SessionContext(runtime, owner_id="alice", count_tokens=len)
    first = await session.prepare(request())
    with pytest.raises(ContextRequiredError, match="commit"):
        await session.before_action(first)
    session.commit(first)
    first.context.attributes["tools"] = ["untrusted-change"]
    first.preferences.effective[0].metadata["tools"] = ["untrusted-change"]
    await session.before_action(first)
    next_turn = await session.prepare(request())
    assert next_turn.update is ContextUpdate.UNCHANGED
    assert not next_turn.context.attributes
    assert "untrusted-change" not in next_turn.preferences.model_dump_json()


async def test_session_context_response_serialization_roundtrips(memory):
    from memory_core.integrations import SessionContextResponse

    runtime, _ = memory
    await save(runtime, pref())
    session = SessionContext(runtime, owner_id="alice", count_tokens=len)
    response = await session.prepare(request())
    assert (
        SessionContextResponse.model_validate_json(response.model_dump_json())
        == response
    )


async def test_provider_compaction_reset_reinjects_without_new_client_or_embedding(
    memory,
):
    runtime, embedder = memory
    await save(runtime, pref())
    session = SessionContext(runtime, owner_id="alice", count_tokens=len)
    client = PersistentClient("local")
    first = await session.prepare(request())
    await client.send(first.text)
    session.commit(first)
    assert (await session.prepare(request())).update is ContextUpdate.UNCHANGED
    session.reset()
    with pytest.raises(ContextRequiredError):
        session.commit(first)
    refreshed = await session.prepare(request())
    assert refreshed.update is ContextUpdate.INITIAL and refreshed.text
    await client.send(refreshed.text)
    session.commit(refreshed)
    assert client.session_id == "existing-session" and len(client.messages) == 2
    assert embedder.calls == 0


async def test_support_already_present_in_preferences_is_counted_as_represented_not_omitted(
    memory,
):
    runtime, _ = memory
    preference = await save(runtime, pref())
    session = SessionContext(runtime, owner_id="alice", count_tokens=len)
    result = await session.prepare(request(query="task"))
    assert result.preferences.effective[0].id == preference.memory.id
    assert not result.supporting
    assert result.deduplicated_supporting == 1 and result.omitted_supporting == 0
    assert result.text.count(preference.memory.id) == 1
    session.commit(result)
    unchanged = await session.prepare(request(query="task"))
    assert unchanged.update is ContextUpdate.UNCHANGED
    assert unchanged.deduplicated_supporting == 1


@pytest.mark.parametrize("change", ["reset", "prepare", "prepare_and_commit"])
async def test_guard_rechecks_session_identity_after_receipt_validation(
    memory, monkeypatch, change
):
    runtime, _ = memory
    await save(runtime, pref())
    session = SessionContext(runtime, owner_id="alice", count_tokens=len)
    current = await session.prepare(request())
    session.commit(current)
    validated, release = asyncio.Event(), asyncio.Event()
    original_validate = runtime.service.validate_context

    async def validate_then_pause(receipt, context, *, owner_id):
        await original_validate(receipt, context, owner_id=owner_id)
        validated.set()
        await release.wait()

    monkeypatch.setattr(runtime.service, "validate_context", validate_then_pause)
    guard = asyncio.create_task(session.before_action(current))
    try:
        await asyncio.wait_for(validated.wait(), 1)
        if change == "reset":
            session.reset()
        else:
            next_context = await session.prepare(request(task="next-turn"))
            if change == "prepare_and_commit":
                session.commit(next_context)
        release.set()
        with pytest.raises(ContextRequiredError, match="changed while validating"):
            await guard
    finally:
        release.set()
        await asyncio.gather(guard, return_exceptions=True)
