"""Memory-first resolution with bounded providers and evidence-bound model output."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from pydantic import TypeAdapter
from sqlalchemy import text

from memory_core.memory.errors import ConflictError
from memory_core.memory.models import (
    MEMORY_ADAPTER,
    DescribeSchemaRequest,
    RecallRequest,
)

from .models import (
    KNOWLEDGE_ADAPTER,
    CaptureResponse,
    Evidence,
    Finding,
    KnowledgeGap,
    KnowledgeInput,
    KnowledgeResponse,
    ResolvedKnowledge,
    UnresolvedKnowledge,
)
from .queue import canonical

EVIDENCE_ADAPTER = TypeAdapter(list[Evidence])


def is_fresh(gap_json, result_json):
    if result_json is None:
        return False
    gap = KnowledgeGap.model_validate_json(gap_json)
    result = KNOWLEDGE_ADAPTER.validate_json(result_json)
    if not isinstance(result, ResolvedKnowledge):
        return False
    now = datetime.now(UTC)
    return all(
        0 <= (now - e.observed_at).total_seconds() <= gap.max_age_seconds
        for e in result.evidence
    )


def is_current(session, owner, gap_json, result_json):
    if not is_fresh(gap_json, result_json):
        return False
    result = KNOWLEDGE_ADAPTER.validate_json(result_json)
    for evidence in result.evidence:
        if not evidence.id.startswith("memory:"):
            continue
        try:
            memory_id, revision = evidence.id[len("memory:") :].rsplit(":", 1)
            revision = int(revision)
        except ValueError:
            return False
        raw = session.execute(
            text("SELECT payload FROM memories_v1 WHERE owner=:o AND id=:id"),
            {"o": owner, "id": memory_id},
        ).scalar_one_or_none()
        if raw is None:
            return False
        memory = MEMORY_ADAPTER.validate_json(raw)
        if memory.revision != revision or memory.status != "active":
            return False
    return True


async def resolve(runtime, request, owner, *, transaction_guard=None):
    service, queue = runtime.service, runtime.queue

    def begin():
        with service._lock, queue.transaction(owner) as session:
            if transaction_guard is not None:
                transaction_guard(session)
            task = queue.task(session, owner, request.task_id)
            if task.outcome is not None:
                raise ConflictError("cannot research a completed task")
            row = session.execute(
                text(
                    "SELECT gap,result FROM memory_gaps WHERE owner=:o AND task_id=:t AND id=:id"
                ),
                {"o": owner, "t": request.task_id, "id": request.gap.id},
            ).first()
            if row and row[0] != canonical(request.gap):
                raise ConflictError(
                    "gap ID reused with different question/requirements"
                )
            session.execute(
                text(
                    "INSERT OR IGNORE INTO memory_gaps(owner,task_id,id,gap) VALUES (:o,:t,:id,:g)"
                ),
                {
                    "o": owner,
                    "t": request.task_id,
                    "id": request.gap.id,
                    "g": canonical(request.gap),
                },
            )
            cached = (
                KNOWLEDGE_ADAPTER.validate_json(row[1])
                if row and is_current(session, owner, row[0], row[1])
                else None
            )
            return task, cached

    task, cached = await asyncio.to_thread(begin)
    if cached is not None:
        return KnowledgeResponse(result=cached)
    evidence, candidates = [], []
    result = UnresolvedKnowledge(reason="No knowledge evaluator is configured")
    try:
        async with asyncio.timeout(runtime.policy.provider_timeout_seconds):
            recalled = await service.recall(
                RecallRequest(
                    query=request.gap.question,
                    project_id=task.context.project_id,
                    limit=10,
                ),
                owner_id=owner,
            )
        candidates = recalled.hits
        for hit in candidates:
            m = hit.memory
            evidence.append(
                Evidence(
                    id=f"memory:{m.id}:{m.revision}",
                    uri=f"memory:{m.id}",
                    excerpt=m.summary,
                    observed_at=getattr(m, "observed_at", m.created_at),
                )
            )
        schema = await service.call(DescribeSchemaRequest(), owner_id=owner)
        now = datetime.now(UTC)
        evidence = [
            e
            for e in evidence
            if 0 <= (now - e.observed_at).total_seconds() <= request.gap.max_age_seconds
        ]
        input_value = KnowledgeInput(
            context=task.context,
            gap=request.gap,
            candidates=candidates,
            evidence=evidence,
            fields=schema.fields,
        )
        result = await evaluate(runtime, input_value)
        if not isinstance(result, ResolvedKnowledge) and runtime.researcher is not None:
            async with asyncio.timeout(runtime.policy.provider_timeout_seconds):
                additional = EVIDENCE_ADAPTER.validate_python(
                    await runtime.researcher(input_value.model_copy(deep=True))
                )
            if len(additional) > 32:
                raise ValueError("research returned more than 32 evidence records")
            known = {e.id: e for e in evidence}
            for item in additional:
                if item.id in known and known[item.id] != item:
                    raise ValueError("research changed an existing evidence ID")
                if (
                    0
                    <= (datetime.now(UTC) - item.observed_at).total_seconds()
                    <= request.gap.max_age_seconds
                ):
                    known[item.id] = item
            input_value = input_value.model_copy(
                update={"evidence": list(known.values())}
            )
            result = await evaluate(runtime, input_value)
    except Exception as exc:  # noqa: BLE001 — failures remain explicit unresolved gaps
        result = UnresolvedKnowledge(
            reason=f"Knowledge resolution failed: {type(exc).__name__}"
        )

    def finish():
        with service._lock, queue.transaction(owner) as session:
            if transaction_guard is not None:
                transaction_guard(session)
            current = queue.task(session, owner, request.task_id)
            if current.outcome is not None:
                raise ConflictError("task completed while research was running")
            old = session.execute(
                text(
                    "SELECT gap,result FROM memory_gaps WHERE owner=:o AND task_id=:t AND id=:id"
                ),
                {"o": owner, "t": request.task_id, "id": request.gap.id},
            ).first()
            if old and is_current(session, owner, old[0], old[1]):
                return KnowledgeResponse(result=KNOWLEDGE_ADAPTER.validate_json(old[1]))
            if isinstance(result, ResolvedKnowledge):
                snapshots = {
                    f"memory:{h.memory.id}:{h.memory.revision}": h.memory
                    for h in candidates
                }
                for evidence_item in result.evidence:
                    original = snapshots.get(evidence_item.id)
                    if original is None:
                        continue
                    raw = session.execute(
                        text(
                            "SELECT payload FROM memories_v1 WHERE owner=:o AND id=:id"
                        ),
                        {"o": owner, "id": original.id},
                    ).scalar_one_or_none()
                    current_memory = MEMORY_ADAPTER.validate_json(raw) if raw else None
                    if (
                        current_memory is None
                        or current_memory.revision != original.revision
                        or current_memory.status != "active"
                    ):
                        raise ConflictError(
                            "memory evidence changed during knowledge resolution; retry"
                        )
            capture = None
            if isinstance(result, ResolvedKnowledge):
                # The evidence and answer, plus queue insertion, commit together.
                event_id = (
                    "knowledge:"
                    + request.gap.id
                    + ":"
                    + str(
                        session.execute(
                            text(
                                "SELECT COUNT(*) FROM memory_events WHERE owner=:o AND task_id=:t"
                            ),
                            {"o": owner, "t": request.task_id},
                        ).scalar_one()
                    )
                )
                job = runtime._capture_finding(
                    session,
                    owner,
                    request.task_id,
                    event_id,
                    Finding(statement=result.answer, evidence=result.evidence),
                )
                capture = CaptureResponse(job_ids=[job])
            session.execute(
                text(
                    "UPDATE memory_gaps SET result=:r,resolved_at=:at WHERE owner=:o AND task_id=:t AND id=:id"
                ),
                {
                    "o": owner,
                    "t": request.task_id,
                    "id": request.gap.id,
                    "r": canonical(result),
                    "at": datetime.now(UTC).timestamp(),
                },
            )
            return KnowledgeResponse(result=result, capture=capture)

    return await asyncio.to_thread(finish)


async def evaluate(runtime, request):
    if runtime.knowledge_evaluator is None:
        return UnresolvedKnowledge(reason="No knowledge evaluator is configured")
    async with asyncio.timeout(runtime.policy.provider_timeout_seconds):
        result = KNOWLEDGE_ADAPTER.validate_python(
            await runtime.knowledge_evaluator(request.model_copy(deep=True))
        )
    if isinstance(result, ResolvedKnowledge):
        known = {e.id: e for e in request.evidence}
        if any(known.get(e.id) != e for e in result.evidence):
            raise ValueError("answer cites evidence outside the supplied snapshot")
        if not is_fresh(canonical(request.gap), canonical(result)):
            raise ValueError("answer uses expired or future evidence")
    return result
