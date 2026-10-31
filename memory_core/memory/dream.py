"""Bounded consolidation passes. Planning is separate from atomic, revision-checked application."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from sqlalchemy import text

from memory_core.storage.codec import decode_embedding

from .errors import ConflictError, SchemaError
from .models import (
    AttachEvidence,
    Create,
    DreamChange,
    DreamGroup,
    DreamPlan,
    DreamProposal,
    DreamResponse,
    MemoryHit,
    Status,
)
from .reconciliation import apply_reconciliation, equivalence_key, equivalent


def due(service, owner):
    with service._lock, service.store.transaction(owner) as session:
        state = service.store.state(session, owner)
        return state["writes"] - state["dream_writes"] >= service.dream_after_writes


async def plan(service, owner, request):
    def snapshot():
        with service._lock, service.store.transaction(owner) as session:
            cache, _, state = service._snapshot(session, owner)
            # Walk changed records in generation order; ties are included using
            # a stable ID cursor encoded in the plan's source batch (no full payload scan).
            rows = session.execute(
                text("""SELECT id,embedding,generation FROM memories_v1
                WHERE owner=:o AND (generation>:cursor OR (generation=:cursor AND id>:cursor_id)) ORDER BY generation,id LIMIT :limit"""),
                {
                    "o": owner,
                    "cursor": state["dream_cursor"],
                    "cursor_id": state["dream_cursor_id"],
                    "limit": request.batch_size,
                },
            ).all()
            if not rows:
                # Revisit old records too: consolidation can improve after new evidence arrives.
                rows = session.execute(
                    text("""SELECT id,embedding,generation FROM memories_v1
                    WHERE owner=:o ORDER BY generation,id LIMIT :limit"""),
                    {"o": owner, "limit": request.batch_size},
                ).all()
            groups = []
            for uid, raw, _ in rows:
                memory = cache.memories[uid]
                if memory.status != Status.ACTIVE or not raw:
                    continue
                ids = cache.eligible(getattr(memory.scope, "project_id", None)) - {uid}
                hits = cache.search(
                    memory.summary,
                    decode_embedding(raw),
                    ids,
                    request.neighbors_per_memory,
                )
                # Prefer one exact duplicate within the existing neighbor budget.
                # This remains useful even when vector similarity is zero.
                exact = cache.equivalents.get(equivalence_key(memory), set()) & ids
                if exact:
                    target = min(exact)
                    hits = [(target, 0.0)] + [(i, s) for i, s in hits if i != target]
                    hits = hits[: request.neighbors_per_memory]
                groups.append(
                    DreamGroup(
                        memory=memory.model_copy(deep=True),
                        neighbors=[
                            MemoryHit(
                                memory=cache.memories[i].model_copy(deep=True), score=s
                            )
                            for i, s in hits
                        ],
                    )
                )
            cursor = rows[-1][2] if rows else state["dream_cursor"]
            cursor_id = rows[-1][0] if rows else state["dream_cursor_id"]
            return groups, dict(state), cursor, cursor_id

    groups, state, cursor, cursor_id = await asyncio.to_thread(snapshot)
    if service.dream_planner is not None:
        proposal = DreamProposal.model_validate(await service.dream_planner(groups))
    else:
        # Local deterministic baseline: consolidate exact structured duplicates only.
        changes = []
        consumed = set()
        for group in groups:
            for hit in group.neighbors:
                a, b = group.memory, hit.memory
                if a.id in consumed or b.id in consumed or a.id <= b.id:
                    continue
                if equivalent(a, b):
                    # Exact duplicate retirement is represented by a dedicated merge below.
                    changes.append(
                        DreamChange(
                            memory=a,
                            decision=AttachEvidence(
                                target_id=b.id, expected_revision=b.revision
                            ),
                        )
                    )
                    consumed.update((a.id, b.id))
                    break
        proposal = DreamProposal(changes=changes)
    result = DreamPlan(
        generation=state["generation"],
        cursor=cursor,
        cursor_id=cursor_id,
        writes=state["writes"],
        groups=groups,
        proposal=proposal,
    )

    def save():
        with service._lock, service.store.transaction(owner) as session:
            current = service.store.state(session, owner)
            if current["generation"] != result.generation:
                raise ConflictError("memory changed while dreaming; replan")
            session.execute(
                text("INSERT INTO memory_dreams VALUES (:o,:id,:p)"),
                {"o": owner, "id": result.id, "p": result.model_dump_json()},
            )
            # Pending proposals count as scheduled work; writes since this snapshot trigger the next pass.
            session.execute(
                text("UPDATE memory_owners SET dream_writes=:w WHERE owner=:o"),
                {"o": owner, "w": result.writes},
            )
        return DreamResponse(plan=result, due=False)

    return await asyncio.to_thread(save)


async def apply(service, owner, plan_id, *, on_commit=None):
    def load():
        with service._lock, service.store.transaction(owner) as session:
            raw = session.execute(
                text("SELECT payload FROM memory_dreams WHERE owner=:o AND id=:id"),
                {"o": owner, "id": plan_id},
            ).scalar_one_or_none()
            if raw is None:
                raise ConflictError("unknown dream plan")
            result = DreamPlan.model_validate_json(raw)
            if result.applied:
                raise ConflictError("dream plan already applied")
            if service.store.state(session, owner)["generation"] != result.generation:
                raise ConflictError("dream plan is stale; replan")
            return result

    result = await asyncio.to_thread(load)
    vectors = []
    originals = {g.memory.id: g.memory for g in result.groups}
    for change in result.proposal.changes:
        if change.memory.id in originals:
            vectors.append(None)  # exact duplicate retirement reuses existing data
        else:
            vectors.append(await service._vector(None, change.memory.summary))

    def commit():
        try:
            with service._lock, service.store.transaction(owner) as session:
                cache, registry, state = service._snapshot(session, owner)
                if state["generation"] != result.generation:
                    raise ConflictError("dream plan is stale; replan")
                # Also reject concurrent application of an empty plan (which may not bump generation).
                saved = session.execute(
                    text("SELECT payload FROM memory_dreams WHERE owner=:o AND id=:id"),
                    {"o": owner, "id": plan_id},
                ).scalar_one()
                if DreamPlan.model_validate_json(saved).applied:
                    raise ConflictError("dream plan already applied")
                allowed = {
                    m.id
                    for g in result.groups
                    for m in [g.memory, *[h.memory for h in g.neighbors]]
                }
                touched = set()
                for change, vector in zip(
                    result.proposal.changes, vectors, strict=True
                ):
                    memory = service._normalize(change.memory, registry)
                    decision = change.decision
                    if isinstance(decision, Create):
                        raise SchemaError(
                            "dream consolidation must reference an existing candidate"
                        )
                    if decision.target_id in touched or memory.id in touched:
                        raise SchemaError(
                            "dream changes must have disjoint source/target IDs"
                        )
                    touched.update((decision.target_id, memory.id))
                    if memory.id in cache.memories:
                        # Existing source can only be retired when it is exactly equivalent.
                        source = cache.memories[memory.id]
                        target = cache.memories.get(decision.target_id)
                        if (
                            memory.id not in originals
                            or not isinstance(decision, AttachEvidence)
                            or target is None
                            or target.id not in allowed
                            or source != memory
                            or target.revision != decision.expected_revision
                            or target.status != Status.ACTIVE
                            or not equivalent(source, target)
                        ):
                            raise SchemaError(
                                "existing dream sources require exact duplicate consolidation"
                            )
                        generation = service.store.bump(
                            session,
                            owner,
                            "dream_merge",
                            {
                                "plan_id": plan_id,
                                "source": source.id,
                                "target": target.id,
                            },
                            write=False,
                        )
                        now = datetime.now(UTC)
                        target = target.model_copy(
                            update={
                                "source_refs": sorted(
                                    set(target.source_refs + source.source_refs)
                                ),
                                "revision": target.revision + 1,
                                "updated_at": now,
                            }
                        )
                        source = source.model_copy(
                            update={
                                "status": Status.SUPERSEDED,
                                "revision": source.revision + 1,
                                "updated_at": now,
                            }
                        )
                        for item in (source, target):
                            raw = session.execute(
                                text(
                                    "SELECT embedding FROM memories_v1 WHERE owner=:o AND id=:id"
                                ),
                                {"o": owner, "id": item.id},
                            ).scalar_one()
                            service.store.put(session, owner, item, raw, generation)
                    else:
                        if (
                            memory.status != Status.ACTIVE
                            or memory.revision != 1
                            or memory.supersedes_id
                        ):
                            raise SchemaError(
                                "dream replacement must be a new active record"
                            )
                        apply_reconciliation(
                            service,
                            session,
                            owner,
                            cache,
                            memory,
                            vector,
                            decision,
                            allowed,
                        )
                applied = result.model_copy(update={"applied": True})
                session.execute(
                    text(
                        "UPDATE memory_dreams SET payload=:p WHERE owner=:o AND id=:id"
                    ),
                    {"p": applied.model_dump_json(), "o": owner, "id": plan_id},
                )
                service.store.bump(
                    session, owner, "dream_applied", {"plan_id": plan_id}, write=False
                )
                session.execute(
                    text(
                        "UPDATE memory_owners SET dream_cursor=:c,dream_cursor_id=:ci,dream_writes=writes WHERE owner=:o"
                    ),
                    {"c": result.cursor, "ci": result.cursor_id, "o": owner},
                )
                response = DreamResponse(plan=applied, due=False)
                if on_commit is not None:
                    on_commit(session, response)
                return response
        finally:
            # A failed transaction must never leave a speculative cache behind.
            with service._lock:
                service._caches.pop(owner, None)

    return await asyncio.to_thread(commit)
