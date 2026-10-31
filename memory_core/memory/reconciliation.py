"""Atomic reconciliation decisions with scope and revision checks."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from sqlalchemy import text

from memory_core.storage.codec import encode_embedding

from .errors import ConflictError, SchemaError
from .models import (
    AttachEvidence,
    Create,
    FlagConflict,
    PreferenceMemory,
    PreserveException,
    ProjectScope,
    Status,
    Supersede,
)


def equivalence_key(memory):
    """Canonical structured content; lifecycle and evidence do not change identity."""
    excluded = {
        "id",
        "created_at",
        "updated_at",
        "revision",
        "source_refs",
        "supersedes_id",
    }
    return json.dumps(memory.model_dump(mode="json", exclude=excluded), sort_keys=True)


def equivalent(a, b):
    return equivalence_key(a) == equivalence_key(b)


def apply_reconciliation(
    service, session, owner, cache, memory, vector, decision, allowed
):
    if memory.id in cache.memories:
        raise ConflictError("new memory id already exists")
    target = None
    if not isinstance(decision, Create):
        target = cache.memories.get(decision.target_id)
        if target is None or target.id not in allowed:
            raise ConflictError(
                "reconciliation target is outside the authorized candidate set"
            )
        if (
            target.revision != decision.expected_revision
            or target.status != Status.ACTIVE
        ):
            raise ConflictError("target revision/status changed")
        if isinstance(decision, (AttachEvidence, Supersede, FlagConflict)):
            if target.kind != memory.kind or target.scope != memory.scope:
                raise SchemaError("merge/supersede requires the same kind and scope")
            if isinstance(memory, PreferenceMemory) and (
                target.preference_key != memory.preference_key
                or target.applicability != memory.applicability
            ):
                raise SchemaError("preference keys and applicability must match")
            if (
                isinstance(decision, AttachEvidence)
                and isinstance(memory, PreferenceMemory)
                and (
                    json.dumps(target.value, sort_keys=True)
                    != json.dumps(memory.value, sort_keys=True)
                )
            ):
                raise SchemaError(
                    "cannot attach contradictory preference values as evidence"
                )
        if isinstance(decision, PreserveException):
            if not isinstance(memory.scope, ProjectScope) or isinstance(
                target.scope, ProjectScope
            ):
                raise SchemaError(
                    "an exception requires a project memory and global target"
                )
            if (
                not isinstance(memory, PreferenceMemory)
                or not isinstance(target, PreferenceMemory)
                or memory.preference_key != target.preference_key
            ):
                raise SchemaError("an exception requires matching preference keys")
    generation = service.store.bump(
        session,
        owner,
        decision.action,
        {
            "new_id": memory.id,
            "target_id": getattr(decision, "target_id", None),
            "sources": memory.source_refs,
        },
    )
    now = datetime.now(UTC)
    if isinstance(decision, AttachEvidence):
        updated = target.model_copy(
            update={
                "source_refs": sorted(set(target.source_refs + memory.source_refs)),
                "revision": target.revision + 1,
                "updated_at": now,
            }
        )
        raw = session.execute(
            text("SELECT embedding FROM memories_v1 WHERE owner=:o AND id=:id"),
            {"o": owner, "id": target.id},
        ).scalar_one()
        service.store.put(session, owner, updated, raw, generation)
        return updated, generation
    if isinstance(decision, (Supersede, FlagConflict)):
        status = (
            Status.SUPERSEDED if isinstance(decision, Supersede) else Status.DISPUTED
        )
        updated = target.model_copy(
            update={
                "status": status,
                "revision": target.revision + 1,
                "updated_at": now,
            }
        )
        raw = session.execute(
            text("SELECT embedding FROM memories_v1 WHERE owner=:o AND id=:id"),
            {"o": owner, "id": target.id},
        ).scalar_one()
        service.store.put(session, owner, updated, raw, generation)
        memory = memory.model_copy(
            update={"supersedes_id": target.id}
            if isinstance(decision, Supersede)
            else {"status": Status.DISPUTED}
        )
    service.store.put(session, owner, memory, encode_embedding(vector), generation)
    return memory, generation
