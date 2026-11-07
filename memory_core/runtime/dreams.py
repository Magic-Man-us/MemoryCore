"""Durable, finite dream scans; continuation and acknowledgement share a transaction."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import TypeAdapter
from sqlalchemy import text

from memory_core.memory.dream import DreamScan
from memory_core.storage.codec import JSON_VALUE

from .models import JobKind

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from memory_core.memory.models import DreamPlan

    from .queue import Claim, Queue


SCAN_ADAPTER = TypeAdapter(DreamScan)


def scan_for(queue: Queue, claim: Claim) -> DreamScan:
    """Pin a high-water mark once, including across retries and process restarts."""
    with (
        queue.service._owner_lock(claim.owner),
        queue.transaction(claim.owner) as session,
    ):
        queue.assert_lease(session, claim)
        payload = JSON_VALUE.validate_json(
            session.execute(
                text("SELECT payload FROM memory_jobs WHERE owner=:o AND id=:id"),
                {"o": claim.owner, "id": claim.id},
            ).scalar_one()
        )
        if "scan" not in payload:
            state = queue.service.store.state(session, claim.owner)
            payload["scan"] = {
                "cursor": state["dream_cursor"],
                "cursor_id": state["dream_cursor_id"],
                "end": state["generation"],
                "writes": state["writes"],
            }
            session.execute(
                text("UPDATE memory_jobs SET payload=:p WHERE owner=:o AND id=:id"),
                {
                    "p": JSON_VALUE.dump_json(payload).decode(),
                    "o": claim.owner,
                    "id": claim.id,
                },
            )
        return SCAN_ADAPTER.validate_python(payload["scan"])


def advance(queue: Queue, session: Session, claim: Claim, plan: DreamPlan) -> None:
    """Advance after application OR review handoff. A retry cannot fork the scan."""
    row = session.execute(
        text("SELECT payload FROM memory_jobs WHERE owner=:o AND id=:id"),
        {"o": claim.owner, "id": claim.id},
    ).scalar_one()
    payload = JSON_VALUE.validate_json(row)
    if payload.get("continued") or "scan" not in payload:
        return
    scan = payload["scan"]
    more = session.execute(
        text("""SELECT 1 FROM memories_v1 WHERE owner=:o AND generation<=:end
        AND (generation>:c OR (generation=:c AND id>:ci)) LIMIT 1"""),
        {"o": claim.owner, "end": scan["end"], "c": plan.cursor, "ci": plan.cursor_id},
    ).first()
    if more:
        queue.enqueue(
            session,
            claim.owner,
            JobKind.DREAM,
            {"scan": {**scan, "cursor": plan.cursor, "cursor_id": plan.cursor_id}},
        )
    else:
        # Completion accounts only for writes present when this pass began.
        # Later writes retain their trigger and are handled by a later finite pass.
        session.execute(
            text(
                "UPDATE memory_owners SET dream_writes=MAX(dream_writes,:w) WHERE owner=:o"
            ),
            {"o": claim.owner, "w": scan["writes"]},
        )
    state = queue.service.store.state(session, claim.owner)
    if (plan.cursor, plan.cursor_id) > (
        state["dream_cursor"],
        state["dream_cursor_id"],
    ):
        session.execute(
            text(
                "UPDATE memory_owners SET dream_cursor=:c,dream_cursor_id=:ci WHERE owner=:o"
            ),
            {"o": claim.owner, "c": plan.cursor, "ci": plan.cursor_id},
        )
    payload["continued"] = True
    session.execute(
        text("UPDATE memory_jobs SET payload=:p WHERE owner=:o AND id=:id"),
        {"p": JSON_VALUE.dump_json(payload).decode(), "o": claim.owner, "id": claim.id},
    )
    if not more:
        queue.schedule_dream(session, claim.owner)
