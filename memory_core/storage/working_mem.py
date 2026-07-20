"""Working memory: rolling per-user session events in the shared SQL database.

Replaces the former Redis list-with-TTL. Same semantics, no service dependency:

- per-user event stream, newest last
- TTL expiry per event (like working-memory decay)
- a hard per-user cap so a chatty session can't grow the table without bound
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import delete, func, select

from .database import Database, WorkingMemoryEventORM


class SqlWorkingMemory:
    """Short-lived session state stored as TTL'd rows, per user."""

    def __init__(
        self,
        db: Database,
        *,
        ttl_seconds: int = 3600,
        max_events: int = 500,
    ) -> None:
        self._db = db
        self.ttl_seconds = ttl_seconds
        self.max_events = max_events

    def add_event(self, user_id: str, payload: dict[str, Any]) -> None:
        """Append one event for the user; opportunistically prunes expired/overflow rows.

        The server-stamped ``ts`` is authoritative — a ``ts`` key in the payload is
        overwritten, not trusted. A payload that cannot be JSON-serialized raises a
        clear ``ValueError`` before anything touches the database.
        """
        now = datetime.now(UTC)
        doc = {**payload, "ts": now.isoformat()}
        try:
            payload_json = json.dumps(doc)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"working-memory event payload must be JSON-serializable: {exc}"
            ) from exc
        session = self._db.session()
        try:
            session.add(
                WorkingMemoryEventORM(
                    user_id=user_id,
                    created_at=now,
                    expires_at=now + timedelta(seconds=self.ttl_seconds),
                    payload=payload_json,
                )
            )
            # Prune this user's expired events on every write (cheap, indexed).
            session.execute(
                delete(WorkingMemoryEventORM)
                .where(WorkingMemoryEventORM.user_id == user_id)
                .where(WorkingMemoryEventORM.expires_at <= now)
            )
            session.commit()
            self._enforce_cap(session, user_id)
        finally:
            session.close()

    def get_recent(self, user_id: str, limit: int = 20) -> list[dict[str, Any]]:
        """The user's most recent non-expired events, oldest first."""
        session = self._db.session()
        try:
            stmt = (
                select(WorkingMemoryEventORM.payload)
                .where(WorkingMemoryEventORM.user_id == user_id)
                .where(WorkingMemoryEventORM.expires_at > datetime.now(UTC))
                .order_by(WorkingMemoryEventORM.id.desc())
                .limit(limit)
            )
            rows = session.execute(stmt).scalars().all()
        finally:
            session.close()

        out: list[dict[str, Any]] = []
        for raw in reversed(rows):
            try:
                loaded = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(loaded, dict):
                out.append(loaded)
        return out

    def clear(self, user_id: str) -> None:
        """Drop all of a user's working-memory events."""
        session = self._db.session()
        try:
            session.execute(
                delete(WorkingMemoryEventORM).where(WorkingMemoryEventORM.user_id == user_id)
            )
            session.commit()
        finally:
            session.close()

    def _enforce_cap(self, session: Any, user_id: str) -> None:
        count = session.execute(
            select(func.count())
            .select_from(WorkingMemoryEventORM)
            .where(WorkingMemoryEventORM.user_id == user_id)
        ).scalar_one()
        overflow = count - self.max_events
        if overflow > 0:
            oldest_ids = session.execute(
                select(WorkingMemoryEventORM.id)
                .where(WorkingMemoryEventORM.user_id == user_id)
                .order_by(WorkingMemoryEventORM.id)
                .limit(overflow)
            ).scalars()
            session.execute(
                delete(WorkingMemoryEventORM).where(
                    WorkingMemoryEventORM.id.in_(list(oldest_ids))
                )
            )
            session.commit()
