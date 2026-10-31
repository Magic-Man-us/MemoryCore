"""Long-term store: durable traces + embeddings in the shared SQL database."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from sqlalchemy import delete, select, update

from memory_core.core.models import MemoryTrace

from .codec import decode_embedding, decode_object, decode_tags, encode_embedding
from .database import Database, LongTermTraceORM

if TYPE_CHECKING:
    from collections.abc import Iterable


class LongTermStore:
    """SQL-backed storage for consolidated, durable traces.

    Upserts use ``Session.merge`` (portable across SQLite/PostgreSQL/MySQL) rather
    than a dialect-specific ``ON CONFLICT``/``ON DUPLICATE KEY`` statement.
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    @property
    def db(self) -> Database:
        return self._db

    def upsert_trace(
        self,
        trace: MemoryTrace,
        embedding: Iterable[float] | None = None,
    ) -> None:
        """Persist the trace (metadata + content) and optional embedding vector."""
        # mode="json" coerces non-native values (datetimes, paths, ...) the same way
        # the STM store does, so one trace serializes identically in both stores.
        extra_json = trace.model_dump(mode="json", include={"extra"})["extra"]

        row = LongTermTraceORM(
            trace_uid=trace.trace_uid,
            user_id=trace.user_id,
            content=trace.content or None,
            summary=trace.summary,
            importance=trace.importance,
            created_at=trace.created_at,
            access_count=trace.access_count,
            tags=json.dumps(sorted(trace.tags)),
            extra=json.dumps(extra_json) if extra_json else None,
            embedding=encode_embedding(embedding),
        )
        with self._db.session_scope() as session:
            session.merge(row)
            session.commit()

    def fetch_traces_for_user(
        self,
        user_id: str,
    ) -> list[tuple[MemoryTrace, list[float] | None]]:
        """All of a user's traces with their stored embeddings: the hydration read path."""
        stmt = (
            select(LongTermTraceORM)
            .where(LongTermTraceORM.user_id == user_id)
            .order_by(LongTermTraceORM.created_at)
        )
        with self._db.session_scope() as session:
            rows = session.execute(stmt).scalars().all()
            return [
                (self._row_to_trace(row), decode_embedding(row.embedding))
                for row in rows
            ]

    def increment_access(self, trace_uids: Iterable[str]) -> int:
        """Add one to ``access_count`` for each trace; returns the rows updated."""
        uids = list(dict.fromkeys(trace_uids))
        if not uids:
            return 0
        with self._db.session_scope() as session:
            result = session.execute(
                update(LongTermTraceORM)
                .where(LongTermTraceORM.trace_uid.in_(uids))
                .values(access_count=LongTermTraceORM.access_count + 1)
            )
            session.commit()
            return int(result.rowcount or 0)

    def delete_trace(self, trace_uid: str) -> bool:
        """Remove a trace permanently. Returns True when a row was deleted."""
        with self._db.session_scope() as session:
            result = session.execute(
                delete(LongTermTraceORM).where(LongTermTraceORM.trace_uid == trace_uid)
            )
            session.commit()
            return bool(result.rowcount)

    @staticmethod
    def _row_to_trace(row: LongTermTraceORM) -> MemoryTrace:
        return MemoryTrace(
            trace_uid=row.trace_uid,
            user_id=row.user_id,
            content=row.content or "",
            summary=row.summary,
            importance=row.importance,
            created_at=row.created_at,
            access_count=row.access_count,
            tags=decode_tags(row.tags),
            extra=decode_object(row.extra),
        )
