"""Long-term store: durable traces + embeddings in the shared SQL database."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from sqlalchemy import delete, select

from memory_core.core.models import MemoryTrace

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

    def upsert_trace(
        self,
        trace: MemoryTrace,
        embedding: Iterable[float] | None = None,
    ) -> None:
        """Persist the trace (metadata + content) and optional embedding vector."""
        emb_bytes = None
        if embedding is not None:
            emb_bytes = json.dumps(list(embedding)).encode("utf-8")

        row = LongTermTraceORM(
            trace_uid=trace.trace_uid,
            user_id=trace.user_id,
            content=trace.content or None,
            summary=trace.summary,
            importance=trace.importance,
            created_at=trace.created_at,
            access_count=trace.access_count,
            tags=json.dumps(sorted(trace.tags)),
            extra=json.dumps(trace.extra) if trace.extra else None,
            embedding=emb_bytes,
        )
        session = self._db.session()
        try:
            session.merge(row)
            session.commit()
        finally:
            session.close()

    def fetch_traces_for_user(
        self,
        user_id: str,
    ) -> list[tuple[MemoryTrace, list[float] | None]]:
        """All of a user's traces with their stored embeddings — the hydration read path."""
        session = self._db.session()
        try:
            stmt = (
                select(LongTermTraceORM)
                .where(LongTermTraceORM.user_id == user_id)
                .order_by(LongTermTraceORM.created_at)
            )
            rows = session.execute(stmt).scalars().all()
            return [(self._row_to_trace(row), self._decode_embedding(row.embedding)) for row in rows]
        finally:
            session.close()

    def delete_trace(self, trace_uid: str) -> bool:
        """Remove a trace permanently. Returns True when a row was deleted."""
        session = self._db.session()
        try:
            result = session.execute(
                delete(LongTermTraceORM).where(LongTermTraceORM.trace_uid == trace_uid)
            )
            session.commit()
            return bool(result.rowcount)
        finally:
            session.close()

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
            tags=_decode_tags(row.tags),
            extra=_decode_extra(row.extra),
        )

    @staticmethod
    def _decode_embedding(raw: bytes | None) -> list[float] | None:
        if not raw:
            return None
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
        if isinstance(decoded, list):
            return [float(x) for x in decoded]
        return None


def _decode_tags(raw_tags: str | None) -> set[str]:
    if not raw_tags:
        return set()
    try:
        tags = json.loads(raw_tags)
    except json.JSONDecodeError:
        return set()
    return set(tags) if isinstance(tags, list) else set()


def _decode_extra(raw_extra: str | None) -> dict[str, object]:
    if not raw_extra:
        return {}
    try:
        extra = json.loads(raw_extra)
    except json.JSONDecodeError:
        return {}
    return extra if isinstance(extra, dict) else {}
