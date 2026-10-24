"""Short-term store: TTL'd traces for fast recency recall, in the shared SQL database."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import delete, select

from memory_core.core.models import MemoryTrace

from .codec import decode_object, decode_tags
from .database import Database, ShortTermTraceORM


class ShortTermStore:
    """SQL-backed cache that keeps short-lived traces for fast recall."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def insert_trace(self, trace: MemoryTrace, expires_at: datetime) -> None:
        """Insert or update the given trace with an explicit expiration time."""
        row = ShortTermTraceORM(**self._serialize_trace(trace, expires_at))
        with self._db.session_scope() as session:
            session.merge(row)
            session.commit()

    def fetch_recent_for_user(
        self,
        user_id: str,
        limit: int = 50,
    ) -> list[MemoryTrace]:
        """The most recent non-expired traces for a user."""
        stmt = (
            select(ShortTermTraceORM)
            .where(ShortTermTraceORM.user_id == user_id)
            .where(ShortTermTraceORM.expires_at > datetime.now(UTC))
            .order_by(ShortTermTraceORM.created_at.desc())
            .limit(limit)
        )
        with self._db.session_scope() as session:
            results = session.execute(stmt).scalars().all()
            return [self._row_to_trace(row) for row in results]

    def delete_trace(self, trace_uid: str) -> bool:
        """Remove one trace. Returns True when a row was deleted."""
        with self._db.session_scope() as session:
            result = session.execute(
                delete(ShortTermTraceORM).where(ShortTermTraceORM.trace_uid == trace_uid)
            )
            session.commit()
            return bool(result.rowcount)

    def purge_expired(self) -> int:
        """Delete expired rows; returns the number removed."""
        with self._db.session_scope() as session:
            result = session.execute(
                delete(ShortTermTraceORM).where(
                    ShortTermTraceORM.expires_at <= datetime.now(UTC)
                )
            )
            session.commit()
            return int(result.rowcount or 0)

    @staticmethod
    def _serialize_trace(trace: MemoryTrace, expires_at: datetime) -> dict[str, Any]:
        base_payload = trace.model_dump(exclude={"tags", "extra"})
        json_payload = trace.model_dump(mode="json", include={"tags", "extra"})
        return {
            **base_payload,
            "last_accessed": trace.created_at,
            "expires_at": expires_at,
            "tags": json.dumps(json_payload["tags"]),
            "extra": json.dumps(json_payload["extra"]),
        }

    @staticmethod
    def _row_to_trace(row: ShortTermTraceORM) -> MemoryTrace:
        return MemoryTrace.model_validate(
            {
                "trace_uid": row.trace_uid,
                "user_id": row.user_id,
                "content": row.content,
                "summary": row.summary,
                "importance": row.importance,
                "access_count": row.access_count,
                "created_at": row.created_at,
                "tags": decode_tags(row.tags),
                "extra": decode_object(row.extra),
            }
        )
