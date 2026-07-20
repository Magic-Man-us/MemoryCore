"""Short-term store: TTL'd traces for fast recency recall, in the shared SQL database."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import delete, select

from memory_core.core.models import MemoryTrace

from .database import Database, ShortTermTraceORM


class ShortTermStore:
    """SQL-backed cache that keeps short-lived traces for fast recall."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def insert_trace(self, trace: MemoryTrace, expires_at: datetime) -> None:
        """Insert or update the given trace with an explicit expiration time."""
        row = ShortTermTraceORM(**self._serialize_trace(trace, expires_at))
        session = self._db.session()
        try:
            session.merge(row)
            session.commit()
        finally:
            session.close()

    def fetch_recent_for_user(
        self,
        user_id: str,
        limit: int = 50,
    ) -> list[MemoryTrace]:
        """The most recent non-expired traces for a user."""
        session = self._db.session()
        try:
            stmt = (
                select(ShortTermTraceORM)
                .where(ShortTermTraceORM.user_id == user_id)
                .where(ShortTermTraceORM.expires_at > datetime.now(UTC))
                .order_by(ShortTermTraceORM.created_at.desc())
                .limit(limit)
            )
            results = session.execute(stmt).scalars().all()
            return [self._row_to_trace(row) for row in results]
        finally:
            session.close()

    def purge_expired(self) -> int:
        """Delete expired rows; returns the number removed."""
        session = self._db.session()
        try:
            result = session.execute(
                delete(ShortTermTraceORM).where(
                    ShortTermTraceORM.expires_at <= datetime.now(UTC)
                )
            )
            session.commit()
            return int(result.rowcount or 0)
        finally:
            session.close()

    def _serialize_trace(
        self,
        trace: MemoryTrace,
        expires_at: datetime,
    ) -> dict[str, Any]:
        base_payload = trace.model_dump(exclude={"tags", "extra"})
        json_payload = trace.model_dump(mode="json", include={"tags", "extra"})
        return {
            **base_payload,
            "last_accessed": trace.created_at,
            "expires_at": expires_at,
            "tags": json.dumps(json_payload["tags"]),
            "extra": json.dumps(json_payload["extra"]),
        }

    def _row_to_trace(self, row: ShortTermTraceORM) -> MemoryTrace:
        return MemoryTrace.model_validate(
            {
                "trace_uid": row.trace_uid,
                "user_id": row.user_id,
                "content": row.content,
                "summary": row.summary,
                "importance": row.importance,
                "access_count": row.access_count,
                "created_at": row.created_at,
                "tags": self._decode_tags(row.tags),
                "extra": self._decode_extra(row.extra),
            }
        )

    @staticmethod
    def _decode_tags(raw_tags: str | None) -> set[str]:
        if not raw_tags:
            return set()
        try:
            tags = json.loads(raw_tags)
        except json.JSONDecodeError:
            return set()
        return set(tags) if isinstance(tags, list) else set()

    @staticmethod
    def _decode_extra(raw_extra: str | None) -> dict[str, Any]:
        if not raw_extra:
            return {}
        try:
            extra = json.loads(raw_extra)
        except json.JSONDecodeError:
            return {}
        return extra if isinstance(extra, dict) else {}
