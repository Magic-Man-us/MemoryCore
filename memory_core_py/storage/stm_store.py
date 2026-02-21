from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from sqlalchemy import func, select
from sqlalchemy.dialects.mysql import insert

from memory_core_py.core.models import MemoryTrace

from .database import ShortTermTraceORM
from .settings import ShortTermStoreSettings, build_engine_and_session

if TYPE_CHECKING:
    from datetime import datetime

    from sqlalchemy.orm import Session


class ShortTermStore:
    """MariaDB-backed cache that keeps short-lived traces for fast recall."""

    def __init__(
        self,
        *,
        settings: ShortTermStoreSettings | None = None,
        **overrides: Any,
    ) -> None:
        base_settings = settings or ShortTermStoreSettings()
        (
            self._settings,
            self._engine,
            self._session_factory,
        ) = build_engine_and_session(base_settings, overrides)

    def _session(self) -> Session:
        """Create a new SQLAlchemy session."""
        return self._session_factory()

    def insert_trace(self, trace: MemoryTrace, expires_at: datetime) -> None:
        """Insert or upsert the given trace with an explicit expiration time."""
        insert_values = self._serialize_trace(trace, expires_at)

        stmt = insert(ShortTermTraceORM).values(**insert_values)

        upsert_columns = (
            "content",
            "summary",
            "importance",
            "access_count",
            "last_accessed",
            "expires_at",
            "tags",
            "extra",
        )
        stmt = stmt.on_duplicate_key_update(
            **{column: getattr(stmt.inserted, column) for column in upsert_columns}
        )

        session = self._session()
        try:
            session.execute(stmt)
            session.commit()
        finally:
            session.close()

    def fetch_recent_for_user(
        self,
        user_id: str,
        limit: int = 50,
    ) -> list[MemoryTrace]:
        """Load the most recent non-expired traces for a user."""
        session = self._session()
        try:
            stmt = (
                select(ShortTermTraceORM)
                .where(ShortTermTraceORM.user_id == user_id)
                .where(ShortTermTraceORM.expires_at > func.now())
                .order_by(ShortTermTraceORM.created_at.desc())
                .limit(limit)
            )

            results = session.execute(stmt).scalars().all()

            return [self._row_to_trace(row) for row in results]
        finally:
            session.close()

    def _serialize_trace(
        self,
        trace: MemoryTrace,
        expires_at: datetime,
    ) -> dict[str, Any]:
        """Prepare trace fields for database persistence."""
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
        """Convert an ORM row back into a MemoryTrace."""
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
