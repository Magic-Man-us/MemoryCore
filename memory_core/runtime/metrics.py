"""On-demand local diagnostics. No payloads, telemetry service, or background polling."""

import time
from pathlib import Path

from pydantic import Field
from sqlalchemy import text

from .models import Model


class MemoryMeasurements(Model):
    owner_id: str
    memories_by_status: dict[str, int]
    jobs_by_status: dict[str, int]
    oldest_ready_wait_seconds: float | None
    cached_records: int
    records_after_dream_cursor: int
    # File/page measurements cover the whole SQLite database, not one owner.
    database_file_bytes: int
    wal_file_bytes: int
    allocated_database_bytes: int
    reusable_database_bytes: int
    sampled_at: float = Field(default_factory=time.time)


def measure(runtime, owner):
    if not owner:
        raise ValueError("authenticated owner must be nonempty")
    with runtime.service._lock, runtime.service.store.db.session_scope() as session:
        memories = dict(
            session.execute(
                text("""SELECT json_extract(payload,'$.status'),COUNT(*)
            FROM memories_v1 WHERE owner=:o GROUP BY json_extract(payload,'$.status')"""),
                {"o": owner},
            ).all()
        )
        jobs = dict(
            session.execute(
                text(
                    "SELECT status,COUNT(*) FROM memory_jobs WHERE owner=:o GROUP BY status"
                ),
                {"o": owner},
            ).all()
        )
        now = time.time()
        oldest = session.execute(
            text("""SELECT MIN(available) FROM memory_jobs
            WHERE owner=:o AND status='pending' AND available>0 AND available<=:now"""),
            {"o": owner, "now": now},
        ).scalar()
        remaining = session.execute(
            text("""SELECT COUNT(*) FROM memories_v1 m
            JOIN memory_owners o ON m.owner=o.owner WHERE m.owner=:owner
            AND (m.generation>o.dream_cursor OR (m.generation=o.dream_cursor AND m.id>o.dream_cursor_id))"""),
            {"owner": owner},
        ).scalar_one()
        page_size = session.execute(text("PRAGMA page_size")).scalar_one()
        allocated = session.execute(text("PRAGMA page_count")).scalar_one() * page_size
        reusable = (
            session.execute(text("PRAGMA freelist_count")).scalar_one() * page_size
        )
        cache = runtime.service._caches.get(owner)
        cached = len(cache.memories) if cache else 0
        filename = runtime.service.store.db.engine.url.database
    path = Path(filename) if filename and filename != ":memory:" else None

    def size(file):
        try:
            return file.stat().st_size if file else 0
        except FileNotFoundError:
            return 0

    return MemoryMeasurements(
        owner_id=owner,
        memories_by_status=memories,
        jobs_by_status=jobs,
        oldest_ready_wait_seconds=now - oldest if oldest is not None else None,
        cached_records=cached,
        records_after_dream_cursor=remaining,
        database_file_bytes=size(path),
        wal_file_bytes=size(Path(str(path) + "-wal") if path else None),
        allocated_database_bytes=allocated,
        reusable_database_bytes=reusable,
    )
