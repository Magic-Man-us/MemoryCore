"""Storage layer tests against a real (temp) SQLite database."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from memory_core import MemoryTrace
from memory_core.storage import LongTermStore, ShortTermStore, SqlWorkingMemory


def _trace(uid: str, user: str = "alice", **kw) -> MemoryTrace:
    defaults = dict(
        trace_uid=uid,
        user_id=user,
        content="full content",
        summary=f"summary {uid}",
        importance=0.7,
        created_at=datetime.now(UTC),
        tags={"t1", "t2"},
    )
    defaults.update(kw)
    return MemoryTrace(**defaults)


class TestLongTermStore:
    def test_upsert_fetch_roundtrip(self, db):
        store = LongTermStore(db)
        store.upsert_trace(_trace("u1"), embedding=[0.1, 0.2, 0.3])

        rows = store.fetch_traces_for_user("alice")
        assert len(rows) == 1
        trace, embedding = rows[0]
        assert trace.trace_uid == "u1"
        assert trace.content == "full content"
        assert trace.tags == {"t1", "t2"}
        assert embedding == [0.1, 0.2, 0.3]

    def test_upsert_is_upsert(self, db):
        store = LongTermStore(db)
        store.upsert_trace(_trace("u1"), embedding=[0.1])
        store.upsert_trace(_trace("u1", summary="summary updated"), embedding=[0.9])
        rows = store.fetch_traces_for_user("alice")
        assert len(rows) == 1
        assert rows[0][0].summary == "summary updated"
        assert rows[0][1] == [0.9]

    def test_fetch_isolates_users(self, db):
        store = LongTermStore(db)
        store.upsert_trace(_trace("a1", user="alice"), embedding=[0.1])
        store.upsert_trace(_trace("b1", user="bob"), embedding=[0.2])
        assert [t.trace_uid for t, _ in store.fetch_traces_for_user("alice")] == ["a1"]

    def test_delete(self, db):
        store = LongTermStore(db)
        store.upsert_trace(_trace("u1"), embedding=[0.1])
        assert store.delete_trace("u1") is True
        assert store.delete_trace("u1") is False
        assert store.fetch_traces_for_user("alice") == []

    def test_no_embedding_roundtrip(self, db):
        store = LongTermStore(db)
        store.upsert_trace(_trace("u2"), embedding=None)
        [(trace, embedding)] = store.fetch_traces_for_user("alice")
        assert embedding is None

    def test_extra_roundtrip(self, db):
        store = LongTermStore(db)
        store.upsert_trace(
            _trace("u3", extra={"source": "slack", "session": "s-1"}), embedding=[0.1]
        )
        [(trace, _)] = store.fetch_traces_for_user("alice")
        assert trace.extra == {"source": "slack", "session": "s-1"}

    def test_extra_with_non_json_native_values(self, db):
        """A datetime in extra must serialize like STM does (mode='json'), not TypeError."""
        stamp = datetime(2026, 7, 2, 3, 4, 5, tzinfo=UTC)
        store = LongTermStore(db)
        store.upsert_trace(_trace("u4", extra={"at": stamp}), embedding=[0.1])
        [(trace, _)] = store.fetch_traces_for_user("alice")
        assert trace.extra == {"at": "2026-07-02T03:04:05Z"}

    def test_schema_upgrade_adds_extra_column(self, tmp_path):
        """A database created before the extra column existed keeps working."""
        from sqlalchemy import create_engine, text

        from memory_core.storage import Database, DatabaseSettings

        url = f"sqlite:///{tmp_path}/old.db"
        old_engine = create_engine(url)
        with old_engine.begin() as connection:
            connection.execute(
                text(
                    "CREATE TABLE ltm_traces ("
                    "trace_uid VARCHAR PRIMARY KEY, user_id VARCHAR NOT NULL, "
                    "content TEXT, summary TEXT NOT NULL, importance FLOAT NOT NULL, "
                    "created_at DATETIME NOT NULL, access_count INTEGER NOT NULL, "
                    "tags TEXT, embedding BLOB)"
                )
            )
        old_engine.dispose()

        upgraded = Database(settings=DatabaseSettings(url=url))
        upgraded.create_schema()  # must ALTER in the missing extra column
        store = LongTermStore(upgraded)
        store.upsert_trace(_trace("u5", extra={"source": "cli"}), embedding=[0.2])
        [(trace, _)] = store.fetch_traces_for_user("alice")
        assert trace.extra == {"source": "cli"}


class TestShortTermStore:
    def test_insert_and_fetch_non_expired(self, db):
        store = ShortTermStore(db)
        now = datetime.now(UTC)
        store.insert_trace(_trace("s1"), expires_at=now + timedelta(seconds=60))
        store.insert_trace(_trace("s2"), expires_at=now - timedelta(seconds=60))

        got = store.fetch_recent_for_user("alice")
        assert [t.trace_uid for t in got] == ["s1"]

    def test_purge_expired(self, db):
        store = ShortTermStore(db)
        now = datetime.now(UTC)
        store.insert_trace(_trace("s1"), expires_at=now - timedelta(seconds=1))
        assert store.purge_expired() == 1
        assert store.purge_expired() == 0


class TestSqlWorkingMemory:
    def test_add_get_ordering(self, db):
        wm = SqlWorkingMemory(db, ttl_seconds=60)
        for i in range(5):
            wm.add_event("alice", {"n": i})
        events = wm.get_recent("alice", limit=3)
        assert [e["n"] for e in events] == [2, 3, 4]  # oldest-first of the last 3
        assert all("ts" in e for e in events)

    def test_ttl_expiry(self, db):
        wm = SqlWorkingMemory(db, ttl_seconds=1)
        wm.add_event("alice", {"n": 1})
        # Force-expire by rewinding the stored expiry
        from sqlalchemy import update

        from memory_core.storage.database import WorkingMemoryEventORM

        with db.session() as session:
            session.execute(
                update(WorkingMemoryEventORM).values(
                    expires_at=datetime.now(UTC) - timedelta(seconds=5)
                )
            )
            session.commit()
        assert wm.get_recent("alice") == []

    def test_per_user_isolation_and_cap(self, db):
        wm = SqlWorkingMemory(db, ttl_seconds=60, max_events=3)
        for i in range(6):
            wm.add_event("alice", {"n": i})
        wm.add_event("bob", {"n": 100})
        alice = wm.get_recent("alice", limit=50)
        assert len(alice) == 3
        assert [e["n"] for e in alice] == [3, 4, 5]
        assert [e["n"] for e in wm.get_recent("bob")] == [100]

    def test_clear(self, db):
        wm = SqlWorkingMemory(db, ttl_seconds=60)
        wm.add_event("alice", {"n": 1})
        wm.clear("alice")
        assert wm.get_recent("alice") == []

    def test_server_ts_is_authoritative(self, db):
        wm = SqlWorkingMemory(db, ttl_seconds=60)
        wm.add_event("alice", {"n": 1, "ts": "1999-01-01T00:00:00+00:00"})
        [event] = wm.get_recent("alice")
        assert event["ts"] != "1999-01-01T00:00:00+00:00"
        assert event["n"] == 1

    def test_non_serializable_payload_raises_value_error(self, db):
        import pytest

        wm = SqlWorkingMemory(db, ttl_seconds=60)
        with pytest.raises(ValueError, match="JSON-serializable"):
            wm.add_event("alice", {"bad": object()})
        assert wm.get_recent("alice") == []  # nothing was written
