"""Local SQLite on request: the sqlite_path shortcut, in-memory databases, and paths."""

from __future__ import annotations

import asyncio

import pytest

from memory_core import Database, DatabaseSettings, build_memory_system, sqlite_url
from memory_core.storage.settings import DEFAULT_DB_URL

EMB = [0.1, 0.2, 0.3]


async def _remember(system, summary: str, user: str = "alice"):
    return await system.remember(
        user_id=user, summary=summary, importance=0.5, tags=[], embedding=EMB
    )


def test_sqlite_url_helper(tmp_path):
    assert sqlite_url(":memory:") == "sqlite:///:memory:"
    assert sqlite_url(tmp_path / "m.db") == f"sqlite:///{tmp_path.as_posix()}/m.db"
    assert sqlite_url("rel/m.db") == "sqlite:///rel/m.db"


async def test_sqlite_path_shortcut_creates_parent_dirs(tmp_path):
    target = tmp_path / "nested" / "deeper" / "memory.db"
    system = build_memory_system(sqlite_path=target)
    await _remember(system, "stored locally")
    assert target.exists()


async def test_sqlite_path_expands_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    system = build_memory_system(sqlite_path="~/.assistant/memory.db")
    await _remember(system, "under home")
    assert (tmp_path / ".assistant" / "memory.db").exists()


def test_sqlite_path_wins_over_url(tmp_path):
    db = Database(
        settings=DatabaseSettings(url="postgresql+psycopg://x@nowhere/db", sqlite_path=":memory:")
    )
    assert db.is_memory


def test_explicit_url_beats_env_sqlite_path(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORY_DB_SQLITE_PATH", str(tmp_path / "env.db"))
    url = f"sqlite:///{tmp_path}/explicit.db"
    system = build_memory_system(overrides={"db": {"url": url}})
    assert system.ltm_store._db.settings.resolved_url == url
    assert DatabaseSettings.from_overrides({"url": url}).resolved_url == url


def test_env_sqlite_path(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORY_DB_SQLITE_PATH", str(tmp_path / "env.db"))
    assert DatabaseSettings().resolved_url == f"sqlite:///{tmp_path.as_posix()}/env.db"


def test_default_is_still_a_local_file(monkeypatch):
    for var in ("MEMORY_DB_URL", "MEMORY_DB_SQLITE_PATH"):
        monkeypatch.delenv(var, raising=False)
    assert DatabaseSettings().resolved_url == DEFAULT_DB_URL


@pytest.mark.parametrize("spec", [{"sqlite_path": ":memory:"}, {"url": "sqlite://"}])
async def test_in_memory_database_survives_worker_threads(spec):
    """Store calls run in worker threads; without a shared connection each thread
    would see its own empty database ("no such table")."""
    system = build_memory_system(overrides={"db": spec})
    assert system.ltm_store._db.is_memory
    traces = await asyncio.gather(*(_remember(system, f"fact {i}") for i in range(20)))
    rows = system.ltm_store.fetch_traces_for_user("alice")
    assert {t.trace_uid for t, _ in rows} == {t.trace_uid for t in traces}
    result = await system.recall(user_id="alice", query_text="fact", limit=50,
                                 query_embedding=EMB)
    assert len(result.ltm_candidates) == 20
    assert len(result.wm_events) == 20


async def test_in_memory_databases_are_isolated():
    a = build_memory_system(sqlite_path=":memory:")
    b = build_memory_system(sqlite_path=":memory:")
    await _remember(a, "only in a")
    assert b.ltm_store.fetch_traces_for_user("alice") == []


async def test_file_database_concurrent_writes(tmp_path):
    system = build_memory_system(sqlite_path=tmp_path / "busy.db")
    await asyncio.gather(*(_remember(system, f"fact {i}") for i in range(20)))
    assert len(system.ltm_store.fetch_traces_for_user("alice")) == 20
