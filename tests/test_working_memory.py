"""RedisWorkingMemory against an in-memory fake Redis client."""

from __future__ import annotations

from conftest import FakeRedisClient
from memory_core_py.storage.settings import RedisWorkingMemorySettings
from memory_core_py.storage.working_mem import RedisWorkingMemory


def make_wm(ttl: int = 60) -> tuple[RedisWorkingMemory, FakeRedisClient]:
    wm = RedisWorkingMemory(settings=RedisWorkingMemorySettings(ttl_seconds=ttl))
    fake = FakeRedisClient()
    wm.client = fake  # constructor never connects; swap in the fake
    return wm, fake


async def test_add_event_stamps_timestamp_and_ttl():
    wm, fake = make_wm(ttl=42)
    await wm.add_event(user_id="u-1", payload={"kind": "note", "value": 7})

    events = await wm.get_recent(user_id="u-1")
    assert len(events) == 1
    assert events[0]["kind"] == "note"
    assert events[0]["value"] == 7
    assert "ts" in events[0]
    assert fake.ttls["wm:u-1"] == 42


async def test_get_recent_returns_latest_events_up_to_limit():
    wm, _ = make_wm()
    for i in range(5):
        await wm.add_event(user_id="u-1", payload={"i": i})

    events = await wm.get_recent(user_id="u-1", limit=2)
    assert [e["i"] for e in events] == [3, 4]  # newest at the end


async def test_users_are_partitioned():
    wm, _ = make_wm()
    await wm.add_event(user_id="u-1", payload={"who": "one"})
    await wm.add_event(user_id="u-2", payload={"who": "two"})

    assert [e["who"] for e in await wm.get_recent(user_id="u-1")] == ["one"]
    assert [e["who"] for e in await wm.get_recent(user_id="u-2")] == ["two"]


async def test_empty_user_has_no_events():
    wm, _ = make_wm()
    assert await wm.get_recent(user_id="ghost") == []
