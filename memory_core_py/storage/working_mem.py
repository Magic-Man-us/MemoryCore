from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from redis.asyncio import Redis

from .settings import RedisWorkingMemorySettings


class RedisWorkingMemory:
    """
    Simple working memory store in Redis.

    - Per-user key: wm:{user_id}
    - Value: list of JSON events
    - TTL enforces recency (like human working memory decay).
    """

    def __init__(
        self,
        url: str | None = None,
        ttl_seconds: int | None = None,
        settings: RedisWorkingMemorySettings | None = None,
    ) -> None:
        base_settings = settings or RedisWorkingMemorySettings()
        overrides = {"url": url, "ttl_seconds": ttl_seconds}
        resolved_settings = base_settings.apply_overrides(overrides)

        # Use the asyncio Redis client so that methods are awaitable.
        self.client: Redis = Redis.from_url(
            resolved_settings.dsn,
            decode_responses=False,
        )
        self.ttl_seconds = resolved_settings.ttl_seconds

    def _key(self, user_id: str) -> str:
        return f"wm:{user_id}"

    async def add_event(self, user_id: str, payload: dict[str, Any]) -> None:
        key = self._key(user_id)
        doc = {
            "ts": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        await self.client.rpush(key, json.dumps(doc))
        await self.client.expire(key, self.ttl_seconds)

    async def get_recent(self, user_id: str, limit: int = 20) -> list[dict[str, Any]]:
        key = self._key(user_id)
        raw = await self.client.lrange(key, -limit, -1)
        out: list[dict[str, Any]] = []
        if isinstance(raw, list):
            for b in raw:
                loaded = json.loads(b)
                if isinstance(loaded, dict):
                    out.append(loaded)

        else:
            # unexpected type
            pass

        return out
