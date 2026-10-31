# Legacy user memory API

The pre-0.4 user API remains available for existing integrations. It uses separate LTM,
STM, and working-memory tables and does not perform typed preference resolution or dreams.
Use the typed service for new integrations.

```python
import asyncio
from memory_core import build_memory_system

async def main():
    system = build_memory_system(sqlite_path=":memory:")
    await system.remember(user_id="alice", summary="Prefers Python", importance=0.8,
                          tags=["preferences"], embedding=[1.0, 0.0])
    result = await system.recall(user_id="alice", query_text="languages", limit=5,
                                 query_embedding=[1.0, 0.0])
    print(result.ltm_candidates)

asyncio.run(main())
```

For file-backed persistence, call `await system.hydrate(user_id)` after restart. An `Embedder`
can supply vectors automatically. `forget(trace_uid)` removes the legacy trace and linked
working-memory events. `purge_expired()` cleans STM and working-memory expiry.

Legacy scoring combines cosine similarity, importance, optional exponential recency decay,
and a bounded access-frequency term. Defaults are semantic 0.7, importance 0.3, recency/access 0.
Configure these via `overrides={"scoring": {...}}`. The typed service uses its own
`RetrievalPolicy` and explicit preference resolution instead.

SQLite paths support `~`, automatically create parent directories, and use WAL with a busy
timeout. `sqlite_path` takes precedence over the database URL. PostgreSQL remains optional
for this legacy API via `.[postgres]` and `MEMORY_DB_URL`; the new typed service requires SQLite.

Working memory defaults to a 3,600-second TTL and 500-event cap. STM is disabled by default,
with a 900-second TTL when enabled using `overrides={"enable_stm": True}`. Settings use the
`MEMORY_CORE_` prefix and nested `__` separator; database settings use `MEMORY_DB_`.
