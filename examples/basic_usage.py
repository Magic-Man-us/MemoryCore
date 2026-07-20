"""Basic usage: a SQLite-backed memory system, no services required.

Run:
    python examples/basic_usage.py
"""

import asyncio

from memory_core import build_memory_system


async def main() -> None:
    # One SQLite file holds LTM, STM, and working memory; tables are created
    # automatically. Point MEMORY_DB_URL (or overrides) at PostgreSQL later if
    # you outgrow a file.
    system = build_memory_system(overrides={"db": {"url": "sqlite:///example_memory.db"}})

    user_id = "user_123"
    embedding = [0.1, 0.2, 0.3] * 128  # 384-dim example; supply your model's vectors,
    # or construct the system with an `Embedder` and skip embeddings entirely.

    trace = await system.remember(
        user_id=user_id,
        summary="User prefers dark mode UI",
        content="Said 'always use dark mode' while configuring the editor",
        importance=0.8,
        tags=["preference", "ui"],
        embedding=embedding,
    )
    print(f"Stored memory: {trace.trace_uid}")

    result = await system.recall(
        user_id=user_id,
        query_text="UI preferences",
        limit=10,
        query_embedding=embedding,
    )
    print(f"Found {len(result.ltm_candidates)} LTM candidates")
    print(f"Found {len(result.wm_events)} working memory events")

    # After a restart, repopulate the in-RAM index from the database:
    restored = await system.hydrate(user_id)
    print(f"Hydrated {restored} traces from LTM")


if __name__ == "__main__":
    asyncio.run(main())
