"""Per-user assistant memory pattern: isolated in-RAM indexes, one shared database.

Each user gets their own MemorySystem (and so their own RustMemoryIndex), while all
of them persist to the same SQLite file — isolation at query time, one file on disk.

Run:
    python examples/per_user_assistant.py
"""

import asyncio

from memory_core import (
    Database,
    LongTermStore,
    MemorySystem,
    RustMemoryIndex,
    SqlWorkingMemory,
)


class PerUserMemoryManager:
    """Manage separate memory systems for each user over one shared database."""

    def __init__(self, db_url: str = "sqlite:///example_memory.db") -> None:
        self._db = Database(url=db_url)
        self._db.create_schema()
        self._user_systems: dict[str, MemorySystem] = {}

    def get_memory_system(self, user_id: str) -> MemorySystem:
        if user_id not in self._user_systems:
            self._user_systems[user_id] = MemorySystem(
                memory_index=RustMemoryIndex(),  # per-user index: natural isolation
                working_mem=SqlWorkingMemory(self._db),
                ltm_store=LongTermStore(self._db),
            )
        return self._user_systems[user_id]


async def main() -> None:
    manager = PerUserMemoryManager()

    user1 = manager.get_memory_system("user_1")
    trace1 = await user1.remember(
        user_id="user_1",
        summary="Likes Python programming",
        importance=0.7,
        tags=["skill", "preference"],
        embedding=[0.1, 0.2, 0.3] * 128,
    )
    print(f"User 1 stored: {trace1.trace_uid}")

    user2 = manager.get_memory_system("user_2")
    trace2 = await user2.remember(
        user_id="user_2",
        summary="Prefers Rust for systems programming",
        importance=0.8,
        tags=["skill", "preference"],
        embedding=[0.4, 0.5, 0.6] * 128,
    )
    print(f"User 2 stored: {trace2.trace_uid}")

    print("\nMemory systems are isolated per user, persisted in one database")
    print(f"User 1 system: {id(user1)}")
    print(f"User 2 system: {id(user2)}")


if __name__ == "__main__":
    asyncio.run(main())
