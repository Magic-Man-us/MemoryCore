"""Per-user assistant memory pattern example.

This example demonstrates how to create separate memory systems
for each user/assistant pair, enabling isolated memory spaces.
"""

import asyncio
from typing import Dict

from memory_core_py import (
    LongTermStore,
    MemorySystem,
    RedisWorkingMemory,
    RustMemoryIndex,
    ShortTermStore,
)


class PerUserMemoryManager:
    """Manage separate memory systems for each user."""

    def __init__(
        self,
        ltm_host: str = "localhost",
        ltm_user: str = "memory_user",
        ltm_password: str = "secure_password",
        ltm_database: str = "memory_db",
        redis_url: str = "redis://localhost:6379/0",
    ):
        self._ltm_config = {
            "host": ltm_host,
            "user": ltm_user,
            "password": ltm_password,
            "database": ltm_database,
        }
        self._redis_url = redis_url
        self._user_systems: dict[str, MemorySystem] = {}

    def get_memory_system(self, user_id: str) -> MemorySystem:
        """Get or create a memory system for a specific user."""
        if user_id not in self._user_systems:
            # Create dedicated components for this user
            ltm_store = LongTermStore(**self._ltm_config)
            stm_store = ShortTermStore(**self._ltm_config)
            working_mem = RedisWorkingMemory(url=self._redis_url)
            memory_index = RustMemoryIndex()

            # Create isolated memory system
            memory_system = MemorySystem(
                memory_index=memory_index,
                working_mem=working_mem,
                ltm_store=ltm_store,
                stm_store=stm_store,
                stm_ttl_seconds=900,
            )
            self._user_systems[user_id] = memory_system

        return self._user_systems[user_id]


async def main():
    """Demonstrate per-user memory isolation."""
    manager = PerUserMemoryManager()

    # User 1 stores a memory
    user1_system = manager.get_memory_system("user_1")
    embedding1 = [0.1, 0.2, 0.3] * 128

    trace1 = await user1_system.remember(
        user_id="user_1",
        summary="Likes Python programming",
        importance=0.7,
        tags=["skill", "preference"],
        embedding=embedding1,
    )
    print(f"User 1 stored: {trace1.trace_uid}")

    # User 2 stores a different memory
    user2_system = manager.get_memory_system("user_2")
    embedding2 = [0.4, 0.5, 0.6] * 128

    trace2 = await user2_system.remember(
        user_id="user_2",
        summary="Prefers Rust for systems programming",
        importance=0.8,
        tags=["skill", "preference"],
        embedding=embedding2,
    )
    print(f"User 2 stored: {trace2.trace_uid}")

    # Each user has isolated memory
    print("\\nMemory systems are isolated per user")
    print(f"User 1 system: {id(user1_system)}")
    print(f"User 2 system: {id(user2_system)}")


if __name__ == "__main__":
    asyncio.run(main())
