"""Basic usage example for the memory_core package.

This example demonstrates how to set up and use the MemorySystem
for basic user memory operations.
"""

import asyncio

from memory_core_py import (
    LongTermStore,
    MemorySystem,
    RedisWorkingMemory,
    RustMemoryIndex,
)


async def main():
    """Demonstrate basic memory system usage."""
    # Initialize storage components
    ltm_store = LongTermStore(
        host="localhost",
        user="memory_user",
        password="secure_password",
        database="memory_db",
    )

    working_mem = RedisWorkingMemory(
        url="redis://localhost:6379/0",
        ttl_seconds=900,
    )

    # Initialize vector index
    memory_index = RustMemoryIndex()

    # Create the memory system
    memory_system = MemorySystem(
        memory_index=memory_index,
        working_mem=working_mem,
        ltm_store=ltm_store,
    )

    # Example: Store a memory
    user_id = "user_123"
    embedding = [0.1, 0.2, 0.3] * 128  # 384-dim example

    trace = await memory_system.remember(
        user_id=user_id,
        summary="User prefers dark mode UI",
        importance=0.8,
        tags=["preference", "ui"],
        embedding=embedding,
        also_working_mem=True,
    )
    print(f"Stored memory: {trace.trace_uid}")

    # Example: Recall memories
    query_embedding = [0.1, 0.2, 0.3] * 128
    results = await memory_system.recall(
        user_id=user_id,
        query_text="UI preferences",
        tags=["preference"],
        limit=10,
        query_embedding=query_embedding,
        include_working_mem=True,
    )

    print(f"Found {len(results['ltm_candidates'])} LTM candidates")
    print(f"Found {len(results['wm_events'])} working memory events")


if __name__ == "__main__":
    asyncio.run(main())
