"""SMK assistant memory example.

This example demonstrates how to use the AssistantMemoryIndex with SMK
(Semantic Memory Kernel) features for assistant-focused memory retrieval.
"""

import asyncio

from logfire import log as logger
from memory_core_py import (
    AssistantMemoryIndex,
    LongTermStore,
    MemoryKind,
    MemorySystem,
    RedisWorkingMemory,
    RustMemoryIndex,
    ToolFlag,
    TopicBucket,
)
from memory_core_py.core.models import AssistantMemoryTrace


async def main():
    """Demonstrate SMK assistant index usage."""
    # Initialize base memory system
    ltm_store = LongTermStore(
        host="localhost",
        user="memory_user",
        password="secure_password",
        database="memory_db",
    )
    working_mem = RedisWorkingMemory(url="redis://localhost:6379/0")
    memory_index = RustMemoryIndex()

    # Initialize SMK assistant index (384-dimensional embedding)
    assistant_index = AssistantMemoryIndex(dim=384)

    # Create memory system with assistant index
    memory_system = MemorySystem(
        memory_index=memory_index,
        working_mem=working_mem,
        ltm_store=ltm_store,
        assistant_index=assistant_index,
    )

    # Create an assistant memory trace
    trace = AssistantMemoryTrace(
        trace_uid="pattern_001",
        context_activity="Integrating Rust with Python using PyO3 and maturin.",
        context_complexity=0.6,
        context_topic=TopicBucket.RUST_PYTHON_TOOLCHAIN,
        kind=MemoryKind.PATTERN,
        tools={ToolFlag.RS, ToolFlag.PY, ToolFlag.MATURIN},
        before_state_confusion=0.3,
        after_state_confidence=0.9,
        generality=0.7,
        importance=0.8,
        summary="Pattern for using Rust with Python via PyO3 and maturin.",
        rationale="Captures a reusable workflow for bridging Rust and Python with PyO3, improving future assistant responses on this topic.",
    )

    # Store assistant memory
    embedding = [0.1, 0.2, 0.3] * 128
    await memory_system.remember_assistant(
        trace=trace,
        embedding=embedding,
    )

    logger(
        "info",
        "Stored assistant memory {trace_uid}",
        {"trace_uid": trace.trace_uid},
    )

    # Query assistant memories with filters
    query_embedding = [0.1, 0.2, 0.3] * 128
    results = await memory_system.recall_assistant(
        query_embedding=query_embedding,
        k=5,
        topic=TopicBucket.RUST_PYTHON_TOOLCHAIN,
        required_tools={ToolFlag.RS, ToolFlag.PY},
        allowed_kinds=[MemoryKind.PATTERN, MemoryKind.ANTI_PATTERN],
    )

    logger(
        "info",
        "Found {count} assistant memories",
        {"count": len(results)},
    )
    for hit in results:
        logger(
            "info",
            "Memory hit trace_uid={trace_uid} score={score}",
            {"trace_uid": hit.trace_uid, "score": hit.score},
        )

if __name__ == "__main__":
    asyncio.run(main())
