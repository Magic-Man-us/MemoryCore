"""SMK assistant memory: store what the assistant learned, recall it by structure.

The AssistantMemoryIndex packs topic/kind/tools/levels into a 64-bit Structured
Memory Key and prunes candidates with bitfield checks before cosine similarity.
Assistant traces persist to the same SQLite file and are reloaded with
``hydrate_assistant()`` after a restart.

Run:
    python examples/smk_assistant.py
"""

import asyncio
import logging

from memory_core import (
    AssistantMemoryTrace,
    MemoryKind,
    ToolFlag,
    TopicBucket,
    build_memory_system,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("smk_assistant")


def build():
    return build_memory_system(
        sqlite_path="example_memory.db",
        overrides={"enable_assistant_index": True, "assistant_index_dim": 384},
    )


async def main() -> None:
    system = build()

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
        rationale="Reusable workflow for bridging Rust and Python with PyO3.",
    )

    embedding = [0.1, 0.2, 0.3] * 128
    await system.remember_assistant(trace=trace, embedding=embedding)
    logger.info("Stored assistant memory %s (smk=%d)", trace.trace_uid, trace.smk_raw)

    # Simulate a restart: a fresh index, reloaded from the database.
    system = build()
    logger.info("Hydrated %d assistant memories", await system.hydrate_assistant())

    results = await system.recall_assistant(
        query_embedding=embedding,
        k=5,
        topic=TopicBucket.RUST_PYTHON_TOOLCHAIN,
        required_tools={ToolFlag.RS, ToolFlag.PY},
        allowed_kinds=[MemoryKind.PATTERN, MemoryKind.ANTI_PATTERN],
    )
    logger.info("Found %d assistant memories", len(results))
    for hit in results:
        logger.info("  %s score=%.3f smk=%d", hit.trace_uid, hit.score, hit.smk_raw)


if __name__ == "__main__":
    asyncio.run(main())
