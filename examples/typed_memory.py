"""Offline example using explicit toy vectors (not a semantic embedding model)."""

import asyncio

from memory_core import Database, EmbeddingSpec, MemoryService
from memory_core.memory.models import (
    ContainsAny,
    DescribeSchemaRequest,
    DreamPlanRequest,
    PreferenceMemory,
    RecallRequest,
    RememberRequest,
    TaskContext,
)


async def main():
    memory = MemoryService(
        Database(":memory:"),
        embedding_spec=EmbeddingSpec(
            model_id="example-only", revision="1", dimensions=3
        ),
    )
    stored = await memory.call(
        RememberRequest(
            memory=PreferenceMemory(
                summary="Use uv for Python dependency management.",
                preference_key="python.package_manager",
                value="uv",
                applicability={"languages": ["python"], "activities": ["write_code"]},
                metadata={"languages": ["python"], "tools": ["uv"]},
                source_refs=["example:user-instruction"],
            ),
            embedding=[1.0, 0.0, 0.0],
        ),
        owner_id="alice",
    )
    context = TaskContext(
        task_id="edit-api",
        project_id="my-project",
        attributes={"languages": ["python"], "activities": ["write_code"]},
    )

    async def coding_action(prepared):
        await memory.validate_context(prepared.receipt, context, owner_id="alice")
        print(
            "Before coding:",
            [(m.preference_key, m.value) for m in prepared.preferences.effective],
        )
        # Pass prepared.preferences and prepared.supporting to your agent here.

    await memory.run_with_context(context, coding_action, owner_id="alice")
    result = await memory.call(
        RecallRequest(
            query="package manager",
            embedding=[1.0, 0.0, 0.0],
            filter=ContainsAny(field="tools", values=["uv"]),
        ),
        owner_id="alice",
    )
    assert result.hits[0].memory.id == stored.memory.id
    schema = await memory.call(DescribeSchemaRequest(), owner_id="alice")
    print("Registered fields:", [f.name for f in schema.fields])
    dream = await memory.call(DreamPlanRequest(), owner_id="alice")
    print("Dream inspected:", len(dream.plan.groups), "group(s)")


if __name__ == "__main__":
    asyncio.run(main())
