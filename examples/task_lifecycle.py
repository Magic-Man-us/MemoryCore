"""Run with: uv run python examples/task_lifecycle.py

Deterministic local providers demonstrate wiring, not semantic model quality.
Replace these callbacks with your own permitted research and reasoning providers.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from memory_core import EmbeddingSpec, MemoryCore
from memory_core.memory.models import TaskContext
from memory_core.runtime import (
    Evidence,
    KnowledgeGap,
    MaintenancePolicy,
    ResolvedKnowledge,
    TaskOutcome,
    TaskRunResult,
    UnresolvedKnowledge,
)
from memory_core.runtime.models import MaintenanceStatusRequest


class DemoEmbedder:
    async def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


async def research(request):
    # A real provider searches only sources permitted by its host configuration.
    return [
        Evidence(
            id="project-policy",
            uri="repo:docs/retries.md",
            excerpt="Retry only idempotent requests.",
        )
    ]


async def evaluate(request):
    # The fixture answers one exact documented question. A real evaluator should
    # check all requirements and return unresolved when evidence is insufficient.
    for evidence in request.evidence:
        if evidence.excerpt == "Retry only idempotent requests.":
            return ResolvedKnowledge(answer=evidence.excerpt, evidence=[evidence])
    return UnresolvedKnowledge(reason="The project policy needs to be researched.")


async def main():
    with tempfile.TemporaryDirectory() as folder:
        async with MemoryCore.local(
            Path(folder) / "memory.db",
            embedding_spec=EmbeddingSpec(model_id="demo", revision="1", dimensions=2),
            embedder=DemoEmbedder(),
            researcher=research,
            knowledge_evaluator=evaluate,
            maintenance=MaintenancePolicy(background=False),
        ) as memory:
            async with memory.task(
                owner_id="demo-user",
                context=TaskContext(
                    task_id="retry-task",
                    project_id="demo-project",
                    attributes={"languages": ["python"], "activities": ["coding"]},
                ),
                objective="Implement the project retry policy",
            ) as task:
                knowledge = await task.require_knowledge(
                    KnowledgeGap(
                        id="retry-policy",
                        question="Which requests may be retried?",
                        requirements=["Find the explicit project policy"],
                    )
                )
                assert knowledge.result.status == "resolved"
                print("Resolved:", knowledge.result.answer)

                async def action(prepared):
                    assert prepared.receipt
                    # Replace with the SDK invocation. The agent receives prepared
                    # context and the resolved knowledge above before it acts.
                    return TaskRunResult(
                        outcome=TaskOutcome(
                            status="succeeded",
                            summary="Demonstrated task lifecycle",
                        )
                    )

                await task.run(action)

            # Manual draining is useful for tests/CLI. background=True runs this
            # automatically while the MemoryCore async context is alive.
            while await memory.worker.run_once():
                pass
            state = await memory.call(MaintenanceStatusRequest(), owner_id="demo-user")
            assert all(job.status == "done" for job in state.jobs)
            print("Durably applied jobs:", len(state.jobs))
        memory.service.store.db.dispose()


if __name__ == "__main__":
    asyncio.run(main())
