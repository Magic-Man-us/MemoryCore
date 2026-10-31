"""Read-only live Claude SDK example. Requires Claude authentication and a local model.

Run --help first. Unlike task_lifecycle.py, this example makes real model calls.
Tool evidence extraction below explicitly opts into bounded read-result snippets.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from claude_agent_sdk import ClaudeAgentOptions

from memory_core import MemoryCore
from memory_core.integrations.claude import ClaudeMemoryAgent, ClaudeMemoryPolicy
from memory_core.memory.embedding import LocalSentenceEmbedder
from memory_core.memory.models import TaskContext
from memory_core.runtime import Evidence


async def read_evidence(event):
    if event.get("hook_event_name") != "PostToolUse" or event.get("tool_name") not in {
        "Read",
        "Glob",
        "Grep",
    }:
        return []
    tool_id = event["tool_use_id"]
    excerpt = json.dumps(event.get("tool_response"), ensure_ascii=False)[:4096]
    return [
        Evidence(
            id=tool_id,
            uri=f"claude-tool:{event['tool_name']}:{tool_id}",
            excerpt=excerpt,
        )
    ]


async def main(args):
    embedder = LocalSentenceEmbedder(
        args.embedding_model,
        model_id=args.model_id,
        revision=args.model_revision,
        dimensions=args.dimensions,
    )
    memory = MemoryCore.local(
        args.database, embedding_spec=embedder.spec, embedder=embedder
    )
    try:
        async with (
            memory,
            ClaudeMemoryAgent(
                memory,
                policy=ClaudeMemoryPolicy(max_concurrent_runs=2),
                evidence_extractor=read_evidence,
            ) as agent,
        ):
            report = await agent.run(
                owner_id=args.owner,
                context=TaskContext(task_id=args.task_id, project_id=args.project_id),
                objective=args.objective,
                options=ClaudeAgentOptions(
                    cwd=args.project,
                    tools=["Read", "Glob", "Grep"],
                    allowed_tools=["Read", "Glob", "Grep"],
                    max_turns=12,
                    model=args.agent_model,
                ),
            )
            print(report.model_dump_json(indent=2))
    finally:
        memory.service.store.db.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-model", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--dimensions", type=int, required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--database", type=Path, default=Path("memory.db"))
    parser.add_argument("--agent-model")
    parser.add_argument(
        "--objective",
        default="Inspect the project and report its testing commands. Cite observed file evidence; if insufficient, report a failed outcome.",
    )
    asyncio.run(main(parser.parse_args()))
