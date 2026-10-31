"""Real MCP protocol acceptance, including the stdio entry point."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from jsonschema import Draft202012Validator
from mcp import Client, StdioServerParameters

from memory_core import EmbeddingSpec, MemoryCore
from memory_core.mcpapp import create_server
from memory_core.runtime import MaintenancePolicy
from memory_core.runtime.catalog import SPECS, schemas
from memory_core.runtime.models import RESPONSE_ADAPTER


class Embedder:
    async def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


@pytest.fixture
def memory(tmp_path):
    runtime = MemoryCore.local(
        tmp_path / "mcp.db",
        embedding_spec=EmbeddingSpec(model_id="test", revision="1", dimensions=2),
        embedder=Embedder(),
        maintenance=MaintenancePolicy(poll_seconds=0.01),
    )
    yield runtime
    runtime.service.store.db.dispose()


async def test_discovery_schemas_and_server_lifespan(memory):
    assert memory.worker._runner is None
    async with Client(create_server(memory, owner_id="alice")) as client:
        tools = (await client.list_tools()).tools
        assert {tool.name for tool in tools} == SPECS.keys()
        for tool in tools:
            Draft202012Validator.check_schema(tool.input_schema)
            Draft202012Validator.check_schema(tool.output_schema)
        assert memory.worker._runner is not None
        lookup = {t.name: t for t in tools}
        assert lookup["recall"].annotations.read_only_hint
        assert lookup["remember"].annotations.destructive_hint
        assert lookup["record_finding"].annotations.idempotent_hint
        result = await client.call_tool("describe_schema", {})
        assert not result.is_error
        assert result.structured_content["operation"] == "describe_schema"
        RESPONSE_ADAPTER.validate_python(result.structured_content)
    assert memory.worker._runner is None


async def test_protocol_task_capture_and_completion(memory):
    async with Client(create_server(memory, owner_id="alice")) as client:
        begin = await client.call_tool(
            "begin_task",
            {
                "context": {
                    "task_id": "task",
                    "project_id": "project",
                    "attributes": {"languages": ["python"]},
                },
                "objective": "Implement retry policy",
            },
        )
        assert not begin.is_error
        prepared = await client.call_tool(
            "prepare_task", {"task_id": "task", "guard": True}
        )
        assert prepared.structured_content["receipt"]
        completed = await client.call_tool(
            "complete_task",
            {
                "task_id": "task",
                "outcome": {"status": "succeeded", "summary": "Completed example"},
            },
        )
        assert not completed.is_error
        async with asyncio.timeout(5):
            while True:
                state = await client.call_tool("maintenance_status", {})
                if state.structured_content["jobs"][0]["status"] == "done":
                    break
                await asyncio.sleep(0.01)
        result = await client.call_tool(
            "recall", {"query": "retry", "project_id": "project"}
        )
        assert len(result.structured_content["hits"]) == 1


async def test_unknown_gap_blocks_protocol_action(memory):
    async with Client(create_server(memory, owner_id="alice")) as client:
        await client.call_tool(
            "begin_task", {"context": {"task_id": "task"}, "objective": "Code"}
        )
        gap = await client.call_tool(
            "require_knowledge",
            {"task_id": "task", "gap": {"id": "g", "question": "Unknown?"}},
        )
        assert gap.structured_content["result"]["status"] == "unresolved"
        blocked = await client.call_tool(
            "prepare_task", {"task_id": "task", "guard": True}
        )
        assert blocked.is_error
        assert blocked.structured_content["code"] == "ContextRequiredError"


async def test_allowlist_enforced_and_cannot_forge_owner_or_operation(memory):
    async with Client(
        create_server(
            memory,
            owner_id="alice",
            allowed_operations={"describe_schema", "maintenance_status"},
        )
    ) as client:
        assert {t.name for t in (await client.list_tools()).tools} == {
            "describe_schema",
            "maintenance_status",
        }
        denied = await client.call_tool(
            "begin_task", {"context": {"task_id": "x"}, "objective": "x"}
        )
        assert (
            denied.is_error and denied.structured_content["code"] == "UnknownOperation"
        )
        denied = await client.call_tool("describe_schema", {"operation": "remember"})
        assert denied.is_error
        denied = await client.call_tool("maintenance_status", {"owner_id": "bob"})
        assert denied.is_error
        assert denied.structured_content["code"] == "ValidationError"
        assert "bob" not in str(denied.structured_content)


def test_catalog_is_pruned_but_preserves_recursive_filters():
    definitions = schemas()
    assert "$defs" not in definitions["describe_schema"][0]
    assert "All" in definitions["recall"][0]["$defs"]
    assert "CompleteTaskRequest" not in definitions["recall"][0].get("$defs", {})
    Draft202012Validator(definitions["recall"][0]).validate(
        {
            "query": "q",
            "filter": {
                "op": "all",
                "conditions": [
                    {
                        "op": "any",
                        "conditions": [
                            {"op": "eq", "field": "languages", "value": "python"}
                        ],
                    }
                ],
            },
        }
    )


async def test_stdio_entrypoint_with_local_factory(tmp_path):
    factory = tmp_path / "memory_fixture.py"
    factory.write_text(
        """from memory_core import EmbeddingSpec, MemoryCore
from memory_core.runtime import MaintenancePolicy
class Embedder:
    async def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]
def create_memory():
    return MemoryCore.local(PATH, embedding_spec=EmbeddingSpec(model_id="test", revision="1", dimensions=2),
        embedder=Embedder(), maintenance=MaintenancePolicy(background=False))
""".replace("PATH", repr(str(tmp_path / "stdio.db")))
    )
    parameters = StdioServerParameters(
        command=sys.executable,
        args=[
            "-m",
            "memory_core.mcpapp",
            "--factory",
            "memory_fixture:create_memory",
            "--owner",
            "local-user",
            "--allow",
            "describe_schema",
        ],
        env={
            "PYTHONPATH": os.pathsep.join(
                [str(tmp_path), str(Path(__file__).resolve().parents[1])]
            )
        },
    )
    async with Client(parameters) as client:
        assert [t.name for t in (await client.list_tools()).tools] == [
            "describe_schema"
        ]
        result = await client.call_tool("describe_schema", {})
        assert not result.is_error
