"""Tool descriptions and minimal schemas derived from the typed dispatcher."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from memory_core.memory.models import Operation

from .models import REQUEST_ADAPTER, RESPONSE_ADAPTER, RuntimeOperation


@dataclass(frozen=True, slots=True)
class ToolSpec:
    description: str
    response: Operation | RuntimeOperation
    read_only: bool = False
    destructive: bool = False
    idempotent: bool = False
    open_world: bool = False


# These are protocol operations, not a taxonomy of memory topics or user activities.
SPECS = {
    Operation.DESCRIBE_SCHEMA: ToolSpec(
        "Discover memory fields and supported typed operations before using unfamiliar metadata.",
        Operation.DESCRIBE_SCHEMA,
        read_only=True,
        idempotent=True,
    ),
    Operation.LIST_METADATA_FIELDS: ToolSpec(
        "List registered metadata fields and aliases for this owner.",
        Operation.LIST_METADATA_FIELDS,
        read_only=True,
        idempotent=True,
    ),
    Operation.LIST_METADATA_VALUES: ToolSpec(
        "Page through known values for a registered field.",
        Operation.LIST_METADATA_VALUES,
        read_only=True,
        idempotent=True,
    ),
    Operation.REGISTER_METADATA_FIELD: ToolSpec(
        "Register a typed metadata field or alias before recording new attributes.",
        Operation.REGISTER_METADATA_FIELD,
    ),
    Operation.RECALL: ToolSpec(
        "Retrieve a bounded set of scoped memories relevant to a query; similarity is not proof of correctness.",
        Operation.RECALL,
        read_only=True,
        idempotent=True,
    ),
    Operation.RESOLVE_PREFS: ToolSpec(
        "Resolve all applicable preferences, including conflicts, independently of semantic top-k.",
        Operation.RESOLVE_PREFS,
        read_only=True,
        idempotent=True,
    ),
    Operation.PREPARE_TASK_CONTEXT: ToolSpec(
        "Prepare preferences and supporting memories from explicit context. For a durable task use prepare_task.",
        Operation.PREPARE_TASK_CONTEXT,
    ),
    Operation.REMEMBER: ToolSpec(
        "Immediately reconcile and store an explicit memory. For deferred task findings use record_finding.",
        Operation.REMEMBER,
        destructive=True,
    ),
    Operation.DREAM_PLAN: ToolSpec(
        "Prepare and persist a bounded consolidation proposal without applying changes.",
        Operation.DREAM,
        open_world=True,
    ),
    Operation.DREAM_APPLY: ToolSpec(
        "Apply a reviewed dream plan; rejects stale or already-applied proposals.",
        Operation.DREAM,
        destructive=True,
    ),
    RuntimeOperation.BEGIN_TASK: ToolSpec(
        "Start or resume a durable task with objective, project, and explicit attributes before acting.",
        RuntimeOperation.TASK,
        idempotent=True,
    ),
    RuntimeOperation.PREPARE_TASK: ToolSpec(
        "Inject preferences, supporting memories, and resolved knowledge for a task. Set guard=true immediately before an action to block conflicts and unresolved gaps.",
        RuntimeOperation.PREPARE_TASK,
        open_world=True,
    ),
    RuntimeOperation.REQUIRE_KNOWLEDGE: ToolSpec(
        "Declare missing information; search memory, then configured research sources. Save supported findings automatically. Unresolved results block guarded actions.",
        RuntimeOperation.KNOWLEDGE,
        open_world=True,
    ),
    RuntimeOperation.RECORD_FINDING: ToolSpec(
        "Durably capture an evidenced discovery for reconciliation. Reuse the same event_id and finding on retries.",
        RuntimeOperation.CAPTURE,
        idempotent=True,
    ),
    RuntimeOperation.COMPLETE_TASK: ToolSpec(
        "Record an explicit task outcome and optional findings. Success requires resolved knowledge gaps. Replay the same content on retries.",
        RuntimeOperation.CAPTURE,
        idempotent=True,
    ),
    RuntimeOperation.MAINTENANCE_STATUS: ToolSpec(
        "Inspect owner-scoped queued, running, failed, or proposed maintenance with pagination.",
        RuntimeOperation.MAINTENANCE_STATUS,
        read_only=True,
        idempotent=True,
    ),
    RuntimeOperation.REVIEW_JOB: ToolSpec(
        "Accept or reject a concrete maintenance proposal after reviewing its evidence; acceptance may modify existing memories.",
        RuntimeOperation.REVIEW_JOB,
        destructive=True,
    ),
    RuntimeOperation.RETRY_JOB: ToolSpec(
        "Retry failed, proposed, or rejected maintenance using fresh reasoning; discards prior approval.",
        RuntimeOperation.RETRY_JOB,
    ),
}


def _references(value):
    if isinstance(value, dict):
        if "$ref" in value:
            yield value["$ref"].rsplit("/", 1)[-1]
        for child in value.values():
            yield from _references(child)
    elif isinstance(value, list):
        for child in value:
            yield from _references(child)


def operation_schema(schema, operation):
    """Keep only the selected model's reachable definitions, including recursive filters."""
    reference = schema["discriminator"]["mapping"][operation]
    definitions = schema["$defs"]
    result = deepcopy(definitions[reference.rsplit("/", 1)[-1]])
    required = set(_references(result))
    remaining = list(required)
    while remaining:
        for name in _references(definitions[remaining.pop()]):
            if name not in required:
                required.add(name)
                remaining.append(name)
    if required:
        result["$defs"] = {
            name: deepcopy(definitions[name]) for name in sorted(required)
        }
    return result


def schemas():
    requests, responses = REQUEST_ADAPTER.json_schema(), RESPONSE_ADAPTER.json_schema()
    if set(requests["discriminator"]["mapping"]) != SPECS.keys():
        raise RuntimeError("every runtime operation must have explicit tool semantics")
    return {
        name: (
            operation_schema(requests, name),
            operation_schema(responses, spec.response),
        )
        for name, spec in SPECS.items()
    }
