"""Tool descriptions and minimal schemas derived from the typed dispatcher."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from .models import REQUEST_ADAPTER, RESPONSE_ADAPTER


@dataclass(frozen=True, slots=True)
class ToolSpec:
    description: str
    response: str
    read_only: bool = False
    destructive: bool = False
    idempotent: bool = False
    open_world: bool = False


# These are protocol operations, not a taxonomy of memory topics or user activities.
SPECS = {
    "describe_schema": ToolSpec(
        "Discover memory fields and supported typed operations before using unfamiliar metadata.",
        "describe_schema",
        read_only=True,
        idempotent=True,
    ),
    "list_metadata_fields": ToolSpec(
        "List registered metadata fields and aliases for this owner.",
        "list_metadata_fields",
        read_only=True,
        idempotent=True,
    ),
    "list_metadata_values": ToolSpec(
        "Page through known values for a registered field.",
        "list_metadata_values",
        read_only=True,
        idempotent=True,
    ),
    "register_metadata_field": ToolSpec(
        "Register a typed metadata field or alias before recording new attributes.",
        "register_metadata_field",
    ),
    "recall": ToolSpec(
        "Retrieve a bounded set of scoped memories relevant to a query; similarity is not proof of correctness.",
        "recall",
        read_only=True,
        idempotent=True,
    ),
    "resolve_prefs": ToolSpec(
        "Resolve all applicable preferences, including conflicts, independently of semantic top-k.",
        "resolve_prefs",
        read_only=True,
        idempotent=True,
    ),
    "prepare_task_context": ToolSpec(
        "Prepare preferences and supporting memories from explicit context. For a durable task use prepare_task.",
        "prepare_task_context",
    ),
    "remember": ToolSpec(
        "Immediately reconcile and store an explicit memory. For deferred task findings use record_finding.",
        "remember",
        destructive=True,
    ),
    "dream_plan": ToolSpec(
        "Prepare and persist a bounded consolidation proposal without applying changes.",
        "dream",
        open_world=True,
    ),
    "dream_apply": ToolSpec(
        "Apply a reviewed dream plan; rejects stale or already-applied proposals.",
        "dream",
        destructive=True,
    ),
    "begin_task": ToolSpec(
        "Start or resume a durable task with objective, project, and explicit attributes before acting.",
        "task",
        idempotent=True,
    ),
    "prepare_task": ToolSpec(
        "Inject preferences, supporting memories, and resolved knowledge for a task. Set guard=true immediately before an action to block conflicts and unresolved gaps.",
        "prepare_task",
        open_world=True,
    ),
    "require_knowledge": ToolSpec(
        "Declare missing information; search memory, then configured research sources. Save supported findings automatically. Unresolved results block guarded actions.",
        "knowledge",
        open_world=True,
    ),
    "record_finding": ToolSpec(
        "Durably capture an evidenced discovery for reconciliation. Reuse the same event_id and finding on retries.",
        "capture",
        idempotent=True,
    ),
    "complete_task": ToolSpec(
        "Record an explicit task outcome and optional findings. Success requires resolved knowledge gaps. Replay the same content on retries.",
        "capture",
        idempotent=True,
    ),
    "maintenance_status": ToolSpec(
        "Inspect owner-scoped queued, running, failed, or proposed maintenance with pagination.",
        "maintenance_status",
        read_only=True,
        idempotent=True,
    ),
    "review_job": ToolSpec(
        "Accept or reject a concrete maintenance proposal after reviewing its evidence; acceptance may modify existing memories.",
        "review_job",
        destructive=True,
    ),
    "retry_job": ToolSpec(
        "Retry failed, proposed, or rejected maintenance using fresh reasoning; discards prior approval.",
        "retry_job",
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
