"""SDK hooks guard actions without granting permissions or replacing host hooks."""

from __future__ import annotations

import asyncio
import json

from pydantic import TypeAdapter

from memory_core.memory.errors import MemoryError
from memory_core.runtime.models import Evidence

from .binding import OPERATIONS

EVIDENCE_ADAPTER = TypeAdapter(list[Evidence])


class MemoryHooks:
    def __init__(self, binding, evidence_extractor=None):
        self.binding = binding
        self.extractor = evidence_extractor
        self.failure = None
        self.memory_tools = {
            f"mcp__{binding.policy.server_name}__{name}" for name in OPERATIONS
        }

    async def before_tool(self, input, tool_use_id, context):
        name = input.get("tool_name", "")
        if name in self.memory_tools or name == "StructuredOutput":
            return {}  # Recovery tools must remain callable while a gap is unresolved.
        try:
            async with asyncio.timeout(self.binding.policy.hook_timeout_seconds):
                prepared = await self.binding.prepare(
                    guard=name not in self.binding.policy.research_tools
                )
                rendered = self.binding.render(prepared)
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "additionalContext": rendered,
                }
            }
        except Exception as exc:  # noqa: BLE001 — fail closed at the SDK action boundary
            reason = str(exc) if isinstance(exc, MemoryError) else type(exc).__name__
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": f"MemoryCore blocked this action: {reason}. Resolve missing knowledge through the memory tools or ask the user.",
                }
            }

    async def after_tool(self, input, tool_use_id, context):
        if self.extractor is None or input.get("tool_name") in self.memory_tools | {
            "StructuredOutput"
        }:
            return {}
        try:
            async with asyncio.timeout(self.binding.policy.hook_timeout_seconds):
                values = await self.extractor(input)
                evidence = EVIDENCE_ADAPTER.validate_python(values)
                if len(evidence) > self.binding.policy.max_evidence_records:
                    raise ValueError("tool extractor exceeded evidence limit")
                rendered = json.dumps([e.model_dump(mode="json") for e in evidence])
                if len(rendered) > self.binding.policy.max_context_chars:
                    raise ValueError("tool evidence exceeded context budget")
                self.binding.remember_evidence(evidence)
            return {
                "hookSpecificOutput": {
                    "hookEventName": input.get("hook_event_name", "PostToolUse"),
                    "additionalContext": "Observed evidence available for memory findings: "
                    + rendered,
                }
            }
        except Exception as exc:  # noqa: BLE001 — executed tools cannot be retroactively denied
            self.failure = type(exc).__name__
            return {
                "continue_": False,
                "stopReason": "MemoryCore evidence capture failed; the task must not report verified success.",
            }
