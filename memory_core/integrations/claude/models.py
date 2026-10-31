"""Bounded run configuration and explicit completion/capture results."""

from __future__ import annotations

from pydantic import Field

from memory_core.memory.errors import ConflictError, SchemaError
from memory_core.memory.models import Nonempty, TaskContext
from memory_core.runtime.models import (
    CaptureResponse,
    KnowledgeGap,
    Model,
    TaskRunResult,
)


class ClaudeMemoryPolicy(Model):
    max_concurrent_runs: int = Field(default=4, ge=1, le=256)
    max_pending_runs: int = Field(default=16, ge=0, le=4096)
    admission_timeout_seconds: float = Field(default=30, gt=0)
    run_timeout_seconds: float = Field(default=300, gt=0)
    hook_timeout_seconds: float = Field(default=15, gt=0)
    cleanup_timeout_seconds: float = Field(default=15, gt=0)
    max_context_chars: int = Field(default=32768, ge=1024, le=1048576)
    max_tool_output_chars: int = Field(default=65536, ge=1024, le=1048576)
    max_recall_hits: int = Field(default=10, ge=1, le=100)
    max_evidence_records: int = Field(default=128, ge=1, le=4096)
    # Explicit host-selected recovery tools may run while knowledge is unresolved.
    # No shell command is presumed safe from its name or contents.
    research_tools: frozenset[str] = frozenset()
    server_name: str = Field(default="memorycore", pattern=r"^[a-z][a-z0-9_]{0,31}$")


class ClaudeTaskRequest(Model):
    owner_id: Nonempty
    context: TaskContext
    objective: Nonempty
    required_knowledge: list[KnowledgeGap] = Field(default_factory=list, max_length=32)


class ClaudeRunReport(Model):
    result: TaskRunResult
    sdk_session_id: str | None = None
    capture: CaptureResponse | None = None
    capture_error: str | None = None
    execution_error: str | None = None
    cleanup_error: str | None = None


class AgentCapacityError(RuntimeError):
    """Admission queue is full or its wait budget elapsed; no SDK run was started."""


class AgentRunConflict(ConflictError):
    """This adapter already has a run for the owner/task pair."""


class ContextBudgetError(SchemaError):
    """Mandatory memory context exceeds the configured budget; never silently omit it."""
