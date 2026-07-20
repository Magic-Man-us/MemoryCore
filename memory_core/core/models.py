"""Pydantic contracts for traces, candidates, and recall results.

The SMK enums are imported at runtime on purpose: Pydantic evaluates field
annotations when the model class is built, so moving these under TYPE_CHECKING
leaves ``AssistantMemoryTrace`` permanently "not fully defined"
(see docs/review-junovera-integration.md, P0-2).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from memory_core.types.smk_types import MemoryKind, ToolFlag, TopicBucket


class AssistantMemoryTrace(BaseModel):
    """Normalized record describing a single assistant-focused learning trace."""

    model_config = ConfigDict(frozen=False, arbitrary_types_allowed=True)

    trace_uid: str

    # situation / context
    context_topic: TopicBucket
    context_activity: str = Field(description="e.g. 'debugging', 'designing', 'explaining'")
    context_artifacts: list[str] = Field(default_factory=list)
    context_complexity: float = Field(ge=0.0, le=1.0)

    # what we actually learned
    kind: MemoryKind
    summary: str
    rationale: str
    before_state_confusion: float = Field(ge=0.0, le=1.0)
    after_state_confidence: float = Field(ge=0.0, le=1.0)
    generality: float = Field(ge=0.0, le=1.0)

    # "cues" for when this should fire again
    tools: set[ToolFlag] = Field(default_factory=set)
    languages: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)
    trigger_smells: list[str] = Field(
        default_factory=list,
        description="Patterns like 'tool isolated PATH', 'version mismatch', etc.",
    )

    # meta
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    importance: float = Field(ge=0.0, le=1.0)
    access_count: int = 0
    extra: dict[str, Any] = Field(default_factory=dict)

    # link to the compact SMK key (what Rust uses)
    smk_raw: int | None = None  # 64-bit integer StructuredMemoryKey


class MemoryTrace(BaseModel):
    """User-facing long-term memory record persisted across sessions."""

    model_config = ConfigDict(frozen=False)

    trace_uid: str
    user_id: str
    content: str
    summary: str
    importance: float = Field(ge=0.0, le=1.0)
    created_at: datetime
    access_count: int = 0
    tags: set[str] = Field(default_factory=set)
    extra: dict[str, Any] = Field(default_factory=dict)

    def to_rust_args(self) -> tuple[str, str, str, float, int, int, list[str]]:
        """Serialize the trace into the argument order PyMemoryEngine expects.

        Returns:
            tuple: (trace_uid, user_id, summary, importance, created_at_ts, access_count, tags)
        """
        # Keeping this as an explicit tuple is clearer and safer than relying on
        # model_dump / as_tuple ordering, which can accidentally change.
        return (
            self.trace_uid,
            self.user_id,
            self.summary,
            float(self.importance),
            int(self.created_at.timestamp()),
            int(self.access_count),
            list(self.tags),
        )


class MemoryCandidate(BaseModel):
    """Search result candidate produced by the Rust memory index."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    trace_uid: str
    score: float
    summary: str
    tags: list[str]
    created_at: datetime


class RecallResult(BaseModel):
    """Everything one recall returns, across the layers."""

    model_config = ConfigDict(frozen=False)

    ltm_candidates: list[MemoryCandidate] = Field(default_factory=list)
    wm_events: list[dict[str, Any]] = Field(default_factory=list)
    stm_traces: list[MemoryTrace] = Field(default_factory=list)
