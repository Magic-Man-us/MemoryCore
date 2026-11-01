"""Public lifecycle boundaries; domain vocabulary remains registry data."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import AwareDatetime, ConfigDict, Field, TypeAdapter

from memory_core.memory.models import (
    ContextResponse,
    DreamPlan,
    FieldDefinition,
    Identifier,
    Memory,
    MemoryHit,
    MetadataValue,
    Nonempty,
    Reconciliation,
    Request,
    Response,
    TaskContext,
)
from memory_core.memory.models import (
    Model as MemoryModel,
)


class Model(MemoryModel):
    model_config = ConfigDict(revalidate_instances="always")


class Evidence(Model):
    id: Nonempty
    uri: Nonempty
    excerpt: Nonempty
    observed_at: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))


class Finding(Model):
    statement: Nonempty
    evidence: list[Evidence] = Field(min_length=1, max_length=32)


class TaskOutcome(Model):
    status: Literal["succeeded", "failed", "cancelled"]
    summary: Nonempty
    evidence: list[Evidence] = Field(default_factory=list, max_length=32)


class KnowledgeGap(Model):
    id: Nonempty
    question: Nonempty
    requirements: list[Nonempty] = Field(default_factory=list, max_length=32)
    max_age_seconds: float = Field(default=86400, gt=0, le=31536000)


class ResolvedKnowledge(Model):
    status: Literal["resolved"] = "resolved"
    answer: Nonempty
    evidence: list[Evidence] = Field(min_length=1, max_length=32)


class NeedsUserInput(Model):
    status: Literal["needs_user_input"] = "needs_user_input"
    question: Nonempty


class UnresolvedKnowledge(Model):
    status: Literal["unresolved"] = "unresolved"
    reason: Nonempty


KnowledgeResult = Annotated[
    ResolvedKnowledge | NeedsUserInput | UnresolvedKnowledge,
    Field(discriminator="status"),
]
KNOWLEDGE_ADAPTER = TypeAdapter(KnowledgeResult)


class KnowledgeInput(Model):
    context: TaskContext
    gap: KnowledgeGap
    candidates: list[MemoryHit]
    evidence: list[Evidence]
    fields: list[FieldDefinition]


class ContextContribution(Model):
    source: Nonempty
    attributes: dict[Identifier, MetadataValue] = Field(default_factory=dict)


class ContextInput(Model):
    context: TaskContext
    objective: Nonempty
    fields: list[FieldDefinition]


class MaintenancePolicy(Model):
    background: bool = True
    poll_seconds: float = Field(default=0.5, gt=0, le=60)
    lease_seconds: float = Field(default=120, ge=1)
    job_timeout_seconds: float = Field(default=60, gt=0)
    provider_timeout_seconds: float = Field(default=30, gt=0)
    max_attempts: int = Field(default=5, ge=1, le=100)
    retry_seconds: float = Field(default=1, ge=0, le=3600)
    dream_after_writes: int = Field(default=100, ge=1)
    dream_batch_size: int = Field(default=16, ge=1, le=100)
    apply_exact_duplicates: bool = True
    apply_semantic_changes: bool = False


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    PROPOSED = "proposed"
    DONE = "done"
    FAILED = "failed"
    REJECTED = "rejected"


class CaptureApplied(Model):
    kind: Literal["capture_applied"] = "capture_applied"
    memory_id: str
    decision: Reconciliation


class CaptureProposal(Model):
    kind: Literal["capture_proposal"] = "capture_proposal"
    incoming: Memory
    neighbors: list[MemoryHit]
    decision: Reconciliation
    generation: int
    approved: bool = False


class DreamReview(Model):
    kind: Literal["dream_proposal"] = "dream_proposal"
    plan_id: str
    plan: DreamPlan
    generation: int
    approved: bool = False


class DreamApplied(Model):
    kind: Literal["dream_applied"] = "dream_applied"
    plan_id: str
    applied: bool
    recommendations: list[str]


JobResult = Annotated[
    CaptureApplied | CaptureProposal | DreamReview | DreamApplied,
    Field(discriminator="kind"),
]


class JobView(Model):
    id: str
    task_id: str | None
    kind: Literal["capture", "dream"]
    status: JobStatus
    attempts: int
    error: str | None
    result: JobResult | None


class BeginTaskRequest(Model):
    operation: Literal["begin_task"] = "begin_task"
    context: TaskContext
    objective: Nonempty


class PrepareTaskRequest(Model):
    operation: Literal["prepare_task"] = "prepare_task"
    task_id: Nonempty
    guard: bool = False


class PreparedTask(ContextResponse):
    operation: Literal["prepare_task"] = "prepare_task"
    context: TaskContext
    objective: Nonempty
    knowledge: dict[str, KnowledgeResult]


class RequireKnowledgeRequest(Model):
    operation: Literal["require_knowledge"] = "require_knowledge"
    task_id: Nonempty
    gap: KnowledgeGap


class RecordFindingRequest(Model):
    operation: Literal["record_finding"] = "record_finding"
    task_id: Nonempty
    event_id: Nonempty
    finding: Finding


class CompleteTaskRequest(Model):
    operation: Literal["complete_task"] = "complete_task"
    task_id: Nonempty
    outcome: TaskOutcome
    findings: list[Finding] = Field(default_factory=list, max_length=32)


class MaintenanceStatusRequest(Model):
    operation: Literal["maintenance_status"] = "maintenance_status"
    status: JobStatus | None = None
    limit: int = Field(default=50, ge=1, le=100)
    offset: int = Field(default=0, ge=0)


class ReviewJobRequest(Model):
    operation: Literal["review_job"] = "review_job"
    job_id: Nonempty
    accept: bool


class RetryJobRequest(Model):
    operation: Literal["retry_job"] = "retry_job"
    job_id: Nonempty


class RetryResponse(Model):
    operation: Literal["retry_job"] = "retry_job"
    job: JobView


class TaskResponse(Model):
    operation: Literal["task"] = "task"
    context: TaskContext
    objective: str
    outcome: TaskOutcome | None = None
    sources: dict[str, str] = Field(default_factory=dict)


class CaptureResponse(Model):
    operation: Literal["capture"] = "capture"
    job_ids: list[str]


class KnowledgeResponse(Model):
    operation: Literal["knowledge"] = "knowledge"
    result: KnowledgeResult
    capture: CaptureResponse | None = None


class MaintenanceResponse(Model):
    operation: Literal["maintenance_status"] = "maintenance_status"
    jobs: list[JobView]
    next_offset: int | None = None


class ReviewResponse(Model):
    operation: Literal["review_job"] = "review_job"
    job: JobView


class TaskRunResult(Model):
    outcome: TaskOutcome
    findings: list[Finding] = Field(default_factory=list, max_length=32)


RuntimeRequest = Annotated[
    Request
    | PrepareTaskRequest
    | BeginTaskRequest
    | RequireKnowledgeRequest
    | RecordFindingRequest
    | CompleteTaskRequest
    | MaintenanceStatusRequest
    | ReviewJobRequest
    | RetryJobRequest,
    Field(discriminator="operation"),
]
RuntimeResponse = Annotated[
    Response
    | PreparedTask
    | TaskResponse
    | CaptureResponse
    | KnowledgeResponse
    | MaintenanceResponse
    | ReviewResponse
    | RetryResponse,
    Field(discriminator="operation"),
]
REQUEST_ADAPTER = TypeAdapter(RuntimeRequest)
RESPONSE_ADAPTER = TypeAdapter(RuntimeResponse)
