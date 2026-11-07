"""Public lifecycle boundaries; domain vocabulary remains registry data."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import AwareDatetime, ConfigDict, Field, JsonValue, TypeAdapter

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


class KnowledgeStatus(StrEnum):
    RESOLVED = "resolved"
    NEEDS_USER_INPUT = "needs_user_input"
    UNRESOLVED = "unresolved"


class OutcomeStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


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
    status: OutcomeStatus
    summary: Nonempty
    evidence: list[Evidence] = Field(default_factory=list, max_length=32)


class KnowledgeGap(Model):
    id: Nonempty
    question: Nonempty
    requirements: list[Nonempty] = Field(default_factory=list, max_length=32)
    max_age_seconds: float = Field(default=86400, gt=0, le=31536000)


class ResolvedKnowledge(Model):
    status: Literal[KnowledgeStatus.RESOLVED] = KnowledgeStatus.RESOLVED
    answer: Nonempty
    evidence: list[Evidence] = Field(min_length=1, max_length=32)


class NeedsUserInput(Model):
    status: Literal[KnowledgeStatus.NEEDS_USER_INPUT] = KnowledgeStatus.NEEDS_USER_INPUT
    question: Nonempty


class UnresolvedKnowledge(Model):
    status: Literal[KnowledgeStatus.UNRESOLVED] = KnowledgeStatus.UNRESOLVED
    reason: Nonempty


KnowledgeResult = Annotated[
    ResolvedKnowledge | NeedsUserInput | UnresolvedKnowledge,
    Field(discriminator="status"),
]
KNOWLEDGE_ADAPTER = TypeAdapter(KnowledgeResult)
EVIDENCE_ADAPTER = TypeAdapter(list[Evidence])


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


class RuntimeOperation(StrEnum):
    BEGIN_TASK = "begin_task"
    PREPARE_TASK = "prepare_task"
    REQUIRE_KNOWLEDGE = "require_knowledge"
    RECORD_FINDING = "record_finding"
    COMPLETE_TASK = "complete_task"
    MAINTENANCE_STATUS = "maintenance_status"
    REVIEW_JOB = "review_job"
    RETRY_JOB = "retry_job"
    TASK = "task"
    CAPTURE = "capture"
    KNOWLEDGE = "knowledge"


class JobKind(StrEnum):
    CAPTURE = "capture"
    DREAM = "dream"


class JobResultKind(StrEnum):
    CAPTURE_APPLIED = "capture_applied"
    CAPTURE_PROPOSAL = "capture_proposal"
    DREAM_PROPOSAL = "dream_proposal"
    DREAM_APPLIED = "dream_applied"


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    PROPOSED = "proposed"
    DONE = "done"
    FAILED = "failed"
    REJECTED = "rejected"


class CaptureApplied(Model):
    kind: Literal[JobResultKind.CAPTURE_APPLIED] = JobResultKind.CAPTURE_APPLIED
    memory_id: str
    decision: Reconciliation


class CaptureProposal(Model):
    kind: Literal[JobResultKind.CAPTURE_PROPOSAL] = JobResultKind.CAPTURE_PROPOSAL
    incoming: Memory
    neighbors: list[MemoryHit]
    decision: Reconciliation
    generation: int
    approved: bool = False


class DreamReview(Model):
    kind: Literal[JobResultKind.DREAM_PROPOSAL] = JobResultKind.DREAM_PROPOSAL
    plan_id: str
    plan: DreamPlan
    generation: int
    approved: bool = False


class DreamApplied(Model):
    kind: Literal[JobResultKind.DREAM_APPLIED] = JobResultKind.DREAM_APPLIED
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
    kind: JobKind
    status: JobStatus
    attempts: int
    error: str | None
    result: JobResult | None


class BeginTaskRequest(Model):
    operation: Literal[RuntimeOperation.BEGIN_TASK] = RuntimeOperation.BEGIN_TASK
    context: TaskContext
    objective: Nonempty


class PrepareTaskRequest(Model):
    operation: Literal[RuntimeOperation.PREPARE_TASK] = RuntimeOperation.PREPARE_TASK
    task_id: Nonempty
    guard: bool = False


class PreparedTask(ContextResponse):
    # A distinct response variant reusing ContextResponse's fields; the tag is meant to differ.
    operation: Literal[RuntimeOperation.PREPARE_TASK] = RuntimeOperation.PREPARE_TASK  # pyright: ignore[reportIncompatibleVariableOverride]
    context: TaskContext
    objective: Nonempty
    knowledge: dict[str, KnowledgeResult]


class RequireKnowledgeRequest(Model):
    operation: Literal[RuntimeOperation.REQUIRE_KNOWLEDGE] = (
        RuntimeOperation.REQUIRE_KNOWLEDGE
    )
    task_id: Nonempty
    gap: KnowledgeGap


class RecordFindingRequest(Model):
    operation: Literal[RuntimeOperation.RECORD_FINDING] = (
        RuntimeOperation.RECORD_FINDING
    )
    task_id: Nonempty
    event_id: Nonempty
    finding: Finding


class CompleteTaskRequest(Model):
    operation: Literal[RuntimeOperation.COMPLETE_TASK] = RuntimeOperation.COMPLETE_TASK
    task_id: Nonempty
    outcome: TaskOutcome
    findings: list[Finding] = Field(default_factory=list, max_length=32)


class MaintenanceStatusRequest(Model):
    operation: Literal[RuntimeOperation.MAINTENANCE_STATUS] = (
        RuntimeOperation.MAINTENANCE_STATUS
    )
    status: JobStatus | None = None
    limit: int = Field(default=50, ge=1, le=100)
    offset: int = Field(default=0, ge=0)


class ReviewJobRequest(Model):
    operation: Literal[RuntimeOperation.REVIEW_JOB] = RuntimeOperation.REVIEW_JOB
    job_id: Nonempty
    accept: bool


class RetryJobRequest(Model):
    operation: Literal[RuntimeOperation.RETRY_JOB] = RuntimeOperation.RETRY_JOB
    job_id: Nonempty


class RetryResponse(Model):
    operation: Literal[RuntimeOperation.RETRY_JOB] = RuntimeOperation.RETRY_JOB
    job: JobView


class TaskResponse(Model):
    operation: Literal[RuntimeOperation.TASK] = RuntimeOperation.TASK
    context: TaskContext
    objective: str
    outcome: TaskOutcome | None = None
    sources: dict[str, str] = Field(default_factory=dict)


class CaptureResponse(Model):
    operation: Literal[RuntimeOperation.CAPTURE] = RuntimeOperation.CAPTURE
    job_ids: list[str]


class KnowledgeResponse(Model):
    operation: Literal[RuntimeOperation.KNOWLEDGE] = RuntimeOperation.KNOWLEDGE
    result: KnowledgeResult
    capture: CaptureResponse | None = None


class MaintenanceResponse(Model):
    operation: Literal[RuntimeOperation.MAINTENANCE_STATUS] = (
        RuntimeOperation.MAINTENANCE_STATUS
    )
    jobs: list[JobView]
    next_offset: int | None = None


class ReviewResponse(Model):
    operation: Literal[RuntimeOperation.REVIEW_JOB] = RuntimeOperation.REVIEW_JOB
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
type RuntimeRequestInput = RuntimeRequest | dict[str, JsonValue]
# TypeAdapter.json_schema() is typed dict[str, Any] by pydantic.
type JsonSchema = dict[str, Any]
REQUEST_ADAPTER: TypeAdapter[RuntimeRequest] = TypeAdapter(RuntimeRequest)
RESPONSE_ADAPTER: TypeAdapter[RuntimeResponse] = TypeAdapter(RuntimeResponse)
