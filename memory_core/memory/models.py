"""Versioned public boundaries. Domain vocabulary is data, never an enum."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    model_validator,
)

Identifier = Annotated[
    str, Field(min_length=1, max_length=128, pattern=r"^[a-z][a-z0-9_.-]*$")
]
Nonempty = Annotated[str, Field(min_length=1, max_length=4096)]
Scalar = str | bool | int | float
MetadataValue = Scalar | list[str]


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class Status(StrEnum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    DISPUTED = "disputed"


class FieldType(StrEnum):
    KEYWORD = "keyword"
    KEYWORDS = "keywords"
    NUMBER = "number"
    BOOLEAN = "boolean"


class FieldDefinition(Model):
    name: Identifier
    type: FieldType
    description: str = ""
    read_only: bool = False
    aliases: list[Identifier] = Field(default_factory=list, max_length=32)


class UserScope(Model):
    kind: Literal["user"] = "user"


class ProjectScope(Model):
    kind: Literal["project"] = "project"
    project_id: Nonempty


Scope = Annotated[UserScope | ProjectScope, Field(discriminator="kind")]


class MemoryBase(Model):
    id: str = Field(default_factory=lambda: str(uuid4()), min_length=1, max_length=128)
    schema_version: Literal[1] = 1
    scope: Scope = Field(default_factory=UserScope)
    status: Status = Status.ACTIVE
    summary: Nonempty
    source_refs: list[Nonempty] = Field(default_factory=list)
    metadata: dict[Identifier, MetadataValue] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    revision: int = Field(default=1, ge=1)
    supersedes_id: str | None = None


class PreferenceMemory(MemoryBase):
    kind: Literal["preference"] = "preference"
    preference_key: Identifier
    value: JsonValue
    # AND across fields, OR across each field's allowed values. Missing task fields do not match.
    applicability: dict[Identifier, list[Scalar]] = Field(default_factory=dict)
    explicit: bool = True


class DecisionMemory(MemoryBase):
    kind: Literal["decision"] = "decision"
    decision: Nonempty
    rationale: Nonempty


class ObservationMemory(MemoryBase):
    kind: Literal["observation"] = "observation"
    observation: Nonempty
    observed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ProcedureMemory(MemoryBase):
    kind: Literal["procedure"] = "procedure"
    trigger: Nonempty
    steps: list[Nonempty] = Field(min_length=1)


Memory = Annotated[
    PreferenceMemory | DecisionMemory | ObservationMemory | ProcedureMemory,
    Field(discriminator="kind"),
]
MEMORY_ADAPTER = TypeAdapter(Memory)


class Eq(Model):
    op: Literal["eq"] = "eq"
    field: Identifier
    value: Scalar


class In(Model):
    op: Literal["in"] = "in"
    field: Identifier
    values: list[Scalar] = Field(max_length=100)


class ContainsAny(Model):
    op: Literal["contains_any"] = "contains_any"
    field: Identifier
    values: list[str] = Field(max_length=100)


class ContainsAll(Model):
    op: Literal["contains_all"] = "contains_all"
    field: Identifier
    values: list[str] = Field(max_length=100)


class Range(Model):
    op: Literal["range"] = "range"
    field: Identifier
    minimum: float | None = None
    maximum: float | None = None

    @model_validator(mode="after")
    def bounds(self):
        if self.minimum is None and self.maximum is None:
            raise ValueError("range needs at least one bound")
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError("minimum exceeds maximum")
        return self


class Exists(Model):
    op: Literal["exists"] = "exists"
    field: Identifier


class All(Model):
    op: Literal["all"] = "all"
    conditions: list[Filter] = Field(max_length=100)


class AnyOf(Model):
    op: Literal["any"] = "any"
    conditions: list[Filter] = Field(max_length=100)


Filter = Annotated[
    Eq | In | ContainsAny | ContainsAll | Range | Exists | All | AnyOf,
    Field(discriminator="op"),
]
All.model_rebuild()
AnyOf.model_rebuild()
FILTER_ADAPTER = TypeAdapter(Filter)


class RetrievalPolicy(Model):
    semantic_weight: float = Field(default=0.7, ge=0, le=1)
    lexical_weight: float = Field(default=0.3, ge=0, le=1)
    candidate_multiplier: int = Field(default=4, ge=1, le=20)
    candidate_floor: int = Field(default=40, ge=1, le=1000)
    minimum_score: float = Field(default=0.0, ge=0, le=1)

    @model_validator(mode="after")
    def weights(self):
        if self.semantic_weight + self.lexical_weight <= 0:
            raise ValueError("at least one retrieval weight must be positive")
        return self


class EmbeddingSpec(Model):
    model_id: Nonempty
    revision: Nonempty
    dimensions: int = Field(ge=1, le=65536)


class TaskContext(Model):
    task_id: Nonempty
    project_id: Nonempty | None = None
    attributes: dict[Identifier, MetadataValue] = Field(default_factory=dict)


class Create(Model):
    action: Literal["create"] = "create"


class TargetAction(Model):
    target_id: Nonempty
    expected_revision: int = Field(ge=1)


class AttachEvidence(TargetAction):
    action: Literal["attach_evidence"] = "attach_evidence"


class Supersede(TargetAction):
    action: Literal["supersede"] = "supersede"


class PreserveException(TargetAction):
    action: Literal["preserve_exception"] = "preserve_exception"


class FlagConflict(TargetAction):
    action: Literal["flag_conflict"] = "flag_conflict"


Reconciliation = Annotated[
    Create | AttachEvidence | Supersede | PreserveException | FlagConflict,
    Field(discriminator="action"),
]
DECISION_ADAPTER = TypeAdapter(Reconciliation)


class RememberRequest(Model):
    operation: Literal["remember"] = "remember"
    memory: Memory
    embedding: list[float] | None = None
    decision: Reconciliation | None = None
    # Required with an explicit decision: guards the candidate snapshot, not just one row.
    expected_generation: int | None = Field(default=None, ge=0)


class RecallRequest(Model):
    operation: Literal["recall"] = "recall"
    query: Nonempty
    project_id: Nonempty | None = None
    filter: Filter | None = None
    statuses: list[Status] = Field(
        default_factory=lambda: [Status.ACTIVE], max_length=3
    )
    limit: int = Field(default=10, ge=1, le=100)
    embedding: list[float] | None = None


class ResolvePreferencesRequest(Model):
    operation: Literal["resolve_prefs"] = "resolve_prefs"
    context: TaskContext


class PrepareContextRequest(Model):
    operation: Literal["prepare_task_context"] = "prepare_task_context"
    context: TaskContext
    query: Nonempty | None = None
    limit: int = Field(default=5, ge=1, le=100)


class DescribeSchemaRequest(Model):
    operation: Literal["describe_schema"] = "describe_schema"


class ListFieldsRequest(Model):
    operation: Literal["list_metadata_fields"] = "list_metadata_fields"


class RegisterFieldRequest(Model):
    operation: Literal["register_metadata_field"] = "register_metadata_field"
    definition: FieldDefinition


class ListValuesRequest(Model):
    operation: Literal["list_metadata_values"] = "list_metadata_values"
    field: Identifier
    prefix: str = ""
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=50, ge=1, le=100)


class MemoryHit(Model):
    memory: Memory
    score: float


class DreamPlanRequest(Model):
    operation: Literal["dream_plan"] = "dream_plan"
    batch_size: int = Field(default=16, ge=1, le=100)
    neighbors_per_memory: int = Field(default=8, ge=1, le=30)


class DreamApplyRequest(Model):
    operation: Literal["dream_apply"] = "dream_apply"
    plan_id: Nonempty


class DreamGroup(Model):
    memory: Memory
    neighbors: list[MemoryHit]


class DreamChange(Model):
    memory: Memory
    decision: Reconciliation


class DreamProposal(Model):
    changes: list[DreamChange] = Field(default_factory=list, max_length=100)
    # Index/model changes are recommendations, never an unmeasured automatic promotion.
    recommendations: list[Nonempty] = Field(default_factory=list, max_length=30)


class DreamPlan(Model):
    id: str = Field(default_factory=lambda: str(uuid4()))
    generation: int
    cursor: int
    cursor_id: str = ""
    writes: int
    groups: list[DreamGroup]
    proposal: DreamProposal
    applied: bool = False


class DreamResponse(Model):
    operation: Literal["dream"] = "dream"
    plan: DreamPlan
    due: bool


Request = Annotated[
    RememberRequest
    | RecallRequest
    | ResolvePreferencesRequest
    | PrepareContextRequest
    | DescribeSchemaRequest
    | ListFieldsRequest
    | RegisterFieldRequest
    | ListValuesRequest
    | DreamPlanRequest
    | DreamApplyRequest,
    Field(discriminator="operation"),
]
REQUEST_ADAPTER = TypeAdapter(Request)


class RecallResponse(Model):
    operation: Literal["recall"] = "recall"
    hits: list[MemoryHit]
    generation: int


class PreferenceConflict(Model):
    preference_key: str
    memories: list[PreferenceMemory]


class PreferencesResponse(Model):
    operation: Literal["resolve_prefs"] = "resolve_prefs"
    effective: list[PreferenceMemory]
    conflicts: list[PreferenceConflict]
    complete: bool = True
    generation: int


class RememberResponse(Model):
    operation: Literal["remember"] = "remember"
    memory: Memory
    decision: Reconciliation
    neighbors: list[MemoryHit]
    generation: int
    dream_plan_id: str | None = None
    maintenance_error: str | None = None


class ContextResponse(Model):
    operation: Literal["prepare_task_context"] = "prepare_task_context"
    preferences: PreferencesResponse
    supporting: list[MemoryHit]
    receipt: str


class SchemaResponse(Model):
    operation: Literal["describe_schema"] = "describe_schema"
    schema_version: Literal[1] = 1
    request_schema: dict[str, JsonValue]
    memory_schema: dict[str, JsonValue]
    fields: list[FieldDefinition]
    generation: int


class FieldsResponse(Model):
    operation: Literal["list_metadata_fields"] = "list_metadata_fields"
    fields: list[FieldDefinition]


class FieldResponse(Model):
    operation: Literal["register_metadata_field"] = "register_metadata_field"
    definition: FieldDefinition


class ValuesResponse(Model):
    operation: Literal["list_metadata_values"] = "list_metadata_values"
    values: list[Scalar]
    next_offset: int | None


Response = Annotated[
    RecallResponse
    | PreferencesResponse
    | RememberResponse
    | ContextResponse
    | SchemaResponse
    | FieldsResponse
    | FieldResponse
    | ValuesResponse
    | DreamResponse,
    Field(discriminator="operation"),
]
RESPONSE_ADAPTER = TypeAdapter(Response)
