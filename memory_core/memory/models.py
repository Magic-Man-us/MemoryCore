"""Versioned public boundaries. Domain vocabulary is data, never an enum."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal, Self
from uuid import uuid4

from pydantic import (
    AfterValidator,
    AwareDatetime,
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


def _to_utc(value: datetime) -> datetime:
    return value.astimezone(UTC)


# Normalized to UTC so ordering compares instants: datetimes sharing a tzinfo compare
# by wall clock and ignore `fold` across a daylight-saving transition.
UtcDatetime = Annotated[AwareDatetime, AfterValidator(_to_utc)]
Scalar = str | bool | int | float
MetadataValue = Scalar | list[str]


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class Status(StrEnum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    DISPUTED = "disputed"


class ScopeKind(StrEnum):
    USER = "user"
    PROJECT = "project"


class MemoryKind(StrEnum):
    PREFERENCE = "preference"
    DECISION = "decision"
    OBSERVATION = "observation"
    PROCEDURE = "procedure"


class FilterOp(StrEnum):
    EQ = "eq"
    IN = "in"
    CONTAINS_ANY = "contains_any"
    CONTAINS_ALL = "contains_all"
    RANGE = "range"
    EXISTS = "exists"
    ALL = "all"
    ANY = "any"


class ReconciliationAction(StrEnum):
    CREATE = "create"
    ATTACH_EVIDENCE = "attach_evidence"
    SUPERSEDE = "supersede"
    PRESERVE_EXCEPTION = "preserve_exception"
    FLAG_CONFLICT = "flag_conflict"


class Operation(StrEnum):
    REMEMBER = "remember"
    RECALL = "recall"
    RESOLVE_PREFS = "resolve_prefs"
    PREPARE_TASK_CONTEXT = "prepare_task_context"
    DESCRIBE_SCHEMA = "describe_schema"
    LIST_METADATA_FIELDS = "list_metadata_fields"
    REGISTER_METADATA_FIELD = "register_metadata_field"
    LIST_METADATA_VALUES = "list_metadata_values"
    DREAM_PLAN = "dream_plan"
    DREAM_APPLY = "dream_apply"
    DREAM = "dream"


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
    kind: Literal[ScopeKind.USER] = ScopeKind.USER

    @property
    def project_id(self) -> None:
        # Uniform with ProjectScope so callers read scope.project_id for either variant.
        return None


class ProjectScope(Model):
    kind: Literal[ScopeKind.PROJECT] = ScopeKind.PROJECT
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
    # Validity window: unset bounds mean always valid. Recall and preference
    # resolution skip records outside it; supersession history stays in `status`.
    valid_from: UtcDatetime | None = None
    valid_until: UtcDatetime | None = None

    @model_validator(mode="after")
    def validity_window(self) -> Self:
        if (
            self.valid_from is not None
            and self.valid_until is not None
            and self.valid_until <= self.valid_from
        ):
            raise ValueError("valid_until must be after valid_from")
        return self


class PreferenceMemory(MemoryBase):
    kind: Literal[MemoryKind.PREFERENCE] = MemoryKind.PREFERENCE
    preference_key: Identifier
    value: JsonValue
    # AND across fields, OR across each field's allowed values. Missing task fields do not match.
    applicability: dict[Identifier, list[Scalar]] = Field(default_factory=dict)
    explicit: bool = True


class DecisionMemory(MemoryBase):
    kind: Literal[MemoryKind.DECISION] = MemoryKind.DECISION
    decision: Nonempty
    rationale: Nonempty


class ObservationMemory(MemoryBase):
    kind: Literal[MemoryKind.OBSERVATION] = MemoryKind.OBSERVATION
    observation: Nonempty
    observed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ProcedureMemory(MemoryBase):
    kind: Literal[MemoryKind.PROCEDURE] = MemoryKind.PROCEDURE
    trigger: Nonempty
    steps: list[Nonempty] = Field(min_length=1)


Memory = Annotated[
    PreferenceMemory | DecisionMemory | ObservationMemory | ProcedureMemory,
    Field(discriminator="kind"),
]
MEMORY_ADAPTER = TypeAdapter(Memory)


class Eq(Model):
    op: Literal[FilterOp.EQ] = FilterOp.EQ
    field: Identifier
    value: Scalar


class In(Model):
    op: Literal[FilterOp.IN] = FilterOp.IN
    field: Identifier
    values: list[Scalar] = Field(max_length=100)


class ContainsAny(Model):
    op: Literal[FilterOp.CONTAINS_ANY] = FilterOp.CONTAINS_ANY
    field: Identifier
    values: list[str] = Field(max_length=100)


class ContainsAll(Model):
    op: Literal[FilterOp.CONTAINS_ALL] = FilterOp.CONTAINS_ALL
    field: Identifier
    values: list[str] = Field(max_length=100)


class Range(Model):
    op: Literal[FilterOp.RANGE] = FilterOp.RANGE
    field: Identifier
    minimum: float | None = None
    maximum: float | None = None

    @model_validator(mode="after")
    def bounds(self) -> Self:
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
    op: Literal[FilterOp.EXISTS] = FilterOp.EXISTS
    field: Identifier


class All(Model):
    op: Literal[FilterOp.ALL] = FilterOp.ALL
    conditions: list[Filter] = Field(max_length=100)


class AnyOf(Model):
    op: Literal[FilterOp.ANY] = FilterOp.ANY
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
    def weights(self) -> Self:
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
    action: Literal[ReconciliationAction.CREATE] = ReconciliationAction.CREATE


class TargetAction(Model):
    target_id: Nonempty
    expected_revision: int = Field(ge=1)


class AttachEvidence(TargetAction):
    action: Literal[ReconciliationAction.ATTACH_EVIDENCE] = (
        ReconciliationAction.ATTACH_EVIDENCE
    )


class Supersede(TargetAction):
    action: Literal[ReconciliationAction.SUPERSEDE] = ReconciliationAction.SUPERSEDE


class PreserveException(TargetAction):
    action: Literal[ReconciliationAction.PRESERVE_EXCEPTION] = (
        ReconciliationAction.PRESERVE_EXCEPTION
    )


class FlagConflict(TargetAction):
    action: Literal[ReconciliationAction.FLAG_CONFLICT] = (
        ReconciliationAction.FLAG_CONFLICT
    )


Reconciliation = Annotated[
    Create | AttachEvidence | Supersede | PreserveException | FlagConflict,
    Field(discriminator="action"),
]
DECISION_ADAPTER = TypeAdapter(Reconciliation)


class RememberRequest(Model):
    operation: Literal[Operation.REMEMBER] = Operation.REMEMBER
    memory: Memory
    embedding: list[float] | None = None
    decision: Reconciliation | None = None
    # Required with an explicit decision: guards the candidate snapshot, not just one row.
    expected_generation: int | None = Field(default=None, ge=0)


class RecallRequest(Model):
    operation: Literal[Operation.RECALL] = Operation.RECALL
    query: Nonempty
    project_id: Nonempty | None = None
    filter: Filter | None = None
    statuses: list[Status] = Field(
        default_factory=lambda: [Status.ACTIVE], max_length=3
    )
    limit: int = Field(default=10, ge=1, le=100)
    embedding: list[float] | None = None
    as_of: UtcDatetime | None = None


class ResolvePreferencesRequest(Model):
    operation: Literal[Operation.RESOLVE_PREFS] = Operation.RESOLVE_PREFS
    context: TaskContext


class PrepareContextRequest(Model):
    operation: Literal[Operation.PREPARE_TASK_CONTEXT] = Operation.PREPARE_TASK_CONTEXT
    context: TaskContext
    query: Nonempty | None = None
    limit: int = Field(default=5, ge=1, le=100)


class DescribeSchemaRequest(Model):
    operation: Literal[Operation.DESCRIBE_SCHEMA] = Operation.DESCRIBE_SCHEMA


class ListFieldsRequest(Model):
    operation: Literal[Operation.LIST_METADATA_FIELDS] = Operation.LIST_METADATA_FIELDS


class RegisterFieldRequest(Model):
    operation: Literal[Operation.REGISTER_METADATA_FIELD] = (
        Operation.REGISTER_METADATA_FIELD
    )
    definition: FieldDefinition


class ListValuesRequest(Model):
    # None retains owner-wide discovery for trusted host callers.
    scope: Scope | None = None
    operation: Literal[Operation.LIST_METADATA_VALUES] = Operation.LIST_METADATA_VALUES
    field: Identifier
    prefix: str = ""
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=50, ge=1, le=100)


class MemoryHit(Model):
    memory: Memory
    score: float


class DreamPlanRequest(Model):
    operation: Literal[Operation.DREAM_PLAN] = Operation.DREAM_PLAN
    batch_size: int = Field(default=16, ge=1, le=100)
    neighbors_per_memory: int = Field(default=8, ge=1, le=30)


class DreamApplyRequest(Model):
    operation: Literal[Operation.DREAM_APPLY] = Operation.DREAM_APPLY
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
    operation: Literal[Operation.DREAM] = Operation.DREAM
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
REQUEST_ADAPTER: TypeAdapter[Request] = TypeAdapter(Request)


class RecallResponse(Model):
    operation: Literal[Operation.RECALL] = Operation.RECALL
    hits: list[MemoryHit]
    generation: int


class PreferenceConflict(Model):
    preference_key: str
    memories: list[PreferenceMemory]


class PreferencesResponse(Model):
    operation: Literal[Operation.RESOLVE_PREFS] = Operation.RESOLVE_PREFS
    effective: list[PreferenceMemory]
    conflicts: list[PreferenceConflict]
    complete: bool = True
    generation: int


class RememberResponse(Model):
    operation: Literal[Operation.REMEMBER] = Operation.REMEMBER
    memory: Memory
    decision: Reconciliation
    neighbors: list[MemoryHit]
    generation: int
    dream_plan_id: str | None = None
    maintenance_error: str | None = None


class ContextResponse(Model):
    operation: Literal[Operation.PREPARE_TASK_CONTEXT] = Operation.PREPARE_TASK_CONTEXT
    preferences: PreferencesResponse
    supporting: list[MemoryHit]
    receipt: str


class SchemaResponse(Model):
    operation: Literal[Operation.DESCRIBE_SCHEMA] = Operation.DESCRIBE_SCHEMA
    schema_version: Literal[1] = 1
    request_schema: dict[str, JsonValue]
    memory_schema: dict[str, JsonValue]
    fields: list[FieldDefinition]
    generation: int


class FieldsResponse(Model):
    operation: Literal[Operation.LIST_METADATA_FIELDS] = Operation.LIST_METADATA_FIELDS
    fields: list[FieldDefinition]


class FieldResponse(Model):
    operation: Literal[Operation.REGISTER_METADATA_FIELD] = (
        Operation.REGISTER_METADATA_FIELD
    )
    definition: FieldDefinition


class ValuesResponse(Model):
    operation: Literal[Operation.LIST_METADATA_VALUES] = Operation.LIST_METADATA_VALUES
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
