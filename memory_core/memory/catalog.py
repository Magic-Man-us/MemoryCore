"""Schema registration and discovery operations."""

from __future__ import annotations

from sqlalchemy import text

from .errors import SchemaError
from .metadata import Registry
from .models import (
    MEMORY_ADAPTER,
    REQUEST_ADAPTER,
    DescribeSchemaRequest,
    FieldResponse,
    FieldsResponse,
    ListFieldsRequest,
    ListValuesRequest,
    RegisterFieldRequest,
    SchemaResponse,
    Status,
    ValuesResponse,
)


def catalog(service, request, owner):
    transaction = (
        service.store.transaction(owner)
        if isinstance(request, RegisterFieldRequest)
        else service.store.read_transaction(
            owner, embedding_spec=service.spec.model_dump_json()
        )
    )
    with service._owner_lock(owner), transaction as session:
        cache, registry, state = service._snapshot(session, owner)
        if isinstance(request, DescribeSchemaRequest):
            return SchemaResponse(
                request_schema=REQUEST_ADAPTER.json_schema(),
                memory_schema=MEMORY_ADAPTER.json_schema(),
                fields=[
                    field.model_copy(deep=True) for field in registry.fields.values()
                ],
                generation=state["generation"],
            )
        if isinstance(request, ListFieldsRequest):
            return FieldsResponse(
                fields=[
                    field.model_copy(deep=True) for field in registry.fields.values()
                ]
            )
        if isinstance(request, RegisterFieldRequest):
            field = request.definition
            old = registry.fields.get(field.name)
            if field.read_only and old != field:
                raise SchemaError("only the engine defines read-only fields")
            if old and old != field:
                raise SchemaError(
                    "registered field definitions are immutable; register a versioned field"
                )
            Registry(
                [
                    *([f for f in registry.fields.values() if f.name != field.name]),
                    field,
                ]
            )
            if field.name in registry.aliases:
                raise SchemaError("field name already used as an alias")
            if not old:
                session.execute(
                    text("INSERT INTO memory_fields VALUES (:o,:n,:d)"),
                    {"o": owner, "n": field.name, "d": field.model_dump_json()},
                )
                service.store.bump(
                    session,
                    owner,
                    "register_field",
                    {"field": field.name},
                    write=False,
                )
            return FieldResponse(definition=field)
        if isinstance(request, ListValuesRequest):
            field = registry.resolve(request.field)
            # Apply scope before pagination so hidden values cannot affect results.
            eligible = cache.by_status[Status.ACTIVE]
            if request.scope is not None:
                eligible = eligible & cache.eligible(request.scope.project_id)
            values = sorted(
                (
                    key[1]
                    for key, ids in cache.metadata.postings.get(field.name, {}).items()
                    if ids & eligible and str(key[1]).startswith(request.prefix)
                ),
                key=str,
            )
            end = request.offset + request.limit
            return ValuesResponse(
                values=values[request.offset : end],
                next_offset=end if end < len(values) else None,
            )
        raise TypeError("unsupported memory operation")
