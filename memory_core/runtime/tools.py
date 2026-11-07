"""Transport-neutral tool adapter; authenticated identity never comes from tool arguments.

Tool arguments are model-authored. A model can report what it inferred, but it cannot
assert that the user stated a preference: ``explicit`` is trusted provenance, so it is
set by host code calling ``MemoryCore.call`` directly, never by a tool call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel

from memory_core.memory.models import PreferenceMemory, RememberRequest

from .models import REQUEST_ADAPTER, RESPONSE_ADAPTER

if TYPE_CHECKING:
    from pydantic import JsonValue

    from .models import (
        JsonSchema,
        RuntimeRequest,
        RuntimeRequestInput,
    )


def untrusted(request: RuntimeRequestInput) -> RuntimeRequest:
    """Downgrade claims a tool caller cannot prove. Returns a validated request."""
    request = REQUEST_ADAPTER.validate_python(
        request.model_dump(mode="json") if isinstance(request, BaseModel) else request
    )
    if (
        isinstance(request, RememberRequest)
        and isinstance(request.memory, PreferenceMemory)
        and request.memory.explicit
    ):
        memory = request.memory.model_copy(update={"explicit": False})
        request = request.model_copy(update={"memory": memory})
    return request


class MemoryBackend(Protocol):
    """MemoryCore and MemoryService both satisfy this."""

    async def call(
        self, request: dict[str, JsonValue], *, owner_id: str
    ) -> BaseModel: ...


class MemoryTools[B: MemoryBackend]:
    def __init__(self, memory: B, *, owner_id: str) -> None:
        if not owner_id:
            raise ValueError("authenticated owner is required")
        self.memory: B = memory
        self.owner_id = owner_id

    @property
    def input_schema(self) -> JsonSchema:
        return REQUEST_ADAPTER.json_schema()

    @property
    def output_schema(self) -> JsonSchema:
        return RESPONSE_ADAPTER.json_schema()

    def input_schema_for(self, operation: str, schema: JsonSchema) -> JsonSchema:
        """Allow a scoped adapter to advertise stricter limits it also enforces."""
        return schema

    async def invoke(self, arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
        request = untrusted(arguments).model_dump(mode="json")
        response = await self.memory.call(request, owner_id=self.owner_id)
        return RESPONSE_ADAPTER.validate_python(
            response.model_dump(mode="json")
        ).model_dump(mode="json")
