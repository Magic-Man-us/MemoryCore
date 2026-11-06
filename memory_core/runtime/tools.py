"""Transport-neutral tool adapter; authenticated identity never comes from tool arguments.

Tool arguments are model-authored. A model can report what it inferred, but it cannot
assert that the user stated a preference: ``explicit`` is trusted provenance, so it is
set by host code calling ``MemoryCore.call`` directly, never by a tool call.
"""

from memory_core.memory.models import PreferenceMemory, RememberRequest

from .models import REQUEST_ADAPTER, RESPONSE_ADAPTER


def untrusted(request):
    """Downgrade claims a tool caller cannot prove. Returns a validated request."""
    request = REQUEST_ADAPTER.validate_python(
        request.model_dump(mode="json") if hasattr(request, "model_dump") else request
    )
    if (
        isinstance(request, RememberRequest)
        and isinstance(request.memory, PreferenceMemory)
        and request.memory.explicit
    ):
        memory = request.memory.model_copy(update={"explicit": False})
        request = request.model_copy(update={"memory": memory})
    return request


class MemoryTools:
    def __init__(self, memory, *, owner_id):
        if not owner_id:
            raise ValueError("authenticated owner is required")
        self.memory, self.owner_id = memory, owner_id

    @property
    def input_schema(self):
        return REQUEST_ADAPTER.json_schema()

    @property
    def output_schema(self):
        return RESPONSE_ADAPTER.json_schema()

    def input_schema_for(self, operation, schema):
        """Allow a scoped adapter to advertise stricter limits it also enforces."""
        return schema

    async def invoke(self, arguments):
        response = await self.memory.call(untrusted(arguments), owner_id=self.owner_id)
        return RESPONSE_ADAPTER.validate_python(
            response.model_dump(mode="json")
        ).model_dump(mode="json")
