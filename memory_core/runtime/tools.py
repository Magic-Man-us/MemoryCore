"""Transport-neutral tool adapter; authenticated identity never comes from tool arguments."""

from .models import REQUEST_ADAPTER, RESPONSE_ADAPTER


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
        response = await self.memory.call(arguments, owner_id=self.owner_id)
        return RESPONSE_ADAPTER.validate_python(
            response.model_dump(mode="json")
        ).model_dump(mode="json")
