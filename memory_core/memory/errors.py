"""Explicit failures at the memory boundary."""


class MemoryError(Exception):
    """Base memory service failure."""


class SchemaError(MemoryError):
    """Unknown field, incompatible type, operator, or schema version."""


class ConflictError(MemoryError):
    """The reconciliation snapshot or target revision is stale."""


class EmbeddingError(MemoryError):
    """Embedding identity, dimensions, or numeric values are invalid."""


class ContextRequiredError(MemoryError):
    """A task context receipt is missing, stale, or belongs to another task."""
