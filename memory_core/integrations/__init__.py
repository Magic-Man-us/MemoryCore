"""Optional agent/transport integrations. Core imports never load vendor SDKs."""

from .session import (
    ContextUpdate,
    SessionContext,
    SessionContextRequest,
    SessionContextResponse,
)

__all__ = [
    "ContextUpdate",
    "SessionContext",
    "SessionContextRequest",
    "SessionContextResponse",
]
