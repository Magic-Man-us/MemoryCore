"""Narrow SDK/testing boundaries; implementations must cooperate with cancellation."""

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Protocol

from claude_agent_sdk import ClaudeAgentOptions, Message
from claude_agent_sdk.types import HookInput

from memory_core.runtime.models import Evidence


class AgentClient(Protocol):
    async def connect(self) -> None: ...
    async def query(self, prompt: str) -> None: ...
    def receive_response(self) -> AsyncIterator[Message]: ...
    async def disconnect(self) -> None: ...


class ClientFactory(Protocol):
    def __call__(self, *, options: ClaudeAgentOptions) -> AgentClient: ...


class EvidenceExtractor(Protocol):
    async def __call__(self, event: HookInput) -> list[Evidence]: ...


MessageObserver = Callable[[Message], Awaitable[None]]
