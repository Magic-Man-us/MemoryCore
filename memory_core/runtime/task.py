"""SDK-neutral task hooks. Completion is explicit, never implied by context-manager exit."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Self

from .models import (
    BeginTaskRequest,
    CompleteTaskRequest,
    PrepareTaskRequest,
    RecordFindingRequest,
    RequireKnowledgeRequest,
    TaskRunResult,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable
    from types import TracebackType

    from pydantic import JsonValue

    from memory_core.memory.models import Nonempty, TaskContext

    from .core import MemoryCore
    from .models import (
        CaptureResponse,
        Finding,
        KnowledgeGap,
        KnowledgeResponse,
        PreparedTask,
        TaskOutcome,
    )


class TaskSession:
    def __init__(
        self, runtime: MemoryCore, owner: str, context: TaskContext, objective: Nonempty
    ) -> None:
        self.runtime, self.owner = runtime, owner
        self.context, self.objective = context, objective
        self.prepared = None

    async def __aenter__(self) -> Self:
        response = await self.runtime.call(
            BeginTaskRequest(context=self.context, objective=self.objective),
            owner_id=self.owner,
        )
        self.context = response.context
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Literal[False]:
        # An exception/exit does not prove a task outcome. Hosts can explicitly
        # report failed/cancelled outcomes; unfinished tasks remain resumable.
        return False

    async def prepare(self) -> PreparedTask:
        self.prepared = await self.runtime.call(
            PrepareTaskRequest(task_id=self.context.task_id),
            owner_id=self.owner,
        )
        return self.prepared

    async def require_knowledge(self, gap: KnowledgeGap) -> KnowledgeResponse:
        return await self.runtime.call(
            RequireKnowledgeRequest(task_id=self.context.task_id, gap=gap),
            owner_id=self.owner,
        )

    async def record_finding(
        self, finding: Finding, *, event_id: Nonempty
    ) -> CaptureResponse:
        return await self.runtime.call(
            RecordFindingRequest(
                task_id=self.context.task_id, event_id=event_id, finding=finding
            ),
            owner_id=self.owner,
        )

    async def complete(
        self, outcome: TaskOutcome, *, findings: Iterable[Finding] = ()
    ) -> CaptureResponse:
        return await self.runtime.call(
            CompleteTaskRequest(
                task_id=self.context.task_id, outcome=outcome, findings=list(findings)
            ),
            owner_id=self.owner,
        )

    async def before_action(self) -> PreparedTask:
        """Refresh context, then refuse conflicts or unresolved declared gaps."""
        self.prepared = await self.runtime.call(
            PrepareTaskRequest(task_id=self.context.task_id, guard=True),
            owner_id=self.owner,
        )
        return self.prepared

    async def execute[ResultT](
        self, action: Callable[[PreparedTask], Awaitable[ResultT]]
    ) -> ResultT:
        """Guard one tool/action without declaring the entire task complete."""
        return await action(await self.before_action())

    async def run(
        self,
        action: Callable[
            [PreparedTask], Awaitable[TaskRunResult | dict[str, JsonValue]]
        ],
    ) -> TaskRunResult:
        """Action receives context; its typed result triggers durable outcome capture."""
        prepared = await self.before_action()
        result = TaskRunResult.model_validate(await action(prepared))
        await self.complete(result.outcome, findings=result.findings)
        return result
