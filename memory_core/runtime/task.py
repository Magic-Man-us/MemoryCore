"""SDK-neutral task hooks. Completion is explicit, never implied by context-manager exit."""

from __future__ import annotations

from .models import (
    BeginTaskRequest,
    CompleteTaskRequest,
    PrepareTaskRequest,
    RecordFindingRequest,
    RequireKnowledgeRequest,
    TaskRunResult,
)


class TaskSession:
    def __init__(self, runtime, owner, context, objective):
        self.runtime, self.owner = runtime, owner
        self.context, self.objective = context, objective
        self.prepared = None

    async def __aenter__(self):
        response = await self.runtime.call(
            BeginTaskRequest(context=self.context, objective=self.objective),
            owner_id=self.owner,
        )
        self.context = response.context
        return self

    async def __aexit__(self, exc_type, exc, tb):
        # An exception/exit does not prove a task outcome. Hosts can explicitly
        # report failed/cancelled outcomes; unfinished tasks remain resumable.
        return False

    async def prepare(self):
        self.prepared = await self.runtime.call(
            PrepareTaskRequest(task_id=self.context.task_id),
            owner_id=self.owner,
        )
        return self.prepared

    async def require_knowledge(self, gap):
        return await self.runtime.call(
            RequireKnowledgeRequest(task_id=self.context.task_id, gap=gap),
            owner_id=self.owner,
        )

    async def record_finding(self, finding, *, event_id):
        return await self.runtime.call(
            RecordFindingRequest(
                task_id=self.context.task_id, event_id=event_id, finding=finding
            ),
            owner_id=self.owner,
        )

    async def complete(self, outcome, *, findings=()):
        return await self.runtime.call(
            CompleteTaskRequest(
                task_id=self.context.task_id, outcome=outcome, findings=list(findings)
            ),
            owner_id=self.owner,
        )

    async def before_action(self):
        """Refresh context, then refuse conflicts or unresolved declared gaps."""
        self.prepared = await self.runtime.call(
            PrepareTaskRequest(task_id=self.context.task_id, guard=True),
            owner_id=self.owner,
        )
        return self.prepared

    async def execute(self, action):
        """Guard one tool/action without declaring the entire task complete."""
        return await action(await self.before_action())

    async def run(self, action):
        """Action receives context; its typed result triggers durable outcome capture."""
        prepared = await self.before_action()
        result = TaskRunResult.model_validate(await action(prepared))
        await self.complete(result.outcome, findings=result.findings)
        return result
