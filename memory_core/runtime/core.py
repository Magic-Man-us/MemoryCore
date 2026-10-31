"""Provider-neutral task memory facade; Python and tools share this dispatcher."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime

from sqlalchemy import text

from memory_core.memory.errors import ConflictError, ContextRequiredError, SchemaError
from memory_core.memory.metadata import Registry
from memory_core.memory.models import DescribeSchemaRequest, SchemaResponse, TaskContext

from .knowledge import is_current
from .models import (
    KNOWLEDGE_ADAPTER,
    REQUEST_ADAPTER,
    BeginTaskRequest,
    CaptureResponse,
    CompleteTaskRequest,
    ContextContribution,
    ContextInput,
    MaintenancePolicy,
    MaintenanceResponse,
    MaintenanceStatusRequest,
    PreparedTask,
    PrepareTaskRequest,
    RecordFindingRequest,
    RequireKnowledgeRequest,
    ResolvedKnowledge,
    RetryJobRequest,
    RetryResponse,
    ReviewJobRequest,
    ReviewResponse,
    UnresolvedKnowledge,
)
from .queue import Queue, canonical
from .worker import Worker


class MemoryCore:
    """Owns lifecycle orchestration around a MemoryService; never infers owner identity."""

    def __init__(
        self,
        service,
        *,
        maintenance=None,
        researcher=None,
        knowledge_evaluator=None,
        context_resolvers=(),
    ):
        self.service = service
        self.policy = MaintenancePolicy.model_validate(
            maintenance or MaintenancePolicy()
        )
        if self.policy.lease_seconds <= self.policy.job_timeout_seconds:
            raise ValueError("lease_seconds must exceed job_timeout_seconds")
        if service._commit_observer is not None:
            raise ValueError("a service can have only one lifecycle runtime")
        self.researcher = researcher
        self.knowledge_evaluator = knowledge_evaluator
        self.context_resolvers = tuple(context_resolvers)
        self.queue = Queue(service, self.policy)
        service._commit_observer = self.queue.schedule_dream
        self.worker = Worker(self)

    async def __aenter__(self):
        if self.policy.background:
            await self.worker.start()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.worker.stop()

    async def measurements(self, *, owner_id):
        """Sample owner counts and whole-database disk usage without loading memories."""
        from .metrics import measure

        return await asyncio.to_thread(measure, self, owner_id)

    def task(self, *, owner_id, context, objective):
        from .task import TaskSession

        return TaskSession(self, owner_id, context, objective)

    async def call(self, request, *, owner_id, _transaction_guard=None):
        if not owner_id:
            raise ValueError("authenticated owner must be nonempty")
        request = REQUEST_ADAPTER.validate_python(
            request.model_dump(mode="json")
            if hasattr(request, "model_dump")
            else request
        )
        if isinstance(request, BeginTaskRequest):
            return await self._begin(request, owner_id)
        if isinstance(request, PrepareTaskRequest):
            return await self._prepare_task(request, owner_id)
        if isinstance(request, RequireKnowledgeRequest):
            from .knowledge import resolve

            return await resolve(
                self, request, owner_id, transaction_guard=_transaction_guard
            )
        if isinstance(request, (RecordFindingRequest, CompleteTaskRequest)):
            return await asyncio.to_thread(
                self._record, request, owner_id, transaction_guard=_transaction_guard
            )
        if isinstance(request, MaintenanceStatusRequest):
            return await asyncio.to_thread(self._status, request, owner_id)
        if isinstance(request, RetryJobRequest):
            return await asyncio.to_thread(self._retry, request, owner_id)
        if isinstance(request, ReviewJobRequest):
            return await asyncio.to_thread(self._review, request, owner_id)
        response = await self.service.call(request, owner_id=owner_id)
        if isinstance(response, SchemaResponse):
            return response.model_copy(
                update={"request_schema": REQUEST_ADAPTER.json_schema()}
            )
        return response

    async def _begin(self, request, owner):
        def existing():
            with self.service._lock, self.queue.transaction(owner) as session:
                row = session.execute(
                    text(
                        "SELECT objective,context FROM memory_tasks WHERE owner=:o AND id=:t"
                    ),
                    {"o": owner, "t": request.context.task_id},
                ).first()
                if row:
                    saved = self.queue.task(session, owner, request.context.task_id)
                    # Resume accepts omitted inferred values, but never changed explicit inputs.
                    if (
                        saved.objective != request.objective
                        or saved.context.project_id != request.context.project_id
                        or any(
                            saved.context.attributes.get(k) != v
                            for k, v in request.context.attributes.items()
                        )
                    ):
                        raise ConflictError(
                            "task ID reused with different context/objective"
                        )
                    return saved
            return None

        saved = await asyncio.to_thread(existing)
        if saved is not None:
            return saved
        schema = await self.service.call(DescribeSchemaRequest(), owner_id=owner)
        registry = Registry(schema.fields)
        attributes, sources = {}, {}
        for resolver in self.context_resolvers:
            async with asyncio.timeout(self.policy.provider_timeout_seconds):
                contribution = ContextContribution.model_validate(
                    await resolver(
                        ContextInput(
                            context=request.context.model_copy(deep=True),
                            objective=request.objective,
                            fields=[f.model_copy(deep=True) for f in schema.fields],
                        )
                    )
                )
            normalized = registry.normalize(contribution.attributes)
            explicit = registry.normalize(request.context.attributes)
            for name, value in normalized.items():
                if name in explicit:
                    continue
                if name in attributes and attributes[name] != value:
                    raise SchemaError(f"context resolvers disagree on {name}")
                attributes[name], sources[name] = value, contribution.source
        explicit = registry.normalize(request.context.attributes)
        attributes.update(explicit)
        sources.update({name: "caller" for name in explicit})
        context = TaskContext(
            task_id=request.context.task_id,
            project_id=request.context.project_id,
            attributes=attributes,
        )

        def save():
            with self.service._lock, self.queue.transaction(owner) as session:
                session.execute(
                    text("""INSERT OR IGNORE INTO memory_tasks
                    (owner,id,context,objective,sources) VALUES (:o,:t,:c,:q,:s)"""),
                    {
                        "o": owner,
                        "t": context.task_id,
                        "c": context.model_dump_json(),
                        "q": request.objective,
                        "s": json.dumps(sources),
                    },
                )
                saved = self.queue.task(session, owner, context.task_id)
                if saved.context != context or saved.objective != request.objective:
                    raise ConflictError(
                        "concurrent task initialization changed context"
                    )
                return saved

        return await asyncio.to_thread(save)

    def _record(self, request, owner, *, transaction_guard=None):
        with self.service._lock, self.queue.transaction(owner) as session:
            if transaction_guard is not None:
                transaction_guard(session)
            self.queue.task(session, owner, request.task_id)
            if isinstance(request, RecordFindingRequest):
                return CaptureResponse(
                    job_ids=[
                        self._capture_finding(
                            session,
                            owner,
                            request.task_id,
                            "finding:" + request.event_id,
                            request.finding,
                        )
                    ]
                )
            serialized = canonical(request)
            old = session.execute(
                text("SELECT completion FROM memory_tasks WHERE owner=:o AND id=:t"),
                {"o": owner, "t": request.task_id},
            ).scalar_one()
            if old is not None:
                if old != serialized:
                    raise ConflictError("task completion is immutable")
                ids = (
                    session.execute(
                        text("""SELECT job_id FROM memory_events
                    WHERE owner=:o AND task_id=:t AND (id='outcome' OR id LIKE 'completion:%') ORDER BY id"""),
                        {"o": owner, "t": request.task_id},
                    )
                    .scalars()
                    .all()
                )
                return CaptureResponse(job_ids=list(ids))
            if request.outcome.status == "succeeded":
                self.assert_gaps(session, owner, request.task_id)
            jobs = [
                self._capture_finding(
                    session, owner, request.task_id, f"completion:{i}", finding
                )
                for i, finding in enumerate(request.findings)
            ]
            jobs.append(
                self.queue.capture(
                    session,
                    owner,
                    request.task_id,
                    "outcome",
                    {
                        "outcome": request.outcome.model_dump(mode="json"),
                        "captured_at": datetime.now(UTC).isoformat(),
                    },
                )
            )
            session.execute(
                text("UPDATE memory_tasks SET completion=:c WHERE owner=:o AND id=:t"),
                {"o": owner, "t": request.task_id, "c": serialized},
            )
            return CaptureResponse(job_ids=jobs)

    def _capture_finding(self, session, owner, task_id, event_id, finding):
        # Use the existing timestamp on replay; event content, not retry time, determines identity.
        old = session.execute(
            text(
                "SELECT payload FROM memory_events WHERE owner=:o AND task_id=:t AND id=:id"
            ),
            {"o": owner, "t": task_id, "id": event_id},
        ).scalar_one_or_none()
        captured_at = (
            json.loads(old)["captured_at"] if old else datetime.now(UTC).isoformat()
        )
        return self.queue.capture(
            session,
            owner,
            task_id,
            event_id,
            {
                "finding": finding.model_dump(mode="json"),
                "captured_at": captured_at,
            },
        )

    @staticmethod
    def assert_gaps(session, owner, task_id):

        for row in session.execute(
            text("SELECT gap,result FROM memory_gaps WHERE owner=:o AND task_id=:t"),
            {"o": owner, "t": task_id},
        ).mappings():
            if not is_current(session, owner, row["gap"], row["result"]):
                raise ContextRequiredError(
                    "task has unresolved or expired knowledge gaps"
                )

    def _status(self, request, owner):
        with self.service._lock, self.queue.transaction(owner) as session:
            condition = " AND status=:s" if request.status else ""
            rows = (
                session.execute(
                    text(
                        "SELECT * FROM memory_jobs WHERE owner=:o"
                        + condition
                        + " ORDER BY id LIMIT :n OFFSET :offset"
                    ),
                    {
                        "o": owner,
                        "s": request.status,
                        "n": request.limit + 1,
                        "offset": request.offset,
                    },
                )
                .mappings()
                .all()
            )
            return MaintenanceResponse(
                jobs=[self.queue.view(r) for r in rows[: request.limit]],
                next_offset=request.offset + request.limit
                if len(rows) > request.limit
                else None,
            )

    def _review(self, request, owner):
        with self.service._lock, self.queue.transaction(owner) as session:
            row = (
                session.execute(
                    text("SELECT * FROM memory_jobs WHERE owner=:o AND id=:id"),
                    {"o": owner, "id": request.job_id},
                )
                .mappings()
                .one_or_none()
            )
            if row is None or row["status"] != "proposed":
                raise ConflictError("job is not awaiting review")
            result = json.loads(row["result"])
            if (
                request.accept
                and result["generation"]
                != self.service.store.state(session, owner)["generation"]
            ):
                raise ConflictError("proposal is stale; retry the job to replan")
            result["approved"] = request.accept
            session.execute(
                text("""UPDATE memory_jobs SET status=:s,result=:r,attempts=0,available=0
                WHERE owner=:o AND id=:id"""),
                {
                    "o": owner,
                    "id": request.job_id,
                    "s": "pending" if request.accept else "rejected",
                    "r": json.dumps(result),
                },
            )
            row = (
                session.execute(
                    text("SELECT * FROM memory_jobs WHERE owner=:o AND id=:id"),
                    {"o": owner, "id": request.job_id},
                )
                .mappings()
                .one()
            )
            return ReviewResponse(job=self.queue.view(row))

    def _retry(self, request, owner):
        with self.service._lock, self.queue.transaction(owner) as session:
            changed = session.execute(
                text("""UPDATE memory_jobs SET status='pending',
                attempts=0,available=0,result=NULL,error=NULL,token=NULL,lease_until=NULL
                WHERE owner=:o AND id=:id AND status IN ('failed','proposed','rejected')"""),
                {"o": owner, "id": request.job_id},
            ).rowcount
            if not changed:
                raise ConflictError(
                    "only failed, proposed, or rejected jobs may be retried"
                )
            row = (
                session.execute(
                    text("SELECT * FROM memory_jobs WHERE owner=:o AND id=:id"),
                    {"o": owner, "id": request.job_id},
                )
                .mappings()
                .one()
            )
            return RetryResponse(job=self.queue.view(row))

    async def prepare_task_context(self, context, *, owner_id, query=None, limit=5):
        from memory_core.memory.models import PrepareContextRequest

        return await self.call(
            PrepareContextRequest(context=context, query=query, limit=limit),
            owner_id=owner_id,
        )

    @classmethod
    def local(
        cls,
        path,
        *,
        embedding_spec,
        embedder,
        reconciler=None,
        dream_planner=None,
        max_cached_owners=16,
        **kwargs,
    ):
        """Create a local runtime with optional reasoning/research providers."""
        from memory_core.config import build_memory_system

        service = build_memory_system(
            sqlite_path=path,
            embedding_spec=embedding_spec,
            embedder=embedder,
            reconciler=reconciler,
            dream_planner=dream_planner,
            max_cached_owners=max_cached_owners,
        ).service
        return cls(service, **kwargs)

    async def _prepare_task(self, request, owner, *, cached=None):
        def load():
            with self.service._lock, self.queue.transaction(owner) as session:
                return self.queue.task(session, owner, request.task_id)

        task = await asyncio.to_thread(load)
        response = None
        if (
            cached is not None
            and cached.context == task.context
            and cached.objective == task.objective
        ):
            try:
                await self.service.validate_context(
                    cached.receipt, task.context, owner_id=owner
                )
                response = cached
            except ContextRequiredError:
                pass
        if response is None:
            response = await self.service.prepare_task_context(
                task.context, owner_id=owner, query=task.objective
            )

        def collect():

            with self.service._lock, self.queue.transaction(owner) as session:
                current = self.queue.task(session, owner, request.task_id)
                if request.guard:
                    if current.outcome is not None:
                        raise ConflictError("task is already complete")
                    if response.preferences.conflicts:
                        raise ContextRequiredError(
                            "resolve preference conflicts before acting"
                        )
                    self.assert_gaps(session, owner, request.task_id)
                knowledge = {}
                for row in session.execute(
                    text(
                        "SELECT id,gap,result FROM memory_gaps WHERE owner=:o AND task_id=:t"
                    ),
                    {"o": owner, "t": request.task_id},
                ).mappings():
                    result = (
                        KNOWLEDGE_ADAPTER.validate_json(row["result"])
                        if row["result"]
                        else UnresolvedKnowledge(reason="Research is pending")
                    )
                    if isinstance(result, ResolvedKnowledge) and not is_current(
                        session, owner, row["gap"], row["result"]
                    ):
                        result = UnresolvedKnowledge(
                            reason="Evidence expired; resolve the gap again"
                        )
                    knowledge[row["id"]] = result
                return knowledge

        knowledge = await asyncio.to_thread(collect)
        if request.guard:
            await self.service.validate_context(
                response.receipt, task.context, owner_id=owner
            )
        return PreparedTask(
            **response.model_dump(
                exclude={"operation", "context", "objective", "knowledge"}
            ),
            context=task.context,
            objective=task.objective,
            knowledge=knowledge,
        )
