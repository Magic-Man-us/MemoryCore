"""Typed local memory operations. Ownership is supplied by trusted application code."""

from __future__ import annotations

import asyncio
import math
import secrets
import threading
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, overload
from weakref import WeakValueDictionary

from pydantic import BaseModel, JsonValue
from sqlalchemy import text

from memory_core.indexing.semantic import HnswIndex, IndexBackend, IndexPolicy
from memory_core.storage.database import Database

from .cache import OwnerCache
from .catalog import catalog
from .errors import ConflictError, ContextRequiredError, EmbeddingError, SchemaError
from .interfaces import DreamPlanner, Embedder, Reconciler
from .metadata import Registry
from .models import (
    DECISION_ADAPTER,
    MEMORY_ADAPTER,
    REQUEST_ADAPTER,
    AttachEvidence,
    ContextResponse,
    Create,
    DreamApplyRequest,
    DreamPlanRequest,
    DreamResponse,
    EmbeddingSpec,
    MemoryHit,
    PreferenceMemory,
    PreferencesResponse,
    PrepareContextRequest,
    RecallRequest,
    RecallResponse,
    RememberRequest,
    RememberResponse,
    Request,
    ResolvePreferencesRequest,
    Response,
    RetrievalPolicy,
    Status,
    TaskContext,
)
from .preferences import resolve_preferences
from .reconciliation import apply_reconciliation, equivalent
from .store import Store

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.engine import RowMapping
    from sqlalchemy.orm import Session

    from .models import (
        DescribeSchemaRequest,
        FieldResponse,
        FieldsResponse,
        ListFieldsRequest,
        ListValuesRequest,
        Memory,
        RegisterFieldRequest,
        SchemaResponse,
        ValuesResponse,
    )


class MemoryService:
    def __init__(
        self,
        db: Database,
        *,
        embedding_spec: EmbeddingSpec,
        embedder: Embedder | None = None,
        reconciler: Reconciler | None = None,
        dream_planner: DreamPlanner | None = None,
        dream_after_writes: int = 100,
        retrieval_policy: RetrievalPolicy | None = None,
        max_cached_owners: int = 16,
        index_policy: IndexPolicy | dict[str, JsonValue] | None = None,
    ) -> None:
        if dream_after_writes < 1:
            raise ValueError("dream_after_writes must be positive")
        if type(max_cached_owners) is not int or max_cached_owners < 1:
            raise ValueError("max_cached_owners must be a positive integer")
        self.index_policy = IndexPolicy.model_validate(index_policy or IndexPolicy())
        self.max_cached_owners = max_cached_owners
        self.retrieval_policy = retrieval_policy or RetrievalPolicy()
        self.store = Store(db)
        self.spec = embedding_spec
        self.embedder = embedder
        self.reconciler = reconciler
        self.dream_planner = dream_planner
        self.dream_after_writes = dream_after_writes
        self._caches: OrderedDict[str, OwnerCache] = OrderedDict()
        self._cache_lock = threading.RLock()
        self._owner_locks: WeakValueDictionary[str, threading.RLock] = (
            WeakValueDictionary()
        )
        self._receipts: dict[
            tuple[str, str], tuple[str, str, int, datetime | None]
        ] = {}
        self._commit_observer: Callable[[Session, str], None] | None = None

    def _owner_lock(self, owner: str) -> threading.RLock:
        with self._cache_lock:
            lock = self._owner_locks.get(owner)
            if lock is None:
                lock = threading.RLock()
                self._owner_locks[owner] = lock
            return lock

    def _snapshot(
        self, session: Session, owner: str
    ) -> tuple[OwnerCache, Registry, RowMapping]:
        state = self.store.state(session, owner)
        identity = self.spec.model_dump_json()
        if state["embedding_spec"] is None:
            session.execute(
                text("UPDATE memory_owners SET embedding_spec=:s WHERE owner=:o"),
                {"o": owner, "s": identity},
            )
        elif state["embedding_spec"] != identity:
            raise EmbeddingError(
                "embedding model/revision/dimension differs; explicit re-embedding required"
            )
        with self._cache_lock:
            cache = self._caches.get(owner)
        if cache is None:
            cache = OwnerCache(
                self.spec.dimensions,
                self.retrieval_policy,
                index_policy=self.index_policy,
            )
            if isinstance(cache.vectors, HnswIndex):
                saved = session.execute(
                    text(
                        "SELECT identity,generation,manifest,graph FROM memory_vector_snapshots WHERE owner=:o"
                    ),
                    {"o": owner},
                ).first()
                if saved and saved[0] == identity and saved[1] <= state["generation"]:
                    try:
                        cache.vectors.restore(saved[2], saved[3])
                        cache.restored_generation = saved[1]
                    except (ValueError, RuntimeError, KeyError, TypeError):
                        cache = OwnerCache(
                            self.spec.dimensions,
                            self.retrieval_policy,
                            index_policy=self.index_policy,
                        )
        try:
            cache.sync(session, owner, state["generation"])
            if cache.restored_generation >= 0 and isinstance(cache.vectors, HnswIndex):
                for uid in cache.vectors.active - cache.memories.keys():
                    cache.vectors.remove(uid)
                cache.restored_generation = -1
            if cache.registry_generation != state["generation"]:
                cache.registry = Registry(self.store.fields(session, owner))
                cache.registry_generation = state["generation"]
        except BaseException:
            with self._cache_lock:
                if self._caches.get(owner) is cache:
                    self._caches.pop(owner, None)
            raise
        evicted = []
        with self._cache_lock:
            self._caches[owner] = cache
            self._caches.move_to_end(owner)
            while len(self._caches) > self.max_cached_owners:
                evicted.append(self._caches.popitem(last=False))
        return cache, cache.registry, state

    async def checkpoint_indexes(self, *, force: bool = False) -> int:
        def save() -> int:
            if self.index_policy.backend != IndexBackend.HNSW:
                return 0
            saved = 0
            with self._cache_lock:
                owners = tuple(self._caches)
            for owner in owners:
                with self._owner_lock(owner):
                    with self._cache_lock:
                        cache = self._caches.get(owner)
                    if cache is None or not isinstance(cache.vectors, HnswIndex):
                        continue
                    if (
                        not force
                        and cache.vectors.updates
                        < self.index_policy.checkpoint_after_updates
                    ):
                        continue
                    with self.store.read_transaction(
                        owner, embedding_spec=self.spec.model_dump_json()
                    ) as session:
                        cache, _, state = self._snapshot(session, owner)
                        generation = state["generation"]
                    vectors = cache.vectors
                    if not isinstance(vectors, HnswIndex):
                        continue
                    manifest, graph = vectors.snapshot()
                    with self.store.transaction(owner) as session:
                        session.execute(
                            text("""INSERT INTO memory_vector_snapshots VALUES (:o,:i,:g,:m,:b)
                            ON CONFLICT(owner) DO UPDATE SET identity=excluded.identity,generation=excluded.generation,
                            manifest=excluded.manifest,graph=excluded.graph"""),
                            {
                                "o": owner,
                                "i": self.spec.model_dump_json(),
                                "g": generation,
                                "m": manifest,
                                "b": graph,
                            },
                        )
                    vectors.updates = 0
                    saved += 1
            return saved

        return await asyncio.to_thread(save)

    async def _vector(
        self, supplied: Sequence[float] | None, value: str
    ) -> list[float]:
        if supplied is None:
            if self.embedder is None:
                raise EmbeddingError(
                    "supply an embedding or configure a local Embedder"
                )
            vectors = await self.embedder.embed([value])
            if len(vectors) != 1:
                raise EmbeddingError("embedder must return one vector per input")
            supplied = vectors[0]
        vector = list(supplied)
        if len(vector) != self.spec.dimensions or any(
            not math.isfinite(v) for v in vector
        ):
            raise EmbeddingError("embedding dimension mismatch or non-finite values")
        if not any(vector):
            raise EmbeddingError("zero embedding is not searchable")
        # Validate float32 representability at the same boundary as the Rust engine.
        if any(abs(v) > 3.402823466e38 for v in vector):
            raise EmbeddingError("embedding exceeds float32 range")
        norm = math.hypot(*vector)
        return [v / norm for v in vector]

    @overload
    async def call(
        self, request: RememberRequest, *, owner_id: str
    ) -> RememberResponse: ...
    @overload
    async def call(
        self, request: RecallRequest, *, owner_id: str
    ) -> RecallResponse: ...
    @overload
    async def call(
        self, request: ResolvePreferencesRequest, *, owner_id: str
    ) -> PreferencesResponse: ...
    @overload
    async def call(
        self, request: PrepareContextRequest, *, owner_id: str
    ) -> ContextResponse: ...
    @overload
    async def call(
        self, request: DescribeSchemaRequest, *, owner_id: str
    ) -> SchemaResponse: ...
    @overload
    async def call(
        self, request: ListFieldsRequest, *, owner_id: str
    ) -> FieldsResponse: ...
    @overload
    async def call(
        self, request: RegisterFieldRequest, *, owner_id: str
    ) -> FieldResponse: ...
    @overload
    async def call(
        self, request: ListValuesRequest, *, owner_id: str
    ) -> ValuesResponse: ...
    @overload
    async def call(
        self, request: DreamPlanRequest | DreamApplyRequest, *, owner_id: str
    ) -> DreamResponse: ...
    @overload
    async def call(
        self, request: dict[str, JsonValue], *, owner_id: str
    ) -> Response: ...
    async def call(
        self, request: Request | dict[str, JsonValue], *, owner_id: str
    ) -> Response:
        # Revalidate Python instances too: model_copy updates can bypass constructors.
        request = REQUEST_ADAPTER.validate_python(
            request.model_dump(mode="json")
            if isinstance(request, BaseModel)
            else request
        )
        if isinstance(request, RememberRequest):
            return await self.remember_reconciled(request, owner_id=owner_id)
        if isinstance(request, RecallRequest):
            return await self.recall(request, owner_id=owner_id)
        if isinstance(request, ResolvePreferencesRequest):
            return await self.resolve_prefs(request.context, owner_id=owner_id)
        if isinstance(request, PrepareContextRequest):
            return await self.prepare_task_context(
                request.context,
                owner_id=owner_id,
                query=request.query,
                limit=request.limit,
            )
        if isinstance(request, DreamPlanRequest):
            return await self.dream(owner_id=owner_id, request=request)
        if isinstance(request, DreamApplyRequest):
            return await self.apply_dream(request.plan_id, owner_id=owner_id)
        return await asyncio.to_thread(catalog, self, request, owner_id)

    @staticmethod
    def _normalize(memory: Memory, registry: Registry) -> Memory:
        data = memory.model_dump()
        data["metadata"] = registry.normalize(memory.metadata)
        if isinstance(memory, PreferenceMemory):
            applicability = {}
            for name, values in memory.applicability.items():
                if not values:
                    raise SchemaError("applicability alternatives must be nonempty")
                field = registry.resolve(name)
                if field.name in applicability:
                    raise SchemaError("duplicate applicability alias")
                registry.check_value(
                    field, values if field.type == "keywords" else values[0]
                )
                for value in values:
                    registry.check_value(
                        field, [value] if field.type == "keywords" else value
                    )
                applicability[field.name] = values
            data["applicability"] = applicability
        return MEMORY_ADAPTER.validate_python(data)

    async def recall(self, request: RecallRequest, *, owner_id: str) -> RecallResponse:
        vector = await self._vector(request.embedding, request.query)

        def run() -> RecallResponse:
            with (
                self._owner_lock(owner_id),
                self.store.read_transaction(
                    owner_id, embedding_spec=self.spec.model_dump_json()
                ) as session,
            ):
                cache, registry, state = self._snapshot(session, owner_id)
                if Status.SUPERSEDED in request.statuses:
                    from .history import recall_history

                    return RecallResponse(
                        hits=recall_history(
                            self, session, owner_id, request, vector, registry
                        ),
                        generation=state["generation"],
                    )
                ids = cache.eligible(request.project_id, active=False) & set.union(
                    set(), *(cache.by_status[status] for status in request.statuses)
                )
                ids -= cache.outside_window(request.as_of or datetime.now(UTC))
                if request.filter is not None:
                    ids = cache.metadata.select(request.filter, registry, ids)
                hits = [
                    MemoryHit(
                        memory=cache.memories[uid].model_copy(deep=True), score=score
                    )
                    for uid, score in cache.search(
                        request.query, vector, ids, request.limit
                    )
                ]
                return RecallResponse(hits=hits, generation=state["generation"])

        return await asyncio.to_thread(run)

    async def resolve_prefs(
        self, context: TaskContext, *, owner_id: str
    ) -> PreferencesResponse:
        def run() -> PreferencesResponse:
            with (
                self._owner_lock(owner_id),
                self.store.read_transaction(
                    owner_id, embedding_spec=self.spec.model_dump_json()
                ) as session,
            ):
                cache, registry, state = self._snapshot(session, owner_id)
                return resolve_preferences(
                    cache, registry, state["generation"], context, datetime.now(UTC)
                )

        return await asyncio.to_thread(run)

    async def remember_reconciled(
        self,
        request: RememberRequest,
        *,
        owner_id: str,
        _commit: Callable[[Session, RememberResponse], None] | None = None,
    ) -> RememberResponse:
        vector = await self._vector(request.embedding, request.memory.summary)

        def inspect() -> tuple[Memory, list[MemoryHit], int]:
            with (
                self._owner_lock(owner_id),
                self.store.read_transaction(
                    owner_id, embedding_spec=self.spec.model_dump_json()
                ) as session,
            ):
                cache, registry, state = self._snapshot(session, owner_id)
                memory = self._normalize(request.memory, registry)
                if (
                    memory.status != Status.ACTIVE
                    or memory.revision != 1
                    or memory.supersedes_id
                ):
                    raise SchemaError(
                        "new records must be active at revision 1; use reconciliation for changes"
                    )
                if self.store.contains(session, owner_id, memory.id):
                    raise ConflictError("memory id already exists")
                ids = cache.eligible(memory.scope.project_id)
                neighbors = [
                    MemoryHit(
                        memory=cache.memories[uid].model_copy(deep=True), score=score
                    )
                    for uid, score in cache.search(memory.summary, vector, ids, 20)
                ]
                exact = cache.equivalents.get(cache.key(memory), set()) & ids
                if exact:
                    uid = min(exact)
                    if all(hit.memory.id != uid for hit in neighbors):
                        # Exact structured matches must remain available even when
                        # semantic scores are zero or the top-k pool is crowded.
                        neighbors.append(
                            MemoryHit(
                                memory=cache.memories[uid].model_copy(deep=True),
                                score=0.0,
                            )
                        )
                return memory, neighbors, state["generation"]

        memory, neighbors, generation = await asyncio.to_thread(inspect)
        decision = request.decision
        if decision is not None and request.expected_generation != generation:
            raise ConflictError(
                "explicit reconciliation requires the current generation"
            )
        if decision is None and self.reconciler is not None:
            decision = DECISION_ADAPTER.validate_python(
                await self.reconciler(
                    memory.model_copy(deep=True),
                    [h.model_copy(deep=True) for h in neighbors],
                )
            )
        if decision is None:
            decision = Create()
            # Only exact structured equivalence is merged automatically.
            for hit in neighbors:
                if equivalent(memory, hit.memory):
                    decision = AttachEvidence(
                        target_id=hit.memory.id, expected_revision=hit.memory.revision
                    )
                    break

        def persist() -> RememberResponse:
            with (
                self._owner_lock(owner_id),
                self.store.transaction(owner_id) as session,
            ):
                cache, _registry, state = self._snapshot(session, owner_id)
                if state["generation"] != generation:
                    raise ConflictError(
                        "memory changed during reconciliation; retrieve and retry"
                    )
                allowed = {h.memory.id for h in neighbors}
                # Explicit target operations may address an exact fact outside top-k.
                if request.decision is not None:
                    allowed = cache.eligible(memory.scope.project_id)
                stored, new_generation = apply_reconciliation(
                    self, session, owner_id, cache, memory, vector, decision, allowed
                )
                response = RememberResponse(
                    memory=stored,
                    decision=decision,
                    neighbors=neighbors,
                    generation=new_generation,
                )

                if self._commit_observer is not None:
                    self._commit_observer(session, owner_id)
                if _commit is not None:
                    _commit(session, response)
                return response

        result = await asyncio.to_thread(persist)
        if self.dream_after_writes and self._commit_observer is None:
            try:
                if await self.dream_due(owner_id):
                    planned = await self.dream(owner_id=owner_id)
                    result = result.model_copy(
                        update={"dream_plan_id": planned.plan.id}
                    )
            except Exception as exc:  # noqa: BLE001 — maintenance must not misreport a committed write
                # The write already committed. Report maintenance failure separately
                # so callers do not retry a successful write as though it failed.
                result = result.model_copy(
                    update={"maintenance_error": type(exc).__name__}
                )
        return result

    async def prepare_task_context(
        self,
        context: TaskContext,
        *,
        owner_id: str,
        query: str | None = None,
        limit: int = 5,
    ) -> ContextResponse:
        vector = await self._vector(None, query) if query else None

        def run() -> ContextResponse:
            with (
                self._owner_lock(owner_id),
                self.store.read_transaction(
                    owner_id, embedding_spec=self.spec.model_dump_json()
                ) as session,
            ):
                cache, registry, state = self._snapshot(session, owner_id)
                now = datetime.now(UTC)
                preferences = resolve_preferences(
                    cache, registry, state["generation"], context, now
                )
                supporting = (
                    []
                    if query is None or vector is None
                    else [
                        MemoryHit(
                            memory=cache.memories[uid].model_copy(deep=True),
                            score=score,
                        )
                        for uid, score in cache.search(
                            query,
                            vector,
                            cache.eligible(context.project_id)
                            - cache.outside_window(now),
                            limit,
                        )
                    ]
                )
                receipt = secrets.token_urlsafe(24)
                # One current receipt per owner/task; bounded retention per service.
                with self._cache_lock:
                    # The context goes stale at the next validity bound as well as
                    # on the next write.
                    self._receipts[(owner_id, context.task_id)] = (
                        receipt,
                        context.model_dump_json(),
                        state["generation"],
                        cache.next_boundary(now),
                    )
                    if len(self._receipts) > 1024:
                        del self._receipts[next(iter(self._receipts))]
                return ContextResponse(
                    preferences=preferences, supporting=supporting, receipt=receipt
                )

        return await asyncio.to_thread(run)

    refresh_task_context = prepare_task_context

    async def validate_context(
        self, receipt: str, context: TaskContext, *, owner_id: str
    ) -> None:
        def run() -> None:
            with (
                self._owner_lock(owner_id),
                self.store.read_transaction(
                    owner_id, embedding_spec=self.spec.model_dump_json()
                ) as session,
            ):
                expected = (
                    receipt,
                    context.model_dump_json(),
                    self.store.state(session, owner_id)["generation"],
                )
                with self._cache_lock:
                    stored = self._receipts.get((owner_id, context.task_id))
                matches = (
                    stored is not None
                    and stored[:3] == expected
                    and (stored[3] is None or datetime.now(UTC) < stored[3])
                )
                if not matches:
                    raise ContextRequiredError(
                        "missing/stale context: prepare_task_context before editing"
                    )

        await asyncio.to_thread(run)

    async def run_with_context[T](
        self,
        context: TaskContext,
        action: Callable[[ContextResponse], Awaitable[T]],
        *,
        owner_id: str,
    ) -> T:
        """SDK-independent hook: caller receives context before invoking its coding action.

        Editing tools should additionally call validate_context immediately before mutation.
        This library cannot intercept tools in an unrelated agent runtime.
        """
        prepared = await self.prepare_task_context(context, owner_id=owner_id)
        return await action(prepared)

    async def dream_due(self, owner_id: str) -> bool:
        from .dream import due

        return await asyncio.to_thread(due, self, owner_id)

    async def dream(
        self, *, owner_id: str, request: DreamPlanRequest | None = None
    ) -> DreamResponse:
        from .dream import plan

        return await plan(self, owner_id, request or DreamPlanRequest())

    async def apply_dream(self, plan_id: str, *, owner_id: str) -> DreamResponse:
        from .dream import apply

        return await apply(self, owner_id, plan_id)
