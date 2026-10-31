"""Bounded async worker. Model calls happen outside database transactions."""

from __future__ import annotations

import asyncio

from memory_core.memory import dream
from memory_core.memory.models import (
    AttachEvidence,
    Create,
    DreamPlan,
    DreamPlanRequest,
    ObservationMemory,
    ProjectScope,
    RememberRequest,
    UserScope,
)
from memory_core.memory.reconciliation import equivalent

from .dreams import advance, scan_for


class ReviewNeeded(Exception):
    def __init__(self, proposal):
        self.proposal = proposal
        super().__init__("semantic change requires review")


class Worker:
    def __init__(self, runtime):
        self.runtime = runtime
        self.queue = runtime.queue
        self.service = runtime.service
        self.policy = runtime.policy
        self._runner = None
        self._stop = asyncio.Event()
        self.last_error = None

    async def start(self):
        if self._runner is None:
            self._stop.clear()
            self._runner = asyncio.create_task(
                self._loop(), name="memorycore-maintenance"
            )

    async def stop(self):
        self._stop.set()
        if self._runner is not None:
            await self._runner  # current job is bounded by job_timeout_seconds
            self._runner = None

    async def _loop(self):
        while not self._stop.is_set():
            try:
                worked = await self.run_once()
                self.last_error = None
            except Exception as exc:  # noqa: BLE001 — retain durable failures from provider callbacks
                self.last_error = type(exc).__name__
                worked = False
            if not worked:
                try:
                    await asyncio.wait_for(self._stop.wait(), self.policy.poll_seconds)
                except TimeoutError:
                    pass

    async def run_once(self, *, owner_id=None):
        claim = await asyncio.to_thread(self.queue.claim, owner_id)
        if claim is None:
            return False
        try:
            async with asyncio.timeout(self.policy.job_timeout_seconds):
                if claim.kind == "capture":
                    await self._capture(claim)
                else:
                    await self._dream(claim)
        except ReviewNeeded as exc:
            await asyncio.to_thread(self._finish, claim, "proposed", exc.proposal)
        except asyncio.CancelledError as exc:
            await asyncio.to_thread(self.queue.fail, claim, exc)
            raise
        except Exception as exc:  # noqa: BLE001 — retain durable failures from provider callbacks
            await asyncio.to_thread(self.queue.fail, claim, exc)
        return True

    def _finish(self, claim, status, result):
        with self.service._lock, self.queue.transaction(claim.owner) as session:
            self.queue.finish(session, claim, status, result)
            if claim.kind == "dream" and status == "proposed":
                advance(
                    self.queue, session, claim, DreamPlan.model_validate(result["plan"])
                )

    async def _capture(self, claim):
        def load_task():
            with self.service._lock, self.queue.transaction(claim.owner) as session:
                return self.queue.task(session, claim.owner, claim.task_id)

        task = await asyncio.to_thread(load_task)
        payload = claim.payload
        finding = payload.get("finding")
        if finding:
            summary = finding["statement"]
            evidence = finding["evidence"]
        else:
            outcome = payload["outcome"]
            summary = f"Reported task {outcome['status']}: {outcome['summary']}"[:4096]
            evidence = outcome["evidence"]
        # Stable ID and timestamp make retries identical. Candidates stay observations;
        # a project discovery can never silently become an explicit user preference.
        memory = ObservationMemory(
            metadata=task.context.attributes,
            id=claim.id,
            summary=summary,
            observation=summary,
            scope=ProjectScope(project_id=task.context.project_id)
            if task.context.project_id
            else UserScope(),
            source_refs=list(dict.fromkeys([e["uri"] for e in evidence]))
            or [f"task:{claim.task_id}"],
            created_at=payload["captured_at"],
            updated_at=payload["captured_at"],
            observed_at=payload["captured_at"],
        )
        approval = (
            claim.result if claim.result and claim.result.get("approved") else None
        )
        request = RememberRequest(
            memory=memory,
            decision=approval["decision"] if approval else None,
            expected_generation=approval["generation"] if approval else None,
        )

        def commit(session, response):
            self.queue.assert_lease(session, claim)
            exact = isinstance(response.decision, AttachEvidence) and any(
                hit.memory.id == response.decision.target_id
                and equivalent(memory, hit.memory)
                for hit in response.neighbors
            )
            permitted = (
                isinstance(response.decision, Create)
                or (exact and self.policy.apply_exact_duplicates)
                or self.policy.apply_semantic_changes
                or approval is not None
            )
            if not permitted:
                raise ReviewNeeded(
                    {
                        "kind": "capture_proposal",
                        "incoming": memory.model_dump(mode="json"),
                        "neighbors": [
                            h.model_dump(mode="json") for h in response.neighbors
                        ],
                        "decision": response.decision.model_dump(mode="json"),
                        "generation": response.generation - 1,
                    }
                )
            self.queue.finish(
                session,
                claim,
                "done",
                {
                    "kind": "capture_applied",
                    "memory_id": response.memory.id,
                    "decision": response.decision.model_dump(mode="json"),
                },
            )

        await self.service.remember_reconciled(
            request, owner_id=claim.owner, _commit=commit
        )

    async def _dream(self, claim):
        scan = await asyncio.to_thread(scan_for, self.queue, claim)
        approval = (
            claim.result if claim.result and claim.result.get("approved") else None
        )
        if approval:
            plan_id = approval["plan_id"]
        else:
            planned = await dream.plan(
                self.service,
                claim.owner,
                DreamPlanRequest(batch_size=self.policy.dream_batch_size),
                scan=scan,
            )
            plan = planned.plan
            exact_only = all(
                isinstance(change.decision, AttachEvidence)
                and any(
                    g.memory.id == change.memory.id
                    and any(
                        h.memory.id == change.decision.target_id
                        and equivalent(change.memory, h.memory)
                        for h in g.neighbors
                    )
                    for g in plan.groups
                )
                for change in plan.proposal.changes
            )
            if plan.proposal.changes and not (
                exact_only
                and self.policy.apply_exact_duplicates
                or self.policy.apply_semantic_changes
            ):
                raise ReviewNeeded(
                    {
                        "kind": "dream_proposal",
                        "plan_id": plan.id,
                        "plan": plan.model_dump(mode="json"),
                        "generation": plan.generation,
                    }
                )
            plan_id = plan.id

        def commit(session, response):
            self.queue.finish(
                session,
                claim,
                "done",
                {
                    "kind": "dream_applied",
                    "plan_id": plan_id,
                    "applied": response.plan.applied,
                    "recommendations": response.plan.proposal.recommendations,
                },
            )
            advance(self.queue, session, claim, response.plan)

        await dream.apply(
            self.service, claim.owner, plan_id, on_commit=commit, _advance_cursor=False
        )
