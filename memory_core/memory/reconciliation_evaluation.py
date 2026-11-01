"""Offline labeled decision evaluation; never applies a decision to stored memories."""

from __future__ import annotations

import asyncio
import math
import statistics
import time
from typing import Annotated, Literal

from pydantic import Field, ValidationError, model_validator

from .interfaces import Reconciler
from .models import DECISION_ADAPTER, Memory, MemoryHit, Model, Nonempty, Reconciliation


class ReconciliationCase(Model):
    id: Nonempty
    memory: Memory
    neighbors: list[MemoryHit]
    acceptable: list[Reconciliation] = Field(min_length=1)
    rationale: Nonempty

    @model_validator(mode="after")
    def valid_labels(self):
        neighbors = {hit.memory.id: hit.memory for hit in self.neighbors}
        if len(neighbors) != len(self.neighbors) or self.memory.id in neighbors:
            raise ValueError(
                "case requires unique neighbor IDs distinct from incoming memory"
            )
        for decision in self.acceptable:
            if decision.action == "create":
                continue
            target = neighbors.get(decision.target_id)
            if target is None or target.revision != decision.expected_revision:
                raise ValueError(
                    "expected decision must reference a supplied target and revision"
                )
        return self


class DecisionResult(Model):
    status: Literal["correct", "incorrect"]
    case_id: str
    elapsed_ms: float
    decision: Reconciliation
    action_correct: bool


class FailureResult(Model):
    status: Literal["invalid", "timeout", "error"]
    case_id: str
    elapsed_ms: float
    error_type: str


CaseResult = Annotated[DecisionResult | FailureResult, Field(discriminator="status")]


class ReconciliationEvaluation(Model):
    cases: int
    correct: int
    incorrect: int
    invalid: int
    timeouts: int
    errors: int
    decision_accuracy: float
    action_accuracy: float
    p50_ms: float
    p95_ms: float
    results: list[CaseResult]


async def evaluate_reconciliation(
    reconciler: Reconciler,
    cases: list[ReconciliationCase],
    *,
    timeout_seconds: float = 30,
) -> ReconciliationEvaluation:
    """Score full decisions, including target and revision, against explicit labels.

    Runs sequentially to bound provider pressure. Callbacks receive copies and must
    be cancellation-cooperative; an asyncio timeout cannot stop blocking synchronous
    provider code. Provider errors are counted without retaining their message/body.
    Cancellation of the overall evaluation propagates to the caller.
    """
    if not cases or len({case.id for case in cases}) != len(cases):
        raise ValueError("evaluation requires nonempty cases with unique IDs")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be finite and positive")
    results = []
    for case in cases:
        started = time.perf_counter()
        try:
            async with asyncio.timeout(timeout_seconds):
                output = await reconciler(
                    case.memory.model_copy(deep=True),
                    [hit.model_copy(deep=True) for hit in case.neighbors],
                )
            if isinstance(output, str):
                decision = DECISION_ADAPTER.validate_json(output)
            else:
                decision = DECISION_ADAPTER.validate_python(
                    output.model_dump(mode="json")
                    if hasattr(output, "model_dump")
                    else output
                )
        except TimeoutError:
            status, error = "timeout", "TimeoutError"
        except ValidationError:
            status, error = "invalid", "ValidationError"
        except Exception as exc:  # noqa: BLE001 — provider failure is an evaluation outcome
            status, error = "error", type(exc).__name__
        else:
            expected = [label.model_dump(mode="json") for label in case.acceptable]
            results.append(
                DecisionResult(
                    status="correct"
                    if decision.model_dump(mode="json") in expected
                    else "incorrect",
                    case_id=case.id,
                    elapsed_ms=(time.perf_counter() - started) * 1000,
                    decision=decision,
                    action_correct=any(
                        label.action == decision.action for label in case.acceptable
                    ),
                )
            )
            continue
        results.append(
            FailureResult(
                status=status,
                case_id=case.id,
                elapsed_ms=(time.perf_counter() - started) * 1000,
                error_type=error,
            )
        )
    counts = {
        status: sum(result.status == status for result in results)
        for status in ("correct", "incorrect", "invalid", "timeout", "error")
    }
    timings = sorted(result.elapsed_ms for result in results)
    return ReconciliationEvaluation(
        cases=len(cases),
        correct=counts["correct"],
        incorrect=counts["incorrect"],
        invalid=counts["invalid"],
        timeouts=counts["timeout"],
        errors=counts["error"],
        decision_accuracy=counts["correct"] / len(cases),
        action_accuracy=sum(
            isinstance(result, DecisionResult) and result.action_correct
            for result in results
        )
        / len(cases),
        p50_ms=statistics.median(timings),
        p95_ms=timings[min(len(timings) - 1, int(0.95 * len(timings)))],
        results=results,
    )
