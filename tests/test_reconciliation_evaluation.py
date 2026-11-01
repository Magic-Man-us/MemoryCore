import asyncio
import json
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from memory_core.memory.models import Create
from memory_core.memory.reconciliation_evaluation import (
    ReconciliationCase,
    evaluate_reconciliation,
)


def cases():
    fixture = Path(__file__).parent / "fixtures/reconciliation.json"
    return TypeAdapter(list[ReconciliationCase]).validate_python(
        json.loads(fixture.read_text())["cases"]
    )


async def test_full_decisions_include_target_and_revision():
    sample = cases()[-1]

    async def wrong_target(memory, neighbors):
        return {
            "action": "attach_evidence",
            "target_id": "distractor",
            "expected_revision": 1,
        }

    report = await evaluate_reconciliation(wrong_target, [sample])
    assert report.action_accuracy == 1 and report.decision_accuracy == 0
    assert report.incorrect == 1

    async def wrong_revision(memory, neighbors):
        return {
            "action": "attach_evidence",
            "target_id": "stored",
            "expected_revision": 1,
        }

    assert (await evaluate_reconciliation(wrong_revision, [sample])).incorrect == 1


async def test_fixture_oracle_scores_labels_and_callbacks_cannot_mutate_cases():
    samples = cases()
    original = [case.model_dump(mode="json") for case in samples]
    decisions = iter(case.acceptable[0] for case in samples)

    async def oracle(memory, neighbors):
        memory.source_refs.append("mutation")
        neighbors[0].memory.source_refs.append("mutation")
        return next(decisions).model_dump_json()

    report = await evaluate_reconciliation(oracle, samples)
    assert report.correct == len(samples)
    assert report.decision_accuracy == report.action_accuracy == 1
    assert [case.model_dump(mode="json") for case in samples] == original


async def test_failures_count_in_denominator_without_error_body():
    samples = cases()[:4]
    calls = 0

    async def provider(memory, neighbors):
        nonlocal calls
        calls += 1
        if calls == 1:
            return Create()
        if calls == 2:
            return {"action": "delete"}
        if calls == 3:
            raise RuntimeError("sensitive provider response")
        await asyncio.sleep(10)

    report = await evaluate_reconciliation(provider, samples, timeout_seconds=0.01)
    assert (report.correct, report.invalid, report.errors, report.timeouts) == (
        1,
        1,
        1,
        1,
    )
    assert report.decision_accuracy == 0.25
    assert "sensitive" not in report.model_dump_json()


async def test_cancellation_propagates():
    async def cancelled(memory, neighbors):
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await evaluate_reconciliation(cancelled, cases()[:1])


async def test_invalid_labels_and_evaluation_configuration_rejected():
    sample = cases()[-1].model_dump(mode="json")
    sample["acceptable"][0]["expected_revision"] = 99
    with pytest.raises(ValidationError):
        ReconciliationCase.model_validate(sample)

    async def provider(memory, neighbors):
        return Create()

    for timeout in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            await evaluate_reconciliation(provider, cases(), timeout_seconds=timeout)
    with pytest.raises(ValueError):
        await evaluate_reconciliation(provider, [])
    with pytest.raises(ValueError):
        await evaluate_reconciliation(provider, [cases()[0], cases()[0]])
