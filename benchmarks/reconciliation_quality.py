"""Evaluate a configured reasoning callback against explicit, reviewable labels."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
from pathlib import Path

from pydantic import TypeAdapter

from memory_core.memory.models import Create
from memory_core.memory.reconciliation_evaluation import (
    ReconciliationCase,
    evaluate_reconciliation,
)
from memory_core.runtime.reasoning import StructuredReasoner


async def run(args):
    fixture = json.loads(Path(args.fixture).read_text())
    cases = TypeAdapter(list[ReconciliationCase]).validate_python(fixture["cases"])
    if args.provider:
        module, attribute = args.provider.split(":", 1)
        # Explicit trusted host callback. Importing arbitrary modules executes code.
        generate = getattr(importlib.import_module(module), attribute)
        reconciler = StructuredReasoner(generate).reconcile
    else:

        async def reconciler(memory, neighbors):
            return Create()

    report = await evaluate_reconciliation(
        reconciler, cases, timeout_seconds=args.timeout
    )
    return {
        "provider": args.provider or "always-create baseline (no AI)",
        "fixture": fixture["description"],
        "evaluation": report.model_dump(mode="json"),
        "interpretation": "Curated smoke labels, not independent model validation. Decisions were scored but never applied. Correct action alone is insufficient: target ID and revision must also match.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider", help="Trusted module:async_generate accepting a ReasoningRequest"
    )
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument(
        "--fixture",
        default=str(
            Path(__file__).resolve().parents[1] / "tests/fixtures/reconciliation.json"
        ),
    )
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(args)), indent=2))
