"""Local growth/restart benchmark with labeled synthetic retrieval cases.

uv run python benchmarks/memory_lifecycle.py --count 50000 --dimensions 384
Run each corpus size in a fresh process for meaningful peak RSS. The seed corpus
is bulk-loaded as a previously maintained database; incremental writes use public
reconciliation and the real maintenance worker. No model/network calls are made.
This measures mechanics, not natural-language embedding or reasoning quality.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import random
import resource
import statistics
import tempfile
import time
from pathlib import Path

from sqlalchemy import text

from memory_core import EmbeddingSpec, MemoryCore
from memory_core.memory.evaluation import RetrievalCase, evaluate_retrieval
from memory_core.memory.models import (
    ObservationMemory,
    RecallRequest,
    RememberRequest,
    Supersede,
)
from memory_core.runtime import MaintenancePolicy
from memory_core.storage.codec import encode_embedding


class SyntheticEmbedder:
    def __init__(self, dimensions):
        self.dimensions = dimensions

    async def embed(self, texts):
        return [self.vector(value) for value in texts]

    def vector(self, value):
        rng = random.Random(value)
        return [rng.uniform(-1, 1) for _ in range(self.dimensions)]


def percentile(samples, fraction):
    return (
        sorted(samples)[min(len(samples) - 1, int(len(samples) * fraction))]
        if samples
        else None
    )


def latency(samples):
    return {
        "samples": len(samples),
        "p50_ms": statistics.median(samples) if samples else None,
        "p95_ms": percentile(samples, 0.95),
    }


def resident_mib():
    # Linux current RSS; unlike ru_maxrss this can show memory released after a phase.
    try:
        values = Path("/proc/self/status").read_text().splitlines()
        return next(
            int(line.split()[1]) / 1024 for line in values if line.startswith("VmRSS:")
        )
    except (OSError, StopIteration):
        return None


async def run(count, dimensions, rounds, burst):
    embedder = SyntheticEmbedder(dimensions)
    spec = EmbeddingSpec(
        model_id="synthetic-lifecycle", revision="1", dimensions=dimensions
    )
    policy = MaintenancePolicy(
        background=False,
        dream_after_writes=burst,
        job_timeout_seconds=300,
        lease_seconds=360,
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "lifecycle.db"

        def open_runtime():
            return MemoryCore.local(
                path, embedding_spec=spec, embedder=embedder, maintenance=policy
            )

        memory = open_runtime()
        cases = [
            RetrievalCase(
                query=f"Unique reference anchor {i}", relevant_ids={f"m{i:08}"}
            )
            for i in range(8)
        ]
        pool = [
            encode_embedding(embedder.vector(f"seed-vector-{i}")) for i in range(512)
        ]
        started = time.perf_counter()
        with memory.queue.transaction("benchmark") as session:
            for start in range(0, count, 256):
                batch = []
                for i in range(start, min(count, start + 256)):
                    summary = (
                        cases[i].query
                        if i < len(cases)
                        else f"Observation {i} about topic {i % 100}"
                    )
                    record = ObservationMemory(
                        id=f"m{i:08}",
                        summary=summary,
                        observation=summary,
                        metadata={"topics": [f"topic_{i % 100}"]},
                    )
                    batch.append(
                        {
                            "o": "benchmark",
                            "id": record.id,
                            "g": i + 1,
                            "p": record.model_dump_json(),
                            "e": encode_embedding(embedder.vector(summary))
                            if i < 8
                            else pool[i % len(pool)],
                        }
                    )
                session.execute(
                    text("INSERT INTO memories_v1 VALUES (:o,:id,:g,:p,:e)"), batch
                )
            session.execute(
                text("""UPDATE memory_owners SET generation=:n,writes=:n,dream_writes=:n,
                                 dream_cursor=:n,dream_cursor_id=:id WHERE owner=:o"""),
                {"n": count, "id": f"m{count - 1:08}", "o": "benchmark"},
            )
        seed_seconds = time.perf_counter() - started
        snapshots = []

        async def snapshot(label):
            snapshots.append(
                {
                    "phase": label,
                    "rss_mib": resident_mib(),
                    **(await memory.measurements(owner_id="benchmark")).model_dump(),
                }
            )

        await snapshot("seeded")
        started = time.perf_counter()
        await memory.call(RecallRequest(query=cases[0].query), owner_id="benchmark")
        cold_seconds = time.perf_counter() - started
        baseline = await evaluate_retrieval(
            memory.service, cases, owner_id="benchmark", k=5
        )
        await snapshot("hydrated")
        cycles = []
        for cycle in range(rounds):
            writes = []
            first = None
            last = None
            for i in range(burst):
                # Repeated evidence for the same structured observation exercises exact reconciliation.
                record = (
                    first.model_copy(update={"id": f"repeat-{cycle}-{i}"})
                    if i % 4 == 3
                    else ObservationMemory(
                        id=f"new-{cycle}-{i}",
                        summary=f"New cycle {cycle} observation {i}",
                        observation=f"Cycle fact {cycle}:{i}",
                    )
                )
                started = time.perf_counter()
                result = await memory.call(
                    RememberRequest(memory=record), owner_id="benchmark"
                )
                writes.append((time.perf_counter() - started) * 1000)
                first = first or record
                last = result
            # Explicitly retire one fact; historical data must remain available after restart.
            await memory.call(
                RememberRequest(
                    memory=ObservationMemory(
                        id=f"replacement-{cycle}",
                        summary=f"Revised cycle {cycle}",
                        observation="Revised fact",
                    ),
                    decision=Supersede(
                        target_id=last.memory.id, expected_revision=last.memory.revision
                    ),
                    expected_generation=last.generation,
                ),
                owner_id="benchmark",
            )
            await snapshot(f"cycle-{cycle}-queued")
            cache = memory.service._caches["benchmark"]

            async def drain(current=memory):
                batches = 0
                while await current.worker.run_once():
                    batches += 1
                    if batches > burst * 4:
                        raise AssertionError("maintenance did not converge")
                return batches

            started = time.perf_counter()
            draining = asyncio.create_task(drain())
            reads = []
            while not draining.done():
                probe_start = time.perf_counter()
                await memory.call(
                    RecallRequest(query=cases[0].query), owner_id="benchmark"
                )
                reads.append((time.perf_counter() - probe_start) * 1000)
                await asyncio.sleep(0.01)
            batches = await draining
            elapsed = time.perf_counter() - started
            assert memory.service._caches["benchmark"] is cache, (
                "dream caused a cold reload"
            )
            measured = await memory.measurements(owner_id="benchmark")
            assert not any(
                measured.jobs_by_status.get(s, 0)
                for s in ("pending", "running", "failed")
            )
            quality = await evaluate_retrieval(
                memory.service, cases, owner_id="benchmark", k=5
            )
            assert quality.mean_recall_at_k == baseline.mean_recall_at_k == 1
            cycles.append(
                {
                    "write_latency": latency(writes),
                    "recall_during_maintenance": latency(reads),
                    "maintenance_seconds": elapsed,
                    "maintenance_batches": batches,
                    "synthetic_quality": quality.model_dump(),
                }
            )
            await snapshot(f"cycle-{cycle}-drained")
            memory.service.store.db.engine.dispose()
            del cache, drain, draining
            memory = open_runtime()
            gc.collect()
            started = time.perf_counter()
            restored = await evaluate_retrieval(
                memory.service, cases, owner_id="benchmark", k=5
            )
            assert restored.mean_recall_at_k == 1
            cycles[-1]["restart_and_evaluation_seconds"] = time.perf_counter() - started
            await snapshot(f"cycle-{cycle}-restarted")
        memory.service.store.db.engine.dispose()
        return {
            "seed_count": count,
            "dimensions": dimensions,
            "seed_seconds": seed_seconds,
            "cold_recall_seconds": cold_seconds,
            "baseline_synthetic_quality": baseline.model_dump(),
            "cycles": cycles,
            "snapshots": snapshots,
            "peak_rss_mib_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024,
            "limitations": "Synthetic labels/vectors; seed bypasses ingestion. Measures incremental public writes and real maintenance. Disk metrics cover the whole database. No real-model semantic quality claim.",
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--dimensions", type=int, default=384)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--burst", type=int, default=32)
    args = parser.parse_args()
    if args.count < 8 or args.dimensions < 2 or args.rounds < 1 or args.burst < 4:
        parser.error("count>=8, dimensions>=2, rounds>=1, burst>=4 required")
    print(
        json.dumps(
            asyncio.run(run(args.count, args.dimensions, args.rounds, args.burst)),
            indent=2,
        )
    )
