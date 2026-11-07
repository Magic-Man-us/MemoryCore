"""Reproducible local end-to-end benchmark. Synthetic vectors measure mechanics, not semantic quality.

Run: uv run python benchmarks/memory_scale.py --count 50000 --dimensions 384
Build the extension with `maturin develop --release`; a debug build is several times slower.
No network, hosted model, or paid infrastructure. Fixture loading bypasses reconciliation
so the benchmark can build a large corpus without 50,000 model calls.

Every run reports wall time, process CPU, worker-thread CPU, and thread counts per query,
plus MemoryCore.measurements for the owner. --profile PATH adds a cProfile pass (stats
written to PATH, top functions in the report) and the tracemalloc peak for that pass.
"""

from __future__ import annotations

import argparse
import asyncio
import cProfile
import json
import pstats
import random
import resource
import statistics
import tempfile
import threading
import time
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import text

from memory_core import Database, EmbeddingSpec, MemoryCore, MemoryService
from memory_core.memory.models import (
    ContainsAny,
    ObservationMemory,
    PreferenceMemory,
    RecallRequest,
    TaskContext,
)
from memory_core.runtime import MaintenancePolicy
from memory_core.storage.codec import encode_embedding

TOP_FUNCTIONS = 15
BYTES_PER_MIB = 1024 * 1024
_to_thread = asyncio.to_thread


@dataclass(slots=True)
class WorkerUsage:
    """CPU time spent inside the service's asyncio.to_thread calls and the threads used."""

    cpu_seconds: float = 0.0
    threads: set[int] = field(default_factory=set)
    peak_active_threads: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def reset(self) -> None:
        with self.lock:
            self.cpu_seconds = 0.0
            self.threads = set()
            self.peak_active_threads = 0


def instrument_worker_threads(usage: WorkerUsage, *, inline: bool) -> None:
    """Route the service's asyncio.to_thread calls through a CPU-time recorder.

    With ``inline`` the calls run on the event-loop thread: cProfile records only the
    thread it was enabled on, so the profiled pass needs the service's work there.
    """

    async def to_thread(func, /, *args, **kwargs):
        def measured():
            started = time.thread_time()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.thread_time() - started
                with usage.lock:
                    usage.cpu_seconds += elapsed
                    usage.threads.add(threading.get_ident())
                    usage.peak_active_threads = max(
                        usage.peak_active_threads, threading.active_count()
                    )

        if inline:
            return measured()
        return await _to_thread(measured)

    asyncio.to_thread = to_thread


async def profile_recall(service, request, repetitions, path):
    usage = WorkerUsage()
    instrument_worker_threads(usage, inline=True)
    tracemalloc.start()
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        for _ in range(repetitions):
            await service.call(request, owner_id="benchmark")
    finally:
        profiler.disable()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        instrument_worker_threads(usage, inline=False)
    profiler.dump_stats(path)
    # get_stats_profile() keys entries by bare function name, so distinct same-named
    # functions (e.g. two lambdas) collide, and it rounds times to 1 ms. Stats.stats
    # keeps (file, line, name) identities and unrounded timings.
    entries = pstats.Stats(profiler).stats
    hottest = sorted(entries.items(), key=lambda item: -item[1][2])
    return {
        "stats_file": str(path),
        "queries": repetitions,
        "python_peak_alloc_mib": round(peak / BYTES_PER_MIB, 1),
        "top_functions": [
            {
                "function": f"{file}:{line}({name})",
                "calls": calls if primitive == calls else f"{calls}/{primitive}",
                "self_ms": round(self_seconds * 1000, 3),
                "cumulative_ms": round(cumulative_seconds * 1000, 3),
            }
            for (file, line, name), (
                primitive,
                calls,
                self_seconds,
                cumulative_seconds,
                _callers,
            ) in hottest[:TOP_FUNCTIONS]
        ],
    }


async def main(count, dimensions, repetitions, profile_path=None):
    usage = WorkerUsage()
    instrument_worker_threads(usage, inline=False)
    with tempfile.TemporaryDirectory() as directory:
        service = MemoryService(
            Database(Path(directory) / "bench.db"),
            embedding_spec=EmbeddingSpec(
                model_id="synthetic", revision="1", dimensions=dimensions
            ),
        )
        rng = random.Random(42)
        # Reused random vector pool keeps fixture generation cheap and reproducible.
        pool = [[rng.uniform(-1, 1) for _ in range(dimensions)] for _ in range(256)]
        encoded = [encode_embedding(v) for v in pool]
        started = time.perf_counter()
        with service.store.transaction("benchmark") as session:
            rows = []
            for i in range(count):
                memory = ObservationMemory(
                    id=f"m{i:06}",
                    summary=f"Engineering observation number {i}",
                    observation=f"Observation {i}",
                    metadata={"topics": [f"topic_{i % 100}"]},
                )
                rows.append(
                    {
                        "o": "benchmark",
                        "id": memory.id,
                        "g": i + 1,
                        "p": memory.model_dump_json(),
                        "e": encoded[i % 256],
                    }
                )
            session.execute(
                text("INSERT INTO memories_v1 VALUES (:o,:id,:g,:p,:e)"), rows
            )
            session.execute(
                text("UPDATE memory_owners SET generation=:g WHERE owner=:o"),
                {"g": count, "o": "benchmark"},
            )
        load = time.perf_counter() - started
        request = RecallRequest(
            query="engineering observations", embedding=pool[0], limit=5
        )
        started = time.perf_counter()
        first = await service.call(request, owner_id="benchmark")
        hydrate = time.perf_counter() - started
        timings = {}
        for label, req in [
            ("all", request),
            (
                "filtered",
                request.model_copy(
                    update={"filter": ContainsAny(field="topics", values=["topic_0"])}
                ),
            ),
        ]:
            samples = []
            usage.reset()
            cpu_started = time.process_time()
            for _ in range(repetitions):
                started = time.perf_counter()
                result = await service.call(req, owner_id="benchmark")
                samples.append((time.perf_counter() - started) * 1000)
                assert len(result.hits) <= 5
                if label == "filtered":
                    assert all(
                        h.memory.metadata["topics"] == ["topic_0"] for h in result.hits
                    )
            ordered = sorted(samples)
            timings[label] = {
                "p50_ms": round(statistics.median(samples), 3),
                "p95_ms": round(
                    ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], 3
                ),
                "process_cpu_ms_per_query": round(
                    (time.process_time() - cpu_started) * 1000 / repetitions, 3
                ),
                "worker_cpu_ms_per_query": round(
                    usage.cpu_seconds * 1000 / repetitions, 3
                ),
                "worker_threads": len(usage.threads),
                "peak_active_threads": usage.peak_active_threads,
            }
        # Mandatory rules are retrieved independently of query similarity or top-k.
        from memory_core.memory.models import RememberRequest

        for i in range(12):
            await service.call(
                RememberRequest(
                    memory=PreferenceMemory(
                        summary=f"Coding rule {i}",
                        preference_key=f"rule_{i}",
                        value=i,
                        applicability={"languages": ["python"]},
                    ),
                    embedding=pool[i],
                ),
                owner_id="benchmark",
            )
        prefs = await service.resolve_prefs(
            TaskContext(task_id="benchmark", attributes={"languages": ["python"]}),
            owner_id="benchmark",
        )
        assert len(prefs.effective) == 12 and prefs.complete
        # Constructed after the writes: the runtime attaches a dream scheduler to commits.
        runtime = MemoryCore(service, maintenance=MaintenancePolicy(background=False))
        measurements = await runtime.measurements(owner_id="benchmark")
        report = {
            "count": count,
            "dimensions": dimensions,
            "repetitions": repetitions,
            "fixture_load_seconds": round(load, 3),
            "cold_recall_including_hydration_seconds": round(hydrate, 3),
            "warm_recall": timings,
            "max_rss_mib_linux": round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1
            ),
            "returned_to_agent": len(first.hits),
            "preference_rules_retrieved": len(prefs.effective),
            "measurements": measurements.model_dump(mode="json"),
            "note": "Synthetic CPU benchmark; not a real-embedding semantic recall evaluation. RSS includes fixture generation.",
        }
        if profile_path is not None:
            report["profile"] = await profile_recall(
                service, request, repetitions, profile_path
            )
        print(json.dumps(report, indent=2))
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=50000)
    parser.add_argument("--dimensions", type=int, default=384)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--profile", type=Path, help="write cProfile stats here")
    args = parser.parse_args()
    asyncio.run(main(args.count, args.dimensions, args.repetitions, args.profile))
