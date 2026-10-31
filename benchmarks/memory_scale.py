"""Reproducible local end-to-end benchmark. Synthetic vectors measure mechanics, not semantic quality.

Run: uv run python benchmarks/memory_scale.py --count 50000 --dimensions 384
No network, hosted model, or paid infrastructure. Fixture loading bypasses reconciliation
so the benchmark can build a large corpus without 50,000 model calls.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import resource
import statistics
import tempfile
import time
from pathlib import Path

from sqlalchemy import text

from memory_core import EmbeddingSpec, build_memory_system
from memory_core.memory.models import (
    ContainsAny,
    ObservationMemory,
    PreferenceMemory,
    RecallRequest,
    TaskContext,
)
from memory_core.storage.codec import encode_embedding


async def main(count, dimensions, repetitions):
    with tempfile.TemporaryDirectory() as directory:
        system = build_memory_system(
            sqlite_path=Path(directory) / "bench.db",
            embedding_spec=EmbeddingSpec(
                model_id="synthetic", revision="1", dimensions=dimensions
            ),
        )
        service = system.service
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
            "note": "Synthetic CPU benchmark; not a real-embedding semantic recall evaluation. RSS includes fixture generation.",
        }
        print(json.dumps(report, indent=2))
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=50000)
    parser.add_argument("--dimensions", type=int, default=384)
    parser.add_argument("--repetitions", type=int, default=30)
    args = parser.parse_args()
    asyncio.run(main(args.count, args.dimensions, args.repetitions))
