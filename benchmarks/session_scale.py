from __future__ import annotations

import argparse
import asyncio
import logging
import math
import os
import random
import statistics
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import Annotated

from pydantic import BaseModel, Field
from sqlalchemy import text

from memory_core import EmbeddingSpec
from memory_core.memory.models import (
    ObservationMemory,
    PreferenceMemory,
    ProjectScope,
    RecallRequest,
    TaskContext,
)
from memory_core.memory.service import MemoryService
from memory_core.storage import Database, DatabaseSettings
from memory_core.storage.codec import encode_embedding

PositiveCount = Annotated[int, Field(gt=0)]
Milliseconds = Annotated[float, Field(ge=0)]
Mebibytes = Annotated[float, Field(ge=0)]


class Options(BaseModel):
    count: PositiveCount = 10_000
    dimensions: PositiveCount = 512
    repetitions: PositiveCount = 100
    owner_count: PositiveCount = 64


class Timing(BaseModel):
    samples: int
    p50_ms: Milliseconds
    p95_ms: Milliseconds
    elapsed_ms: Milliseconds


class Cohort(BaseModel):
    sessions: int
    preference: Timing
    recall: Timing


class Residency(BaseModel):
    owners: int
    records_per_owner: int
    before_mib: Mebibytes
    hydrated_mib: Mebibytes
    cold: Timing
    revisit: Timing


class Report(BaseModel):
    options: Options
    process_id: int
    cold_10k_ms: Milliseconds
    preference: Timing
    filtered_recall: Timing
    concurrent: list[Cohort]
    residency: Residency
    notes: str = (
        "Synthetic precomputed vectors; no model or network calls. "
        "Concurrent latency includes executor and database queueing. "
        "RSS is process residency, not an exact per-cache allocation count. "
        "Concurrent cohorts retain all owners; residency uses the default 16-owner limit."
    )


def timing(samples: list[float], elapsed: float) -> Timing:
    return Timing(
        samples=len(samples),
        p50_ms=statistics.median(samples) * 1000,
        p95_ms=sorted(samples)[math.ceil(len(samples) * 0.95) - 1] * 1000,
        elapsed_ms=elapsed * 1000,
    )


def rss_mib() -> float:
    resident = int(Path("/proc/self/statm").read_text().split()[1])
    return resident * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)


def seed(
    service: MemoryService, owners: int, count: int, pool: list[list[float]]
) -> None:
    encoded = [encode_embedding(vector) for vector in pool]
    for owner_number in range(owners):
        owner = f"owner-{owner_number}"
        rows = []
        for index in range(count):
            project = "project" if index % 2 == 0 else "other-project"
            memory = ObservationMemory(
                id=f"m{index:06}",
                summary=f"{owner} retry policy {index}",
                observation=f"{owner} retries idempotent requests only {index}",
                scope=ProjectScope(project_id=project),
            )
            rows.append(
                {
                    "o": owner,
                    "id": memory.id,
                    "g": index + 1,
                    "p": memory.model_dump_json(),
                    "e": encoded[index % len(pool)],
                }
            )
        preference = PreferenceMemory(
            id="preference",
            summary=f"{owner} prefers concise responses",
            preference_key="response_style",
            value="concise",
            applicability={"languages": ["python"]},
        )
        rows.append(
            {
                "o": owner,
                "id": preference.id,
                "g": count + 1,
                "p": preference.model_dump_json(),
                "e": None,
            }
        )
        with service.store.transaction(owner) as session:
            session.execute(
                text("INSERT INTO memories_v1 VALUES (:o,:id,:g,:p,:e)"), rows
            )
            session.execute(
                text(
                    "UPDATE memory_owners SET generation=:g,embedding_spec=:s WHERE owner=:o"
                ),
                {"o": owner, "g": count + 1, "s": service.spec.model_dump_json()},
            )


async def measure(
    service: MemoryService, options: Options, *, sessions: int, preference: bool
) -> Timing:
    context = TaskContext(
        task_id="benchmark", project_id="project", attributes={"languages": ["python"]}
    )
    query = RecallRequest(
        query="retry policy",
        embedding=[1.0] * options.dimensions,
        project_id="project",
        limit=10,
    )
    samples: list[float] = []

    async def turn(owner: str) -> None:
        started = perf_counter()
        if preference:
            result = await service.resolve_prefs(context, owner_id=owner)
            assert result.complete and len(result.effective) == 1
        else:
            result = await service.recall(query, owner_id=owner)
            assert result.hits
            assert all(
                hit.memory.summary.startswith(owner + " ") for hit in result.hits
            )
            assert all(
                hit.memory.scope.kind != "project"
                or hit.memory.scope.project_id == "project"
                for hit in result.hits
            )
        samples.append(perf_counter() - started)

    started = perf_counter()
    for _ in range(math.ceil(options.repetitions / sessions)):
        async with asyncio.TaskGroup() as group:
            for number in range(sessions):
                group.create_task(turn(f"owner-{number}"))
    return timing(samples, perf_counter() - started)


def make_service(
    *, sqlite_path: Path, embedding_spec: EmbeddingSpec, max_cached_owners: int = 16
) -> MemoryService:
    return MemoryService(
        Database(settings=DatabaseSettings(sqlite_path=str(sqlite_path))),
        embedding_spec=embedding_spec,
        max_cached_owners=max_cached_owners,
    )


async def benchmark(options: Options) -> Report:
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=64))
    rng = random.Random(42)
    pool = [[rng.uniform(-1, 1) for _ in range(options.dimensions)] for _ in range(32)]
    spec = EmbeddingSpec(
        model_id="synthetic", revision="1", dimensions=options.dimensions
    )
    with tempfile.TemporaryDirectory() as directory:
        systems = []
        try:
            large = make_service(
                sqlite_path=Path(directory) / "large.db", embedding_spec=spec
            )
            systems.append(large)
            seed(large, 1, options.count, pool)
            cold = await measure(
                large,
                options.model_copy(update={"repetitions": 1}),
                sessions=1,
                preference=False,
            )
            await measure(
                large,
                options.model_copy(update={"repetitions": 10}),
                sessions=1,
                preference=True,
            )
            prefs = await measure(large, options, sessions=1, preference=True)
            recall = await measure(large, options, sessions=1, preference=False)
            shared = make_service(
                sqlite_path=Path(directory) / "shared.db",
                embedding_spec=spec,
                max_cached_owners=options.owner_count,
            )
            systems.append(shared)
            seed(shared, options.owner_count, 100, pool)
            await measure(
                shared,
                options.model_copy(update={"repetitions": 1}),
                sessions=options.owner_count,
                preference=True,
            )
            cohorts = []
            for count in (1, 16, 64):
                if count <= options.owner_count:
                    cohorts.append(
                        Cohort(
                            sessions=count,
                            preference=await measure(
                                shared, options, sessions=count, preference=True
                            ),
                            recall=await measure(
                                shared,
                                options,
                                sessions=count,
                                preference=False,
                            ),
                        )
                    )
            churn = make_service(
                sqlite_path=Path(directory) / "churn.db", embedding_spec=spec
            )
            systems.append(churn)
            seed(churn, 17, 1000, pool)
            before = rss_mib()
            cold_samples, revisit_samples = [], []
            context = TaskContext(task_id="cache", attributes={"languages": ["python"]})
            for samples in (cold_samples, revisit_samples):
                for number in range(17):
                    started = perf_counter()
                    await churn.resolve_prefs(context, owner_id=f"owner-{number}")
                    samples.append(perf_counter() - started)
            return Report(
                options=options,
                process_id=os.getpid(),
                cold_10k_ms=cold.p50_ms,
                preference=prefs,
                filtered_recall=recall,
                concurrent=cohorts,
                residency=Residency(
                    owners=17,
                    records_per_owner=1001,
                    before_mib=before,
                    hydrated_mib=rss_mib(),
                    cold=timing(cold_samples, sum(cold_samples)),
                    revisit=timing(revisit_samples, sum(revisit_samples)),
                ),
            )
        finally:
            for system in systems:
                system.store.db.dispose()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10_000)
    parser.add_argument("--dimensions", type=int, default=512)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--owner-count", type=int, default=64)
    options = Options.model_validate(vars(parser.parse_args()))
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger(__name__).info(
        asyncio.run(benchmark(options)).model_dump_json(indent=2)
    )


if __name__ == "__main__":
    main()
