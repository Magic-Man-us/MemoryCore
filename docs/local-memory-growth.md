# Growing a local memory collection

MemoryCore keeps its database, retrieval indexes, and maintenance queue on the local
machine. Embedding and reasoning providers are configured separately; using the
Claude integration still sends requests to Claude.

## Automatic maintenance

A successful write can enqueue a dream after `dream_after_writes` writes. The first
worker claim saves a fixed generation boundary in the durable job. Each batch visits
at most `dream_batch_size` records up to that boundary. A continuation is queued in
the same transaction that acknowledges the batch. New writes cannot extend a pass
indefinitely; writes beyond its boundary remain eligible for the next pass.

```python
from memory_core.runtime import MaintenancePolicy

policy = MaintenancePolicy(
    dream_after_writes=100,
    dream_batch_size=16,
    background=True,
)
# Supply maintenance=policy to MemoryCore.local(...).
# Keep `async with memory:` open for the application's lifetime.
```

Pending continuations survive restart. Without the background lifespan, call
`await memory.worker.run_once()` explicitly. Failed jobs retain their bounded retry
behavior and eventually become visible failures; they are not silently discarded.

A semantic proposal awaiting review no longer prevents the rest of its scan from
running. Handing a proposal to review advances scan coverage, **not approval**.
Approvals still check generation/revision freshness. Later writes or maintenance
may make an earlier proposal stale; retry it to replan instead of force-applying it.
Exact duplicate consolidation remains the automatic baseline. A configured reasoning
provider supplies semantic proposals; the storage engine does not infer their truth.

Dream searches yield between candidates and do not retain a SQLite write reservation
while scoring vectors. Each candidate still holds the service lock while reading its
indexes. Successful application keeps those indexes resident; the next access syncs
changed rows. Rollback invalidates the cache to prevent speculative state escaping.

The original service-level `dream()` / `apply_dream()` API still performs one explicit
batch. Automatic continuation belongs to the runtime worker. Old completed jobs
created before continuation support are not retroactively expanded; a new triggered
pass resumes from the stored cursor.

## Active memory and retained history

Active and disputed records remain in the resident cache. Disputed preferences must
remain available so unresolved conflicts can block an action. Superseded records
leave the vector, lexical, metadata, and payload caches when changes synchronize.
They are not deleted from SQLite, and their IDs cannot be reused.

Normal recall uses active memories. Explicit historical recall with
`statuses=[Status.SUPERSEDED]` streams retained records in batches of 256 and retains
only the best requested hits between batches. It does not populate the resident
cache with retired records. Historical search is intentionally more expensive than
normal recall and can scan the owner's retained collection.

`max_cached_owners` limits resident owner indexes, not bytes within one owner.
An owner's active index still grows with its active memories. The default native
vector search is exact and scans eligible vectors; returning a small result does
not imply sublinear search. Metadata filters reduce eligible work. No approximate
index or automatic clustering policy is silently enabled by these changes.

## Embedding storage

New embeddings use the `MCV1` binary format: a four-byte version marker followed by
little-endian IEEE binary64 values. A 384-dimensional vector occupies 3,076 bytes.
This retains Python float precision while avoiding JSON number parsing and storage
overhead. Both the new format and existing JSON embeddings are readable, so mixed
databases require no blocking conversion. Old cells are not automatically rewritten.
Older MemoryCore versions that only understand JSON cannot read the new cells;
keep a backup if a downgrade may be needed.

Consolidation reduces active retrieval work, not all historical disk usage. Events,
audit records, completed jobs, dream proposals, and superseded records remain
retained. This release does not silently prune evidence or idempotency records,
run `VACUUM`, or promise indefinitely bounded disk usage. A retention policy needs
explicit requirements for history and replay before deleting those records.

## Local measurements

```python
measurements = await memory.measurements(owner_id=authenticated_user.id)
print(measurements.model_dump_json(indent=2))
```

This reports:

- Memory counts by status and resident record count for this owner.
- Job counts by status, including failed and proposed jobs.
- Age of the oldest currently ready job, measured from its `available` timestamp.
  Retries reset availability; this is not total age since original capture. Older
  jobs with the legacy zero timestamp are excluded from the age calculation.
- Records after the persisted dream cursor. This is scan coverage, not unresolved
  semantic work; proposals have their own job count.
- SQLite file and WAL bytes, allocated page bytes, and reusable page bytes.
  These storage numbers cover the **whole database**, not a single owner's share.

Diagnostics aggregate stored rows without hydrating memory payloads into the cache.
They are intended for occasional sampling, not every action or tool call. They can
still scan tables and acquire the service lock. No telemetry leaves the machine.

## Reproducible lifecycle benchmark

Run each size in a separate process:

```sh
uv run python benchmarks/memory_lifecycle.py --count 5000 --rounds 2
uv run python benchmarks/memory_lifecycle.py --count 50000 --rounds 2
uv run python benchmarks/memory_lifecycle.py --count 100000 --rounds 2
```

The script loads a previously maintained seed corpus in bounded batches, then runs
public reconciliation writes, repeated evidence, explicit supersession, real worker
maintenance, foreground recall, and database reopen cycles. It checks that the
maintenance queue drains, successful dreams retain the cache, and labeled synthetic
queries still retrieve the expected records before and after maintenance/restart.

Reports include write latency, recall latency during maintenance, drain duration,
batch counts, recall@5/precision@5/MRR, current Linux RSS, process peak RSS, and disk
snapshots. The default workload uses 384 dimensions, two rounds, and 32 writes per
round plus one supersession. Bulk seeding bypasses reconciliation; reported write
latencies apply to incremental writes, not loading the entire seed through the API.
RSS includes interpreter, fixture, index, and allocator overhead. Small foreground
sample counts make tail-latency estimates noisy.

The synthetic queries test mechanics, isolation from unrelated records, and lifecycle
regressions. They do **not** establish natural-language semantic quality or the
correctness of an LLM's merge decisions. For that, use a fixed local embedding model
and held-out labeled queries with `memory_core.memory.evaluation.evaluate_retrieval`.
Compare proposed retrieval/index changes against exact search on the same corpus,
including rare exceptions and contradictory preferences, before promoting them.

## Recorded local run

These are single-machine results from the committed synthetic workload, with two
incremental rounds at each size. They are observations, not hardware-independent
performance guarantees. Each size ran in a fresh Python 3.13 process.

| Seed memories | Cold recall | Peak process RSS | Drain time per round | Foreground recall p95 per round | Allocated SQLite before → after |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5,000 | 0.34 s | 119 MiB | 0.25 / 0.24 s | 12.5 / 11.8 ms | 20.14 → 20.54 MiB |
| 50,000 | 3.37 s | 494 MiB | 3.17 / 3.25 s | 168.1 / 134.4 ms | 200.24 → 200.64 MiB |
| 100,000 | 5.96 s | 1,075 MiB | 7.41 / 7.21 s | 293.4 / 291.3 ms | 400.53 → 400.92 MiB |

All rounds drained without pending/running/failed jobs. All eight labeled synthetic
queries retained recall@5 = 1 and MRR = 1 through maintenance and reopen. Each round
added 24 active records net, retained one superseded record, and completed two dream
batches. The database grew because history and job evidence are retained.

Full measurements: [5k](benchmarks/lifecycle-5000.json),
[50k](benchmarks/lifecycle-50000.json), [100k](benchmarks/lifecycle-100000.json).
