# Session memory performance

Use the [persistent-session adapter](persistent-sessions.md) with the existing provider client. Exact preference preparation does not call an embedder. Query-based supporting recall embeds the query; stored memory embeddings are computed on ingestion. The local embedding adapter keeps its model loaded. Index hydration and generation catch-up reuse the existing owner cache. Consolidation remains in the lifecycle worker.

File-backed SQLite reads use deferred snapshots. Different owners can read concurrently; writes retain SQLite's single-writer transaction boundary. Same-owner operations remain serialized because their index and lazy payload cache share snapshot state. In-memory SQLite retains its single-connection serialization. First access may initialize owner metadata; warmed reads issue no writes.

## Recorded baseline

November 2, 2026, Python 3.13.13, one machine, temporary SQLite WAL databases. Baseline service: `280a44a`; updated service: `063e360`. The same harness and dependency environment ran sequentially, with other verification jobs idle. These measurements exclude embedding generation and provider inference.

```sh
uv run --extra dev python benchmarks/session_scale.py --repetitions 1000
```

The corpus uses seeded synthetic 512-dimensional vectors. Warm lookup uses 10,000 observations plus one preference, with half the observations eligible for the project. Concurrent cohorts contain 100 observations and one preference per owner, five warmup batches, at least 1,000 measured operations, and a 64-thread executor. Latency includes executor and connection-pool queueing. Shared-owner cohorts represent many sessions using the same authenticated memory owner.

| Measurement | Before p95 | After p95 | Initial target |
|---|---:|---:|---:|
| Warm exact preference lookup | 0.455 ms | 0.174 ms | <5 ms |
| Warm filtered recall, 10,000 observations | 4.466 ms | 3.970 ms | <20 ms |
| 16 sessions, different owners: preferences | 11.49 ms | 8.47 ms | Measure |
| 16 sessions, different owners: recall | 14.50 ms | 13.77 ms | Measure |
| 64 sessions, different owners: preferences | 34.81 ms | 64.20 ms | Measure |
| 64 sessions, different owners: recall | 52.51 ms | 52.95 ms | Measure |
| 16 sessions, shared owner: preferences | 9.09 ms | 6.21 ms | Measure |
| 16 sessions, shared owner: recall | 13.05 ms | 11.42 ms | Measure |
| 64 sessions, shared owner: preferences | 34.18 ms | 15.98 ms | Measure |
| 64 sessions, shared owner: recall | 47.07 ms | 36.84 ms | Measure |

The 64-owner preference tail regressed in this run. Independent read progress is covered by deterministic overlap tests; it does not guarantee lower latency for every workload. Future concurrency tuning should improve that tail without weakening snapshots or increasing other cohorts' latency. Raw reports include sample counts and total elapsed time: [before](benchmarks/session-scale-before.json), [after](benchmarks/session-scale-after.json).

## Cache residency

Cold loading 10,000 observations took 251 ms before and 253 ms after. A cyclic 17-owner workload with the default 16-owner cache loaded 1,001 records per owner. After the change, process RSS rose from 175 MiB to 288 MiB during that phase; revisiting owners still took about 25 ms median because the working set exceeded the cache limit. RSS includes allocator and database residency, not just index allocations.

`max_cached_owners` limits resident owner count, not bytes. `IndexPolicy.payload_cache_size` limits decoded payload count, not vector or metadata residency. Size these settings from the measured working set and deployment RAM allowance. One large owner can exceed an intended memory allowance; neither setting is a hard byte limit. A byte-based policy requires accounting for native vectors, Python indexes, and pinned active readers, rather than treating owner count as a RAM estimate.

## Retrieval quality and backend choice

The existing HNSW implementation was compared with exact search on 10,000 synthetic 512-dimensional vectors and 100 held-out queries. With `ef_search=2048`, top-ten overlap was 1.0; exact p95 was 2.98 ms and HNSW p95 was 3.22 ms. Exact remains the default. This index-only comparison does not establish semantic quality or end-to-end latency.

The existing offline MiniLM fixture has 20 labeled questions. Recall@3 remained 0.95 and reciprocal rank 0.90 before and after maintenance/restart. It is a small curated smoke corpus. Unknown questions can still retrieve candidates; a similarity result is not evidence that the question is answerable. See [the evaluation commands](indexed-memory.md#evaluation).

## Correctness gates

Tests cover owner/project isolation, current revisions, explicit conflicts, separate concurrent sessions, stale action receipts, failed context delivery, compaction reset, and optional-support budgets. Mandatory preferences never silently disappear to satisfy a budget. Memory updates preserve provider clients and system prompts. Tests also cover concurrent writers during readers, owner-cache eviction during an active operation, in-memory database cleanup, and schema-cache invalidation.
