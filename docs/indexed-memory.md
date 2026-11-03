# Local indexed memory and retention

MemoryCore keeps SQLite as its source of truth. Optional HNSW accelerates semantic
candidate search; it does not decide whether a memory is true, merge records, or
replace exact preference resolution. Application metadata remains registered through
the existing schema APIs rather than a fixed language/project taxonomy.

## Enable an index

```sh
uv sync --extra ann
```

```python
from memory_core import IndexPolicy, MemoryCore, RetentionPolicy

memory = MemoryCore.local(
    "memory.db",
    embedding_spec=embedder.spec,
    embedder=embedder,
    index_policy=IndexPolicy(
        backend="hnsw",
        payload_cache_size=256,
        exact_threshold=1000,
        ef_search=256,
    ),
    retention=RetentionPolicy(enabled=True),
)

async with memory:
    # Use memory.task(...) and its prepare/require_knowledge/record_finding/
    # complete lifecycle as documented in the agent integration guide.
    ...
```

`embedder` is an explicitly configured embedding provider. The `ann` extra adds
hnswlib and NumPy; building hnswlib may require a C++ compiler. It does not download
an embedding model. The default backend remains `exact`, with no ANN dependency.

Semantic retrieval first computes eligible IDs using owner, scope, status and
metadata constraints. Small eligible sets use exact scoring. Larger sets retrieve
approximate candidates, apply eligibility, and widen the search when too few remain.
If the candidate limit still cannot supply enough eligible hits, retrieval scores
the eligible set exactly in bounded chunks. This fallback ensures enough eligible
results, but does not guarantee exact ranking when ANN supplies enough candidates.
Mandatory preferences use their exact applicability indexes regardless of backend.

`ef_search` trades query cost for recall. `max_candidates` caps the candidate list,
not total graph traversal or exact fallback cost. Validate these settings on your
own embeddings and filters. Eligibility sets and metadata postings still grow with
collection size; this is not an entirely sublinear request path.

## Payloads and recovery

Each resident owner keeps compact headers, metadata/lexical postings, IDs and vector
indexes. Full validated payloads are fetched from SQLite on demand and retained in
an LRU cache bounded by `payload_cache_size` (zero disables retention). Returned
responses and a current operation's candidates can temporarily hold additional
payloads. This bounds the payload cache, not total process RAM. `max_cached_owners`
still bounds the number of resident owner indexes.

HNSW snapshots are stored atomically in SQLite with embedding identity and generation.
Idle worker ticks checkpoint after `checkpoint_after_updates`; runtime exit forces
a checkpoint. Hosts using only MemoryService can explicitly call
`await service.checkpoint_indexes(force=True)`. Evicted/uncheckpointed graphs can
be rebuilt from SQLite. A stale snapshot replays newer records and archival removals.
Invalid snapshots rebuild rather than becoming the source of truth. Native snapshots
are trusted local database data, not an interchange format for untrusted uploads.

Checkpoint serialization happens outside the SQLite write transaction, but holds the
service lock. Checkpoints can therefore delay foreground operations. Graph deletion
tombstones are rebuilt at checkpoint when sufficiently numerous; rebuilds are not
incremental and temporarily require both graphs in RAM. Startup still reads records
to rebuild metadata even when the graph is restored.

## Opt-in retention

Retention defaults to disabled. When enabled, idle maintenance rotates through owners,
processing at most `batch_size` rows in each category per pass. Continuous job traffic
can defer this idle work; hosts can schedule `await memory.maintain(owner_id=owner)`
explicitly and inspect `await memory.measurements(owner_id=owner)`.

- Superseded records older than `superseded_after_days` move to lossless compressed
  archives. Their vectors leave the active graph. Historical recall still searches
  them and IDs cannot be reused.
- Completed jobs older than `completed_job_days` lose duplicate input payloads;
  results and identity records remain. Pending, running, failed and review jobs are
  protected. Legacy completed jobs without a recorded completion time are skipped.
- Applied dream plans move to compressed archives and remain loadable for replay.
- Events older than `event_replay_days` move to lossless archives only after task
  completion and completion of its jobs, with no unresolved knowledge gaps.
  Replaying an archived event raises `ConflictError` explaining that its replay
  window expired. It never creates a new side effect from that event ID.

Evidence, audit history, task identities and cold archives are retained. Disk use can
continue growing; this policy does not impose a hard storage cap or erase provenance.
`await memory.reclaim_storage(pages=256)` requests a passive WAL checkpoint and bounded
incremental vacuum only if the database already uses that mode. It does not perform
a blocking full vacuum or change an existing database's vacuum mode.

## Evaluation

Run reproducible index comparisons separately from end-to-end lifecycle measurements:

```sh
uv run --extra ann python benchmarks/ann_accuracy.py --count 50000 --ef 2048
uv run --extra ann python benchmarks/memory_lifecycle.py --count 5000 --backend hnsw
uv run --extra ann --extra local python benchmarks/semantic_quality.py \
  --model-path /absolute/local/model \
  --model-id sentence-transformers/all-MiniLM-L6-v2 \
  --revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41
```

The semantic script loads only the supplied local model. Its 20 labeled paraphrases
are a curated smoke corpus, not independent quality validation. Both backends achieved
recall@3 of 0.95 and MRR of 0.90 before and after dream maintenance/restart in the
recorded run. This small corpus takes the exact candidate path even with HNSW selected.
Large-corpus ANN accuracy is measured separately using held-out random-vector queries.
Unanswerable queries still returned candidates: similarity is not proof of support.
Use the knowledge evaluator/researcher boundary to resolve declared gaps.

These runs do not establish live Claude behavior or LLM reconciliation quality.
Those require authenticated provider execution and a separately labeled decision
corpus. No retrieval score should automatically authorize a semantic merge.

### Recorded index results

Single-machine Python 3.13 runs, 384 dimensions and 30 independent random queries.
Overlap compares ANN's top ten against exact search on identical vectors. These runs
measure index mechanics, not semantic correctness or complete application latency.

| Vectors | Search effort | Mean top-10 overlap | Exact p50 | HNSW p50 | Graph bytes |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5,000 | 2,048 | 100% | 2.47 ms | 3.31 ms | 8,744,396 |
| 50,000 | 2,048 | 98% | 30.36 ms | 18.52 ms | 87,422,996 |
| 100,000 | 2,048 | 90.3% | 61.47 ms | 28.71 ms | 174,839,396 |
| 100,000 | 8,192 | 100% | 65.52 ms | 51.57 ms | 174,839,396 |

The 100k run at effort 2,048 **failed** the 98% acceptance target. Raising effort
recovered accuracy with additional query cost. None of these rows validate the
production default effort of 256 on a 100k corpus. The exact backend remains the
safe baseline when approximate misses are unacceptable. Runs shared a development
machine with other checks, so timings are indicative rather than isolated benchmarks.

Raw reports: [5k](benchmarks/ann-5000.json), [50k](benchmarks/ann-50000.json),
[100k initial](benchmarks/ann-100000-ef2048.json),
[100k tuned](benchmarks/ann-100000-ef8192.json),
[real embedding smoke](benchmarks/semantic-quality.json).

### Recorded lifecycle results

These runs exercise public writes, worker draining, recall during maintenance and
restart using the HNSW backend at its default settings. Synthetic anchors include
lexical signals, so their perfect recall does not contradict the random-vector
ANN misses above. Each run used two rounds of 32 incremental writes.

| Seed memories | Cold recall | Peak RSS | Drain seconds by round | Recall p95 during maintenance | Final allocated DB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5,000 | 2.28 s | 141 MiB | 0.30 / 0.26 | 44.0 / 12.6 ms | 29.12 MiB |
| 50,000 | 35.13 s | 590 MiB | 2.34 / 1.75 | 104.6 / 72.8 ms | 285.74 MiB |
| 100,000 | 43.79 s | 1090 MiB | 5.32 / 4.01 | 270.6 / 160.1 ms | 571.05 MiB |

All eight synthetic queries retained recall@5 and MRR of 1. All queues drained;
final payload caches held eight records, while indexes retained all active IDs.
Peak RSS includes graph construction, snapshots, interpreter and allocator overhead.
The database also stores graph snapshots; lossless archival is not disk deletion.

Raw lifecycle reports: [5k](benchmarks/lifecycle-hnsw-5000.json),
[50k](benchmarks/lifecycle-hnsw-50000.json),
[100k](benchmarks/lifecycle-hnsw-100000.json).

For decision-level checks, use the [reconciliation evaluation guide](reconciliation-evaluation.md).
It scores actions, target IDs and revisions independently of retrieval metrics.
