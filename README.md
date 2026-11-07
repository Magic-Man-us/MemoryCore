# MemoryCore

Local memory for AI agents: typed Python operations, SQLite persistence, indexed
metadata filtering, and Rust vector search. Retrieve a small set of relevant memories,
resolve applicable coding preferences, and consolidate overlapping records with an audit trail.

No AWS, hosted vector database, or remote embedding service is required. Supply vectors
or use an offline local embedder. Python 3.12+ is required; building from source also
requires Rust.

## Install and run

```bash
git clone https://github.com/Magic-Man-us/MemoryCore
cd MemoryCore
uv venv
uv pip install -e ".[dev]"
uv run maturin develop --release --uv
uv run python examples/typed_memory.py
```

The example stores a preference, loads it before a coding action, searches by tool,
and prepares a dream pass. It uses explicitly supplied toy vectors so it runs offline.
For real semantic retrieval, use a real embedding model consistently for writes and queries.

## Typed operations

```python
import asyncio
from memory_core import Database, EmbeddingSpec, MemoryService
from memory_core.memory.models import PreferenceMemory, RememberRequest, TaskContext

async def main():
    memory = MemoryService(
        Database("./memory.db"),
        embedding_spec=EmbeddingSpec(model_id="example-only", revision="1", dimensions=3),
    )
    await memory.call(
        RememberRequest(
            memory=PreferenceMemory(
                summary="Use uv for Python dependencies.",
                preference_key="python.package_manager",
                value="uv",
                applicability={"languages": ["python"]},
                metadata={"languages": ["python"], "tools": ["uv"]},
                source_refs=["conversation:123"],
            ),
            embedding=[1.0, 0.0, 0.0],  # Toy vector for this example only.
        ),
        owner_id="alice",  # Taken from trusted application context, never model arguments.
    )
    preferences = await memory.resolve_prefs(
        TaskContext(task_id="edit-api", attributes={"languages": ["python"]}),
        owner_id="alice",
    )
    print([(p.preference_key, p.value) for p in preferences.effective])

asyncio.run(main())
```

`REQUEST_ADAPTER` parses the discriminated operation union; `RESPONSE_ADAPTER` handles
responses. HTTP, MCP, or SDK wrappers can all call `memory.call(request, owner_id=...)`.
`python -m memory_core.mcpapp --factory module:function --owner ID` runs a local stdio MCP
server; there is no HTTP server.

| Operation | Behavior |
|---|---|
| `remember` | Find neighbors, reconcile, persist atomically |
| `recall` | Filter authorized, currently valid records, combine semantic and keyword candidates, return top-k |
| `resolve_prefs` | Return all applicable active rules and explicit conflicts |
| `prepare_task_context` | Resolve preferences and issue a task-bound context receipt |
| `describe_schema` | Discover request/memory schemas and registered fields |
| `register_metadata_field` | Add a typed field and aliases without changing code |
| `list_metadata_fields` / `list_metadata_values` | Discover field definitions and paginated vocabulary |
| `dream_plan` | Inspect bounded neighborhoods and persist a consolidation proposal |
| `dream_apply` | Atomically apply a current, validated proposal |

See [architecture and filtering](docs/architecture.md),
[agent integration and dreams](docs/agent-memory.md), and the
[runnable example](examples/typed_memory.py).

## Extensible metadata

There are no runtime topic, language, or tool enums and no 16-tool bitmask. The default
field **shapes** include `languages`, `tools`, `topics`, `domains`, `activities`, and
`frameworks`; their values are unrestricted strings. Register additional fields as
`keyword`, `keywords`, `number`, or `boolean`. Aliases resolve to a canonical field.
`kind` and `status` are read-only system fields.

Filtering supports `eq`, `in`, `contains_any`, `contains_all`, `range`, `exists`, `all`,
and `any`. Unknown fields, incompatible values, and invalid operators fail explicitly.
An empty list never silently removes a filter. Owner and project eligibility are enforced
outside the agent's filter expression.

## Automatic context and maintenance

Use `system.service.run_with_context(context, coding_action, owner_id=...)` to load
preferences before invoking your agent. Editing-tool wrappers can call `validate_context`
to reject a missing, stale, or differently scoped receipt. Refresh when task attributes
change. The host must wire these hooks into its SDK/tool execution paths; a standalone
memory library cannot intercept arbitrary tools or prove that a model obeyed a rule.

Every `dream_after_writes` successful writes (default 100), the service prepares a bounded
dream proposal. The built-in planner identifies exact structured duplicates. An injected
async `dream_planner` can propose deeper consolidation and improvements to metadata or
retrieval. Applying a plan is a separate explicit operation. Model/index recommendations
are never silently promoted; evaluate them against held-out cases first.

## Local embeddings

```bash
uv pip install -e ".[local]"
```

`LocalSentenceEmbedder` in `memory_core.memory.embedding` loads an **existing local model
directory**, with downloads and remote model code disabled. Supply its `.spec` as
`embedding_spec` and the instance as `embedder` to `MemoryService` or `MemoryCore.local`.
The optional model runtime and model weights are not bundled. The core tests use deterministic
vectors and do not validate a particular pretrained model's semantic quality.

Embedding model ID, revision, and dimensions are persisted per owner. Switching to an
incompatible identity fails even when dimensions match. Re-embedding requires an explicit
migration into a separately evaluated database; it is not an automatic dream action.

## Storage

SQLite is the only backend. `Database(":memory:")` gives an ephemeral database; a file
path persists across restarts. Queries automatically hydrate changed records into per-owner
indexes. Writes use SQLite transactions and generation/revision
checks, including when multiple service instances share the database.

`MemoryService.forget(memory_id, owner_id=...)` hard-deletes a record and every stored copy
(archives, dream plans, the HNSW snapshot). It is a host call, not a tool operation.

## Measured performance

A local synthetic run with **50,000 records and 384-dimensional vectors**, using 30 warm
queries and a release build, measured approximately:

| Operation | Median | p95 |
|---|---:|---:|
| Unfiltered recall | 30 ms | 31 ms |
| Recall filtered to about 500 records | 3.2 ms | 4.3 ms |

Cold recall including hydration took 1.5 seconds. Process peak RSS was about 300 MiB,
including fixture construction. Recall ran on one worker thread, with worker CPU time
equal to wall time. Each query returned at most five memories; all 12 matching preference
rules were returned independently of top-k.

These are measurements from one machine, not latency guarantees or a semantic-quality
benchmark. Reproduce with `uv run python benchmarks/memory_scale.py` against a release build
(`maturin develop --release`); a debug build measured 240 ms unfiltered. `--profile PATH`
adds a cProfile pass and reports the hottest functions. Compare retrieval
configurations using `memory_core.memory.evaluation.evaluate_retrieval` and your own labeled
queries before changing models or adding approximate search.

## Development

```bash
cargo clippy --all-targets -- -D warnings
cargo test
uv run pytest -q
uv run python examples/typed_memory.py
```

CI exercises Rust, Python 3.12/3.13, and the examples. The scale benchmark is manual and
requires no additional CI service. The [project website](https://magic-man-us.github.io/MemoryCore/)
contains the earlier visual overview; this README and the linked docs describe the 0.4 API.

Package metadata declares the MIT license.

### Native candidate scoring

`RustVectorIndex(dim)` ranks caller-selected string IDs by cosine similarity.
`upsert(id, vector)`, `remove(id)`, and `search(query, ids, k)` operate on the
in-memory index; persistence and authorization stay with the caller. Supply only
authorized IDs to `search`. Equal scores sort by ID, ascending. Scoring releases
the GIL and uses a native read lock.

Zero vectors are rejected by default. Use `RustVectorIndex(dim, allow_zero=True)`
for lexical embedders that produce empty vectors; their similarity is zero.
Non-finite values and dimension mismatches are always rejected.

Existing persistent Claude, OpenAI, and local conversations can use the provider-neutral
[`SessionContext` integration](docs/persistent-sessions.md) to refresh scoped preferences
and supporting memory under a caller-measured token budget, while retaining their clients
and system prompts.

[Session performance](docs/session-performance.md) records concurrency, retrieval, and cache-residency baselines.
