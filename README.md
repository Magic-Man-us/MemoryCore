# MemoryCore

A memory engine for AI assistants: SQL persistence with in-RAM vector search, built with
Rust and exposed to Python via PyO3.

[Explore the MemoryCore website](https://magic-man-us.github.io/MemoryCore/) for the visual
overview, architecture, performance profile, and quickstart.

All runtime queries hit a Rust-backed in-memory index (1-10ms). One SQL database (a local
SQLite file by default, PostgreSQL if you point the URL at one) persists long-term memory,
short-term memory, working memory, and the assistant's learned patterns. No services
required.

## Architecture

```mermaid
graph TB
    subgraph MemorySystem
        direction TB
        subgraph Query Layer
            RI[RustMemoryIndex<br><i>Vector search, in RAM</i>]
            AI[AssistantMemoryIndex<br><i>SMK bitfield filtering</i>]
        end
        subgraph "Persistence (one SQL database, SQLite by default)"
            LTM[LongTermStore<br><i>durable traces + embeddings</i>]
            STM[ShortTermStore<br><i>TTL'd traces</i>]
            WM[SqlWorkingMemory<br><i>rolling session events, TTL + cap</i>]
            AS[AssistantStore<br><i>SMK traces + embeddings</i>]
        end
    end
    LTM -. hydrate .-> RI
    AS -. hydrate_assistant .-> AI
```

### Data flow

- **Startup**: `await system.hydrate(user_id)` loads the user's persisted traces into the
  Rust index; `await system.hydrate_assistant()` does the same for the SMK index.
- **Query**: `recall` searches the in-RAM index only, then bumps each hit's access count
  (in RAM and in the database) and attaches the STM/WM sidebands.
- **Write**: `remember` validates, persists, then publishes to the index. Input the index
  would reject (a wrong embedding dimension) never reaches the database, a failed
  database write leaves the index untouched, and a trace is never searchable before it
  is durable. Assistant writes follow the same order, serialized per system.
- **Forget**: `forget(trace_uid)` removes the trace from the index (freeing its memory),
  LTM, STM, and any working-memory events linked to it.

Store I/O and Rust search run in worker threads with the GIL released, so an async
caller's event loop never blocks.

## Requirements

- Python >= 3.12
- Rust toolchain (only when building from source)

That's the whole list. SQLite ships with Python; PostgreSQL is optional (`.[postgres]`).

## Installation

```bash
git clone https://github.com/Magic-Man-us/MemoryCore && cd MemoryCore
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
maturin develop --uv
```

## Quick start

```python
import asyncio
from memory_core import build_memory_system

async def main():
    # Local SQLite file + auto-created schema; zero configuration.
    system = build_memory_system(sqlite_path="~/.my-assistant/memory.db")

    await system.remember(
        user_id="alice",
        summary="Prefers Python for scripting tasks",
        importance=0.8,
        tags=["preferences", "programming"],
        embedding=[0.1, 0.2, ...],   # from your embedding model
    )

    result = await system.recall(
        user_id="alice",
        query_text="programming languages",
        limit=5,
        query_embedding=[0.15, 0.25, ...],
    )
    print(result.ltm_candidates)

asyncio.run(main())
```

### Choosing the database

| You want | Do |
|---|---|
| The default local file (`./memory_core.db`) | `build_memory_system()` |
| A local file somewhere specific | `build_memory_system(sqlite_path="~/data/memory.db")` (`~` expanded, parent dirs created) |
| A throwaway in-process database | `build_memory_system(sqlite_path=":memory:")` |
| PostgreSQL | `build_memory_system(overrides={"db": {"url": "postgresql+psycopg://..."}})` |

`sqlite_path` wins over any URL. The same choices work via environment
(`MEMORY_DB_SQLITE_PATH`, `MEMORY_DB_URL`) or directly on `Database(sqlite_path=...)`.
File databases run in WAL mode with a 30 s busy timeout, so a CLI and a daemon can share
one file. In-memory databases share one connection across worker threads and serialize
access to it, so they behave like a file database that disappears on exit.

### Bring an embedder instead of vectors

Pass anything satisfying the `Embedder` protocol (`async def embed(texts) -> vectors`) and
call `remember`/`recall` with raw text, no embeddings at call sites:

```python
system = build_memory_system(embedder=MyEmbedder())
await system.remember(user_id="alice", summary="Timezone is US/Eastern",
                      importance=0.6, tags=["profile"])
result = await system.recall(user_id="alice", query_text="what timezone?", limit=5)
```

### Restart persistence

```python
system = build_memory_system()
loaded = await system.hydrate("alice")        # LTM -> Rust index
learned = await system.hydrate_assistant()    # assistant_traces -> SMK index
```

### Forgetting and housekeeping

```python
await system.forget(trace_uid)            # index, LTM, STM, linked WM events
await system.forget_assistant(trace_uid)  # SMK index + assistant store
await system.purge_expired()              # expired STM/WM rows for every user
```

## Scoring

Each candidate is scored as

```
semantic * cosine + importance * importance
  + recency * 0.5 ** (age / half_life)
  + access  * a / (1 + a),   a = ln(1 + access_count)
```

The defaults (`semantic=0.7`, `importance=0.3`, `recency=0`, `access=0`) are the original
semantic + importance score, so ranking does not change unless you opt in:

```python
system = build_memory_system(overrides={
    "scoring": {"semantic": 0.6, "importance": 0.2, "recency": 0.15, "access": 0.05,
                "recency_half_life_days": 30},
})
```

Ties break newer-first, then by trace uid, so results are deterministic.

## SMK (Structured Memory Key) assistant index

The assistant-level index packs structured metadata into a 64-bit key for fast bitfield
filtering before cosine similarity:

```
bits  0-7:   topic (TopicBucket, any 1..=255)
bits  8-10:  kind (MemoryKind)
bits 11-26:  tool_mask (16-bit ToolFlag mask)
bits 27-28:  difficulty (Level2Bits)
bits 29-30:  generality (Level2Bits)
bits 31-32:  importance (Level2Bits)
```

Topic and tool meanings live only in Python (`TopicBucket`, `ToolFlag`); Rust treats them
as validated integers, so adding a topic is a one-line Python change.

```python
from memory_core import AssistantMemoryTrace, MemoryKind, ToolFlag, TopicBucket

system = build_memory_system(overrides={
    "enable_assistant_index": True,
    "assistant_index_dim": 384,
})

trace = await system.remember_assistant(trace=AssistantMemoryTrace(...), embedding=[...])
trace.smk_raw   # the packed key, set on ingest
hits = await system.recall_assistant(
    query_embedding=[...], k=5,
    topic=TopicBucket.RUST_PYTHON_TOOLCHAIN,
    required_tools={ToolFlag.RS, ToolFlag.PY},
    allowed_kinds=[MemoryKind.PATTERN, MemoryKind.ANTI_PATTERN],
)
```

Re-remembering a `trace_uid` replaces it. `recall_assistant` defaults to
PATTERN/ANTI_PATTERN with at least MEDIUM generality and importance; pass
`allowed_kinds=list(MemoryKind), min_generality=None, min_importance=None` to search
everything. `remember_assistant` raises if the assistant index is not enabled.

## Project structure

```
MemoryCore/
├── src/                      # Rust implementation
│   ├── lib.rs                # Conversation engine + PyO3 bindings (memory_core._native)
│   └── smk_index.rs          # Structured Memory Key index
├── memory_core/              # Python package (ships in the wheel with the extension)
│   ├── core/                 # Protocols, models, MemorySystem orchestrator
│   ├── storage/              # Database, LTM, STM, working memory, assistant store
│   ├── indexing/             # RustMemoryIndex, AssistantMemoryIndex wrappers
│   └── types/                # SMK enums and feature extraction
├── examples/                 # Runnable usage examples
├── tests/                    # pytest suite
└── docs/                     # Architecture + review documentation
```

## Configuration

Everything has a working default; environment variables override, and programmatic
`overrides` (validated; unknown keys raise) beat the environment:

| Variable | Default | Description |
|---|---|---|
| `MEMORY_DB_SQLITE_PATH` | unset | Local SQLite file (or `:memory:`); wins over the URL |
| `MEMORY_DB_URL` | `sqlite:///memory_core.db` | SQLAlchemy URL for the single database |
| `MEMORY_CORE_WM_TTL_SECONDS` | `3600` | Working-memory event TTL |
| `MEMORY_CORE_WM_MAX_EVENTS` | `500` | Per-user working-memory cap |
| `MEMORY_CORE_STM_TTL_SECONDS` | `900` | Short-term trace TTL |
| `MEMORY_CORE_ENABLE_STM` | `false` | Enable the STM layer |
| `MEMORY_CORE_ENABLE_ASSISTANT_INDEX` | `false` | Enable the SMK assistant index |
| `MEMORY_CORE_ASSISTANT_INDEX_DIM` | unset | Assistant index dimension (required when enabled) |
| `MEMORY_CORE_SCORING__<FIELD>` | see Scoring | `SEMANTIC`, `IMPORTANCE`, `RECENCY`, `ACCESS`, `RECENCY_HALF_LIFE_DAYS` |

```python
system = build_memory_system(overrides={
    "db": {"url": "postgresql+psycopg://memory:secret@db.example.com/memories"},
    "enable_stm": True,
})
```

## Development

```bash
cargo clippy --all-targets -- -D warnings
cargo test                 # Rust unit tests
maturin develop --uv
pytest                     # Python suite (real Rust engine, temp SQLite)
```

CI runs all of the above plus the examples on Python 3.12 and 3.13.

## License

MIT
