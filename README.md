# MemoryCore

A memory engine for AI assistants: SQL persistence with in-RAM vector search, built with
Rust and exposed to Python via PyO3.

All runtime queries hit a Rust-backed in-memory index (1-10ms). One SQL database — a local
SQLite file by default, PostgreSQL if you point the URL at one — persists long-term memory,
short-term memory, and working memory. No services required.

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
        end
    end
```

### Data flow

- **Startup** — `await system.hydrate(user_id)` loads the user's persisted traces from the
  database into the Rust index.
- **Query** — `recall` searches the in-RAM index only; the database is read just for the
  STM/WM sidebands.
- **Write** — `remember` persists to the database and updates the index in one call; store
  I/O runs in a worker thread so an async caller's event loop never blocks.

## Requirements

- Python >= 3.12
- Rust toolchain (only when building from source)

That's the whole list. SQLite ships with Python; PostgreSQL is optional (`.[postgres]`).

## Installation

```bash
git clone <repo-url> && cd MemoryCore
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
maturin develop
```

## Quick start

```python
import asyncio
from memory_core import build_memory_system

async def main():
    # SQLite file + auto-created schema; zero configuration.
    system = build_memory_system()

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

### Bring an embedder instead of vectors

Pass anything satisfying the `Embedder` protocol (`async def embed(texts) -> vectors`) and
call `remember`/`recall` with raw text — no embeddings at call sites:

```python
system = build_memory_system(embedder=MyEmbedder())
await system.remember(user_id="alice", summary="Timezone is US/Eastern",
                      importance=0.6, tags=["profile"])
result = await system.recall(user_id="alice", query_text="what timezone?", limit=5)
```

### Restart persistence

```python
system = build_memory_system()
loaded = await system.hydrate("alice")   # LTM → Rust index
```

### Forgetting

```python
await system.forget(trace_uid)   # removes from the index and the database
```

## SMK (Structured Memory Key) assistant index

The assistant-level index packs structured metadata into a 64-bit key for fast bitfield
filtering before cosine similarity:

```
bits  0-7:   topic (TopicBucket)
bits  8-10:  kind (MemoryKind)
bits 11-26:  tool_mask (16-bit flags)
bits 27-28:  difficulty (Level2Bits)
bits 29-30:  generality (Level2Bits)
bits 31-32:  importance (Level2Bits)
```

```python
from memory_core import AssistantMemoryTrace, MemoryKind, ToolFlag, TopicBucket

system = build_memory_system(overrides={
    "enable_assistant_index": True,
    "assistant_index_dim": 384,
})

await system.remember_assistant(trace=AssistantMemoryTrace(...), embedding=[...])
hits = await system.recall_assistant(
    query_embedding=[...], k=5,
    topic=TopicBucket.RUST_PYTHON_TOOLCHAIN,
    required_tools={ToolFlag.RS, ToolFlag.PY},
    allowed_kinds=[MemoryKind.PATTERN, MemoryKind.ANTI_PATTERN],
)
```

## Project structure

```
MemoryCore/
├── src/                      # Rust implementation
│   ├── lib.rs                # PyO3 bindings (compiled as memory_core._native)
│   └── smk_index.rs          # Structured Memory Key index
├── memory_core/              # Python package (ships in the wheel with the extension)
│   ├── core/                 # Protocols, models, MemorySystem orchestrator
│   ├── storage/              # Database, LTM, STM, SQL working memory
│   ├── indexing/             # RustMemoryIndex, AssistantMemoryIndex wrappers
│   └── types/                # SMK enums and feature extraction
├── examples/                 # Runnable usage examples
├── tests/                    # pytest suite (pytest tests)
└── docs/                     # Architecture + review documentation
```

## Configuration

Everything has a working default; environment variables override:

| Variable | Default | Description |
|---|---|---|
| `MEMORY_DB_URL` | `sqlite:///memory_core.db` | SQLAlchemy URL for the single database |
| `MEMORY_CORE_WM_TTL_SECONDS` | `3600` | Working-memory event TTL |
| `MEMORY_CORE_WM_MAX_EVENTS` | `500` | Per-user working-memory cap |
| `MEMORY_CORE_STM_TTL_SECONDS` | `900` | Short-term trace TTL |
| `MEMORY_CORE_ENABLE_STM` | `false` | Enable the STM layer |
| `MEMORY_CORE_ENABLE_ASSISTANT_INDEX` | `false` | Enable the SMK assistant index |

Programmatic overrides beat the environment:

```python
system = build_memory_system(overrides={
    "db": {"url": "postgresql+psycopg://memory:secret@db.example.com/memories"},
    "enable_stm": True,
})
```

## License

MIT
