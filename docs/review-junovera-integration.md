# MemoryCore review — readiness for driving Junovera's memory system

Date: 2026-07-20. Scope: every Python module under `memory_core_py/`, both Rust sources
(`src/lib.rs`, `src/smk_index.rs`), packaging (`pyproject.toml`, `Cargo.toml`, `uv.lock`),
docs, and examples. All defects below were **reproduced empirically** on Python 3.12 with the
locked dependency versions, and the Rust engine was built and exercised via the compiled
extension. The companion integration plan lives in the junovera repo at
`docs/memorycore-integration-plan.md`.

## Verdict

The architecture is right: query/storage separation with a RAM-resident Rust index, layered
WM → STM → LTM stores, and a bitfield-prefiltered (SMK) assistant index is a sound shape for a
personal-assistant memory engine, and the Rust core compiles clean (zero warnings) and scores
correctly (the SMK bit packing round-trips exactly as documented). But the Python layer is
currently **not runnable**: the package cannot be imported, the flagship `AssistantMemoryTrace`
model cannot be instantiated, the documented constructor pattern raises without env vars, and
the wheel ships without the Python layer entirely. These are all small, mechanical fixes — the
design survives; the plumbing needs a pass.

## P0 — blockers (nothing works until these land)

### 1. `import memory_core_py` fails: ORM annotations unresolvable at runtime

`memory_core_py/storage/database.py:10-11` imports `datetime` only under
`if TYPE_CHECKING:`, but the ORM columns are annotated `Mapped[datetime]`
(`database.py:28,45-47`). SQLAlchemy resolves `Mapped[...]` annotations at class-definition
time, so importing the module raises:

```
MappedAnnotationError: Could not resolve all types within mapped annotation:
"Mapped[datetime]".
```

Reproduced on the locked SQLAlchemy 2.0.44 and current 2.0.51. Because
`memory_core_py/__init__.py` eagerly imports config → storage → database, **every** import
of the package (or any submodule) dies here. Fix: move the `datetime` import out of the
`TYPE_CHECKING` block. SQLAlchemy's declarative models, like Pydantic models, evaluate their
annotations at runtime.

### 2. `AssistantMemoryTrace` can never be instantiated

`memory_core_py/core/models.py:9-10` imports `MemoryKind`, `ToolFlag`, `TopicBucket` only
under `TYPE_CHECKING`, but `AssistantMemoryTrace` (a Pydantic model) uses them as field
annotations (`context_topic: TopicBucket`, `kind: MemoryKind`, `tools: set[ToolFlag]`).
Pydantic defers the unresolvable annotations and raises on first use:

```
PydanticUserError: `AssistantMemoryTrace` is not fully defined; you should define
`TopicBucket`, then call `AssistantMemoryTrace.model_rebuild()`.
```

The entire SMK path — `MemorySystem.remember_assistant`, `examples/smk_assistant.py` — is
dead at runtime. Fix: import the enums at runtime.

Root cause for both #1 and #2 is the same habit: a linter moving "type-only" imports into
`TYPE_CHECKING` without knowing Pydantic/SQLAlchemy evaluate annotations at runtime. Junovera
already guards against exactly this in its ruff config
(`[tool.ruff.lint.flake8-type-checking] runtime-evaluated-base-classes = ["pydantic.BaseModel",
"pydantic_settings.BaseSettings"]`); adopt the same config here, extended with
`sqlalchemy.orm.DeclarativeBase`.

### 3. The wheel ships without the Python layer, and a base install can't import it anyway

Two independent packaging faults:

- **`pydantic-settings` is not a declared dependency** (`pyproject.toml` lists numpy, pydantic,
  redis, sqlalchemy, mysql-connector-python), yet `storage/settings.py:9` and `config.py:8`
  import it at module load. It only reaches the environment transitively via the optional
  `[ai]` extra (pydantic-ai → mcp → pydantic-settings), confirmed in `uv.lock`
  (`requires-dist` for memory-core). A base install → instant `ModuleNotFoundError`.
- **maturin packages only the Rust module.** Built wheel contents (verified):
  `memory_core/__init__.py` + the compiled `.so` and dist-info — no `memory_core_py`
  anywhere. With `[tool.maturin] module-name = "memory_core"` and the Python package living in
  a differently-named directory, maturin has no reason to include it. Junovera cannot depend
  on MemoryCore until this is fixed.

Recommended fix: adopt maturin's mixed-layout convention — rename the Python package to the
distribution's import name and mount the extension inside it:

```toml
[tool.maturin]
python-source = "python"          # python/memory_core_py/...
module-name = "memory_core_py._native"
```

with `memory_core_py/__init__.py` re-exporting from `._native` where the raw bindings are
needed. (Any equivalent layout works; the invariant is: one wheel that contains both the
Python package and the extension, plus `pydantic-settings` in `[project.dependencies]`.
`numpy` is declared but unused — drop it — and `Cargo.toml`'s `rand` is never `use`d — drop
that too.)

### 4. Explicit constructor config cannot substitute for env vars

`MySQLStoreSettings` has four required fields (`host`, `user`, `password`, `database`), and
`LongTermStore.__init__` (`ltm_store.py:28`) runs `settings or LongTermStoreSettings()` —
constructing from env **before** overrides are applied. Without `LTM_DB_*` env vars:

```
ValidationError: 4 validation errors for LongTermStoreSettings — host/user/password/database
  Field required
```

This breaks the README Quick Start, `docs/architecture.md`, and all three examples, which all
pass connection info as kwargs. `MemoryCoreSettings` has the same fault one level up: its
`default_factory=lambda: LongTermStoreSettings()` fields (`config.py:33-37`) mean
`build_memory_system(overrides=...)` also raises before overrides are seen. Fix: pass
overrides as init kwargs so they participate in settings resolution (init kwargs beat env in
pydantic-settings):

```python
base_settings = settings or LongTermStoreSettings(
    **_filter_overrides(overrides, MYSQL_CONNECTION_KEYS)
)
```

and make `MemoryCoreSettings`'s nested factories lazy (build only when that store is enabled).

## P1 — correctness and durability

### 5. The documented startup path does not exist

Every diagram says "on startup: load all traces from LTM into the Rust index" — but
`LongTermStore` has exactly one method, `upsert_trace`. There is no
`fetch_traces_for_user`/`iter_all`, no `MemorySystem.hydrate()`, and **no schema creation**
(`Base.metadata.create_all` is called nowhere; no alembic). Today: writes fail on a fresh
database (tables don't exist), and a restart silently starts with an empty index while LTM
data sits unreachable. Needed: a `create_schema()` bootstrap, a read path on `LongTermStore`
(including the stored embedding bytes), and `MemorySystem.hydrate(user_id)` that replays
LTM traces into `memory_index` at startup.

### 6. Multi-user recall silently collapses (reproduced)

`MemoryEngine::search_candidates` (`src/lib.rs:217-245`) seeds candidates from tag/keyword
indexes that are **global across users**, then applies the user filter only at scoring. The
fallback to "all of this user's traces" triggers only when the *pre-filter* seed count is
below `limit`. Demonstrated: user alice has 2 traces, user bob has 5 traces matching the
query keyword — `search(user="alice", text="python", limit=3)` returns **0 results**, while
`limit=10` returns her 2. Any shared-index deployment (one engine, many `user_id`s — the
shape `MemorySystem` implies) degrades unpredictably. Fix: count candidates *after* the
user filter (or maintain per-user indexes internally).

### 7. Sync SQLAlchemy calls inside async APIs block the event loop

`MemorySystem.remember`/`recall` are `async` but call `ltm_store.upsert_trace`,
`stm_store.insert_trace`, and `fetch_recent_for_user` synchronously — every DB roundtrip
stalls the caller's event loop (in Junovera that freezes the token stream and the Slack
bridge). The README's "async, durable" write path is not async. Fix: wrap store calls in
`await asyncio.to_thread(...)` (or move to an async driver); alternatively accept a
write-behind queue so `remember` returns after the index write and LTM persistence trails.

### 8. Rust panics cross the FFI as `PanicException`

- `smk_index.rs:253,280` use `assert_eq!` on embedding dims — a wrong-dimension query from
  Python aborts with a raw panic + backtrace (reproduced) instead of `ValueError`.
- `smk_index.rs:292` sorts with `partial_cmp(...).unwrap()` — a NaN score (one NaN in any
  embedding produces one) panics the query. `lib.rs:280-284` already does this right
  (`unwrap_or(Equal)`); mirror it.

Return `PyResult<...>` with `PyValueError` for dim mismatches; treat NaN as
`Ordering::Equal` or filter non-finite scores.

### 9. Unknown enum values are silently coerced (reproduced)

`lib.rs` maps out-of-range discriminants to defaults in four places: unknown `topic` →
`RustPythonToolchain` (`lib.rs:450-455,524-529`), unknown `kind` → `Workflow` via the
wildcard arm (`lib.rs:458-467,532-537`). Demonstrated: `add(topic=99, ...)` then querying
`topic=1` **returns that trace**. Filtered recall quietly returns wrong memories instead of
rejecting bad input. Fix: `TryFrom` with an error branch surfaced as `PyValueError`.

Related drift: Python `TopicBucket.LOCAL_ENVIRONMENT = 3` vs Rust `TopicBucket::AwsIam = 3`
— same wire value, different meaning. Since the taxonomy will grow (Junovera needs its own
topics), consider making topic a validated plain `u8` range in Rust and keeping the semantic
enum only in Python — one source of truth, no triple-edit (smk_types.py, lib.rs ×2,
smk_index.rs Display) per new topic.

### 10. Assistant-index IDs are process-random

`assistant_index.py:38`: `mem_id = int(hash(trace.trace_uid) & 0xFFFF...)` — Python string
hashing is salted per process (PYTHONHASHSEED), so IDs differ every run and the
`_id_to_uid` map is in-memory only. Any future persistence/reload of the SMK index will
orphan every ID, and collisions are undetected. Use a deterministic digest
(`blake2b(digest_size=8)`) or `uuid.int & mask`, and keep a persisted uid↔id map alongside
LTM. Also: re-ingesting the same `trace_uid` appends a duplicate vector
(`smk_index.rs:252-255` has no upsert) — dedupe by id.

### 11. Access tracking and forgetting are dead code

`MemoryIndex.mark_accessed`/`remove` exist on the protocol and engine, but
`MemorySystem.recall` never marks results accessed, nothing ever calls `remove`, no
access_count flows back to LTM, and tombstoned traces keep their RAM forever
(`lib.rs:177-184` leaves vectors in place; `reindex_entry` at `lib.rs:161-165` accretes
stale index entries on update by design). STM rows expire logically
(`fetch_recent_for_user` filters) but are never purged. For a system whose premise is
memory dynamics (importance, recency, decay), the loop "recalled → reinforced; stale →
decayed; deleted → gone" needs to actually close.

### 12. Validation gaps that corrupt scoring

- `MemoryTrace.importance` has no bounds (accepts `7.3`, verified) while the score is
  `0.7·cosine + 0.3·importance` — one bad write outranks every honest memory forever. Add
  `ge=0.0, le=1.0` (the assistant trace already does).
- Query-embedding dim mismatch in the main engine returns cosine `0.0` silently
  (`lib.rs:187-189`), degrading ranking to importance-only with no signal. Surface an error.
- `RedisWorkingMemory.add_event` RPUSHes without `LTRIM` — within a TTL window a chatty
  session grows the list unboundedly; cap it (e.g. keep last N).

## P2 — design gaps to close before "drives everything"

- **Assistant memory has no persistence at all**: no LTM table for `AssistantMemoryTrace`,
  so Juno's learned patterns vanish on restart. Add an `assistant_traces` table (+ embedding
  column) and hydrate the SMK index from it.
- **`recall_assistant` hardcodes `min_generality`/`min_importance` to MEDIUM**
  (`system.py:150-160`) — parameters exist on the index wrapper but aren't exposed.
- **`recall` returns an untyped dict** (`{"ltm_candidates": ..., "wm_events": ...,
  "stm_traces": ...}`) — make it a typed `RecallResult` model; Junovera's codebase is
  strictly typed and will immediately wrap it otherwise.
- **No embedder abstraction**: every caller must bring vectors, which pushes the hardest
  integration decision onto every call site. Add an `Embedder` protocol (async
  `embed(texts) -> list[list[float]]`) accepted by `MemorySystem` so `remember`/`recall`
  can take raw text; keep explicit-embedding overloads.
- **`content` is persisted in STM but not LTM** (`ltm_traces` has no content column) — the
  full text of a memory is unrecoverable once STM expires. Decide (summary-only LTM is
  defensible) and document; if keeping, add the column.
- **No recency term in scoring** — `created_at` is stored, never used. A personal assistant
  wants `w₁·semantic + w₂·importance + w₃·recency_decay + w₄·access_frequency`; weights are
  already centralized in one place (`lib.rs:268`), so this is a cheap, high-leverage change.
- **MySQL-only SQL**: `on_duplicate_key_update` (mysql dialect insert) hard-locks storage to
  MariaDB/MySQL. Fine as a stated choice; for a single-user local assistant a SQLite profile
  would remove the heaviest operational dependency — worth a roadmap note either way.
- **Zero tests** (pytest configured, no tests exist) and **README drift**: env vars are
  documented as `WORKING_MEM_URL`/`WORKING_MEM_TTL_SECONDS` but the code reads
  `REDIS_DB_URL`/`REDIS_DB_TTL_SECONDS`; `LTM_DB_DATABASE` documented default
  `memory_core` doesn't exist (field required); Quick Start passes `redis_url=` but the
  parameter is `url=`; `examples/smk_assistant.py` imports `logfire` (not a dependency, and
  `from logfire import log` isn't its API). The P0/P1 repros above are natural first tests.
- **GIL/no-release**: engine searches hold the GIL; fine at 1–10 ms, but wrap hot paths in
  `py.allow_threads` when the index grows.

## What is genuinely good (keep it)

- The **layer split** (ephemeral Redis WM / TTL'd STM / durable LTM / RAM query index) maps
  cleanly onto how an assistant should remember, and `MemorySystem` keeps orchestration in
  one place behind a `MemoryIndex` protocol — swapping the index or adding an embedder won't
  disturb callers.
- The **SMK design** is the differentiator: a 64-bit packed key with bitfield prefilter
  before cosine is exactly the right cheap trick for "which of my learned patterns apply
  here", and the packing/unpacking is verified correct (raw key `5637150977` decomposes to
  topic=1, kind=1, tools=0b11, diff/gen/imp=HIGH exactly per the documented layout).
- The Rust engine's tag/keyword seeding + semantic scoring hybrid is a sensible recall
  shape; with #6 fixed it's a good foundation.
- Typed pydantic-settings config with per-store env prefixes and discriminated
  `MemorySettings` is the right idiom (once #4 makes kwargs work), and matches Junovera's
  configuration style closely.

## Suggested landing order

1. P0-1/P0-2 (two one-line import moves) + ruff `runtime-evaluated-base-classes` guard.
2. P0-4 settings construction; P0-3 packaging (maturin mixed layout + pydantic-settings dep).
3. P1-5 schema bootstrap + LTM read path + hydrate; then a pytest suite pinning all of the
   above (the repro scripts in this review convert directly into tests).
4. P1-6..12 engine correctness batch (one Rust PR: user-filtered fallback, PyResult errors,
   NaN-safe sort, enum TryFrom; one Python PR: to_thread, bounds, LTRIM, mark_accessed).
5. P2 items as they're needed by the Junovera integration (embedder protocol and
   `RecallResult` first — the integration plan depends on those).
