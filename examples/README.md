# Examples

Run from the repository after `uv pip install -e ".[dev]"` and `uv run maturin develop --release --uv`.

| Example | Purpose |
|---|---|
| `uv run python examples/typed_memory.py` | Current typed preferences, schema discovery, filtering, and dreams |
| `uv run python examples/basic_usage.py` | Legacy user-memory persistence and recall |
| `uv run python examples/per_user_assistant.py` | Legacy per-user systems sharing a database |

The examples supply illustrative vectors; they do not load a semantic model. The typed
example uses an ephemeral database. Legacy examples may create local SQLite files.
