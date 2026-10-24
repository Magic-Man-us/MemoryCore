# MemoryCore examples

Each example uses a local SQLite file (`example_memory.db` in the current directory)
created on first run. No services, no environment variables. Build the extension
first (`maturin develop`).

| Example | Shows |
|---|---|
| `basic_usage.py` | `remember` / `recall` / `hydrate` / `forget` on one SQLite file |
| `per_user_assistant.py` | One `MemorySystem` (and in-RAM index) per user over a shared `Database` |
| `smk_assistant.py` | The SMK assistant index: store a learned pattern, restart, `hydrate_assistant`, filtered recall |

```bash
python examples/basic_usage.py
python examples/per_user_assistant.py
python examples/smk_assistant.py
```

The examples use placeholder embeddings (`[0.1, 0.2, 0.3] * 128`). In real use, pass
vectors from your embedding model, or construct the system with an `Embedder` and
call `remember` / `recall` with text only.
