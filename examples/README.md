# Memory Core Examples

This directory contains usage examples demonstrating different aspects of the memory_core package.

## Examples

### basic_usage.py
Demonstrates the fundamental usage pattern for the MemorySystem:
- Setting up LTM, STM, and working memory stores
- Initializing the vector index
- Storing and recalling user memories
- Basic memory operations

**Run with:**
```bash
python examples/basic_usage.py
```

### per_user_assistant.py
Shows how to implement per-user memory isolation:
- Creating separate memory systems for each user
- Managing multiple user memory spaces
- Preventing memory leakage between users
- Pattern for multi-tenant applications

**Run with:**
```bash
python examples/per_user_assistant.py
```

### smk_assistant.py
Demonstrates the SMK (Semantic Memory Kernel) assistant index:
- Using AssistantMemoryIndex for specialized memory
- Working with SMK features (topic, kind, tools, etc.)
- Filtering memories by semantic attributes
- Pattern-based memory retrieval

**Run with:**
```bash
python examples/smk_assistant.py
```

## Prerequisites

Before running these examples, ensure you have:

1. **MariaDB** running with appropriate credentials
2. **Redis** running (default: localhost:6379)
3. **Environment variables** set (optional):
   - `LTM_DB_HOST`, `LTM_DB_USER`, `LTM_DB_PASSWORD`, `LTM_DB_DATABASE`
   - `STM_DB_HOST`, `STM_DB_USER`, `STM_DB_PASSWORD`, `STM_DB_DATABASE`
   - `WORKING_MEM_URL`, `WORKING_MEM_TTL_SECONDS`

## Notes

- These examples use placeholder embeddings (`[0.1, 0.2, 0.3] * 128`)
- In production, use actual embeddings from an embedding model
- Database tables are created automatically by the stores
- Adjust connection parameters as needed for your environment
