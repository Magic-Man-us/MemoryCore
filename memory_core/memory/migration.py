"""One-way import of pre-0.4 records. Fixed vocabulary exists only in this decoder.

Legacy tables remain unchanged. Unscoped assistant records require an explicit trusted
owner assignment. This administrative import is intentionally not an agent operation.
"""

from __future__ import annotations

import asyncio
import json

from sqlalchemy import text

from memory_core.storage.codec import decode_embedding, encode_embedding

from .errors import ConflictError
from .models import ObservationMemory

# Historical wire values, not a runtime taxonomy. Unknown IDs are preserved verbatim.
_TOPICS = {
    1: "rust_python_toolchain",
    2: "memory_architecture",
    3: "local_environment",
    4: "db_schema",
}
_TOOLS = {1: "rust", 2: "python", 4: "uv", 8: "maturin", 16: "cfn"}


async def import_legacy(
    service, *, owner_id: str, include_unscoped_assistant: bool = False
):
    def read():
        with service._lock, service.store.transaction(owner_id) as session:
            rows = []
            for row in session.execute(
                text("SELECT * FROM ltm_traces WHERE user_id=:o"), {"o": owner_id}
            ).mappings():
                raw = dict(row)
                payload = {k: v for k, v in raw.items() if k != "embedding"}
                metadata = {"topics": json.loads(raw["tags"] or "[]")}
                rows.append(
                    (
                        "ltm",
                        raw["trace_uid"],
                        payload,
                        raw["summary"],
                        raw["content"] or raw["summary"],
                        metadata,
                        raw["embedding"],
                    )
                )
            if include_unscoped_assistant:
                for row in session.execute(
                    text("SELECT * FROM assistant_traces")
                ).mappings():
                    payload = json.loads(row["payload"])
                    topic = payload.get("context_topic")
                    tools = payload.get("tools", [])
                    metadata = {
                        "topics": [_TOPICS.get(topic, f"legacy_topic_{topic}")],
                        "tools": [_TOOLS.get(t, f"legacy_tool_{t}") for t in tools],
                        "languages": payload.get("languages", []),
                        "domains": payload.get("domains", []),
                        "activities": [payload.get("context_activity", "unknown")],
                    }
                    rows.append(
                        (
                            "assistant",
                            row["trace_uid"],
                            payload,
                            payload["summary"],
                            payload.get("rationale") or payload["summary"],
                            metadata,
                            row["embedding"],
                        )
                    )
            imported = {
                (r[0], r[1])
                for r in session.execute(
                    text("SELECT source,id FROM memory_imports WHERE owner=:o"),
                    {"o": owner_id},
                )
            }
            return [r for r in rows if (r[0], r[1]) not in imported]

    rows = await asyncio.to_thread(read)
    prepared = []
    for source, uid, payload, summary, content, metadata, raw in rows:
        vector = await service._vector(decode_embedding(raw), summary)
        memory = ObservationMemory(
            id=f"legacy:{source}:{uid}",
            summary=summary,
            observation=content,
            metadata=metadata,
            source_refs=[f"legacy:{source}:{uid}"],
        )
        prepared.append((source, uid, payload, memory, vector))

    def write():
        with service._lock, service.store.transaction(owner_id) as session:
            cache, registry, _ = service._snapshot(session, owner_id)
            session.execute(
                text("""CREATE TABLE IF NOT EXISTS memory_legacy_archive (
                owner TEXT NOT NULL, source TEXT NOT NULL, id TEXT NOT NULL,payload TEXT NOT NULL,
                PRIMARY KEY(owner,source,id))""")
            )
            imported = 0
            for source, uid, payload, memory, vector in prepared:
                exists = session.execute(
                    text(
                        "SELECT 1 FROM memory_imports WHERE owner=:o AND source=:s AND id=:id"
                    ),
                    {"o": owner_id, "s": source, "id": uid},
                ).first()
                if exists:
                    continue
                if memory.id in cache.memories:
                    raise ConflictError("legacy import ID collision")
                memory = service._normalize(memory, registry)
                generation = service.store.bump(
                    session, owner_id, "legacy_import", {"source": source, "id": uid}
                )
                service.store.put(
                    session, owner_id, memory, encode_embedding(vector), generation
                )
                params = {
                    "o": owner_id,
                    "s": source,
                    "id": uid,
                    "p": json.dumps(payload, default=str),
                }
                session.execute(
                    text("INSERT INTO memory_legacy_archive VALUES (:o,:s,:id,:p)"),
                    params,
                )
                session.execute(
                    text("INSERT INTO memory_imports VALUES (:o,:s,:id)"), params
                )
                imported += 1
            return imported

    return await asyncio.to_thread(write)
