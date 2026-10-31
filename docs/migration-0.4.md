# Migrating to MemoryCore 0.4

0.4 removes `TopicBucket`, `ToolFlag`, `Level2Bits`, the packed SMK index, and the assistant
`remember_assistant`/`recall_assistant` APIs. Use typed operations and registered metadata.
The old `assistant_traces`, `ltm_traces`, STM, and working-memory tables are not deleted.

Back up your SQLite file before upgrading. Stop legacy writers while importing; migration
is idempotent and deliberately does not continuously mirror changes to old records.

1. Build the 0.4 native extension and instantiate the system with `embedding_spec`.
2. Use the **actual old embedding model ID/revision/dimension**. If unknown, re-embed into a
   new database rather than label old vectors as compatible with an arbitrary model.
3. Import the selected owner's historical user traces:

```python
from memory_core.memory.migration import import_legacy

count = await import_legacy(system.service, owner_id=authenticated_owner)
```

Historical assistant traces have no owner. An administrator must explicitly assign them:

```python
count = await import_legacy(
    system.service,
    owner_id=chosen_owner,
    include_unscoped_assistant=True,
)
```

This administrative method is intentionally excluded from the agent operation union. Never
expose the unscoped import switch directly to an untrusted model or unauthenticated endpoint.

The decoder maps historical numeric topic/tool values to strings. Unknown IDs become
`legacy_topic_<id>` or `legacy_tool_<id>` so they are not discarded. These old numeric mappings
exist only in the migration decoder, not the runtime taxonomy. Every original payload is
archived under `memory_legacy_archive`; the old tables also remain intact.

Imported records are observations. The migration does not guess that an old pattern or
summary is a mandatory user preference. Classify genuine preferences explicitly with their
scope, applicability, value, and provenance. Legacy IDs are namespaced to avoid collisions.

Malformed, oversized, or incompatible records fail the import rather than being silently
lost. Import stages validation/embedding before its atomic write. Missing embeddings require
a configured embedder; existing embeddings must match the configured specification.
Verify counts, representative recall cases, and archived source data before retiring legacy
call sites. The legacy user API continues to operate separately; its writes do not populate
the typed store automatically.
