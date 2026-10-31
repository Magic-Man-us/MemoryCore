# Agent context, reconciliation, and dreams

## Automatic lookup before coding

Construct `TaskContext` from the current task and repository. Use dynamic attributes such
as `languages`, `activities`, `frameworks`, and registered custom fields. Do not infer an
owner from model output: bind the authenticated user in the host application.

```python
prepared = await memory.prepare_task_context(context, owner_id=owner_id)
# Inject prepared.preferences and prepared.supporting into the agent's context.
await memory.validate_context(prepared.receipt, context, owner_id=owner_id)
# Invoke the coding action through the host's tool wrapper.
```

`run_with_context` combines preparation and invocation of an async callback. When the task
expands to another language/project, call `refresh_task_context`. Tool wrappers should
validate the receipt immediately before an edit. Receipts are process-local, task-bound,
and invalidated by owner generation changes. Restarts require new preparation.

The library does not enforce another SDK's shell/file tools, and a receipt cannot guarantee
that an AI followed its instructions. Integrators must wire all mutation paths and run their
normal code checks. Memory content is retrieved evidence, not permission to override current
user instructions or the host's security policy.

## Schema discovery

Expose the generated request/response schemas through your preferred SDK or MCP wrapper.
Use `describe_schema`, `list_metadata_fields`, and paginated `list_metadata_values` before
inventing new metadata names. Field registration is an explicit operation, audited and
scoped to the caller. Existing definitions cannot be silently changed.

## Reconciliation

`remember` retrieves at most 20 neighboring active memories before writing. With no injected
reasoner, exact structured duplicates attach evidence and distinct facts are preserved.
The summaries, types, scope, applicability, and other semantic fields must match for the
automatic duplicate path; similar embeddings alone never authorize a merge.

An async `Reconciler` receives a candidate and neighbor list and returns a discriminated
`Reconciliation` decision. It can use a local model. Model inference is not bundled.

| Decision | Effect |
|---|---|
| `create` | Persist a distinct memory |
| `attach_evidence` | Add source references and increment the target revision |
| `supersede` | Keep the old record as superseded; insert a replacement with a link |
| `preserve_exception` | Keep a project-specific preference alongside a global one |
| `flag_conflict` | Preserve both as disputed |

Merge/supersede require the same kind/scope, and preference keys/applicability must match.
Explicit decisions need `expected_generation` and the target's `expected_revision`. A stale
snapshot raises `ConflictError`; fetch new candidates and reconsider rather than blindly
repeating a mutation. Callback-selected targets must be in the retrieved neighborhood.
The trusted caller may explicitly target another eligible record using its known ID.

## Dream lifecycle

A dream is bounded maintenance over long-term memory, not an unrestricted full-corpus prompt.
After `dream_after_writes` committed writes (100 by default), the service creates a proposal.
Manual `dream_plan` is also available. Defaults inspect 16 source records and up to eight
neighbors per source. A cursor visits records in generation/ID order across passes.

1. Take a versioned snapshot of source memories and relevant neighbors.
2. Run the built-in exact-duplicate planner or an injected async `DreamPlanner`.
3. Validate and persist the plan without modifying existing memories.
4. Return its ID; `RememberResponse.dream_plan_id` identifies automatically prepared work.
5. Apply with `dream_apply` when the host chooses to accept the proposal.
6. Reject stale or already-applied plans; apply all accepted changes in one transaction.

The default dream planner retires exact duplicates, transfers evidence to the retained
record, and keeps the retired source as superseded. A custom planner can propose replacement
summaries, revised metadata, exceptions, conflicts, or additional evidence. Replacements
must reference a source in the inspected neighborhood and obey normal reconciliation rules.
Each plan must use disjoint source/target IDs to avoid ambiguous chains within one transaction.

A `DreamProposal` also has recommendations for better keys, embedding models, or indexing
strategies. Those recommendations do not execute arbitrary code, register fields, change
models, or replace indexes. Register required fields explicitly; evaluate candidate retrieval
configurations against held-out queries and preference/isolation fixtures first. Re-embedding
currently means constructing a separate evaluated database and deliberately switching the
host after validation. Automatic model selection/promotion is not implemented.

Dream planning failure never misreports a committed memory as a failed write.
`RememberResponse.maintenance_error` reports the failure type, and the threshold remains due
for a later retry. Plans persist across restarts. Preparing a plan resets the scheduling
counter; applying it advances the consolidation cursor. Pending plans do not silently apply.

The SQLite audit trail records memory IDs, decisions, revisions through stored records, and
source references. No payload-log fingerprint feature is introduced.

### Indexed preference and duplicate lookup

Preference applicability uses per-owner field/value posting lists. A task reads only
postings for its supplied attributes, unions alternatives within each field, and requires
all fields of each rule to match. Unconditional rules are included separately. Project
scope, explicit-rule precedence, conflicts, and complete (not top-k) resolution still apply.
The host must supply task attributes; the store does not infer that a tool call is Python.
The worst case remains proportional to the matching rules, including broad global rules.

An incremental exact-content index finds structured duplicates independently of semantic
ranking. Lifecycle timestamps, IDs, revisions, and evidence references do not change
content identity; scope, applicability, values, metadata, and substantive text do. This is
strict equality, not a claim that differently worded statements mean the same thing.
Reconciliation adds at most one exact candidate beyond its 20 semantic neighbors. Dream
planning reserves a slot within its existing neighbor budget for an exact duplicate.
Supersession removes old entries, and dream merges update both records' timestamps.
These indexes rebuild from SQLite on startup and trade additional RAM for less query work.
