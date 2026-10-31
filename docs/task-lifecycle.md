# Automatic task memory

MemoryCore owns context preparation, durable capture, bounded knowledge resolution,
reconciliation, and background dreaming. Python orchestrates these operations; the
existing Rust engine performs vector search. No hosted service is required.

## Start with a runnable example

```sh
uv sync --extra dev
uv run python examples/task_lifecycle.py
```

The example uses deterministic local providers and a temporary SQLite database. It
shows a full task with a declared knowledge gap, research, evidence validation,
automatic finding capture, explicit completion, and durable maintenance. It does
not demonstrate real embedding or model quality.

For real embeddings, install `uv sync --extra local` and configure
`LocalSentenceEmbedder` with an existing local model directory. Model files are not
downloaded automatically. Supply your permitted research/evaluation callbacks.

```python
from memory_core import MemoryCore
from memory_core.memory.models import TaskContext
from memory_core.runtime import KnowledgeGap, MaintenancePolicy, TaskOutcome

async with MemoryCore.local(
    "memory.db",
    embedding_spec=embedder.spec,
    embedder=embedder,
    reconciler=reconciler,
    dream_planner=dream_planner,
    researcher=researcher,
    knowledge_evaluator=evaluator,
    maintenance=MaintenancePolicy(dream_after_writes=100),
) as memory:
    async with memory.task(
        owner_id=authenticated_user.id,
        context=TaskContext(
            task_id=task_id,
            project_id=project_id,
            attributes={"languages": ["python"], "activities": ["coding"]},
        ),
        objective="Implement the project retry policy",
    ) as task:
        prepared = await task.prepare()
        knowledge = await task.require_knowledge(
            KnowledgeGap(
                id="retry-policy",
                question="Which requests may this project retry?",
                requirements=["Find an explicit project policy"],
            )
        )
        # execute() refreshes context and blocks conflicts/unresolved declared gaps.
        # The callback supplies prepared context + knowledge to your agent/SDK.
        result = await task.execute(lambda refreshed: agent_run(refreshed))
        await task.complete(
            TaskOutcome(
                status="succeeded",
                summary=result.summary,
                evidence=result.evidence,
            )
        )
```

`MemoryCore(service, ...)` can wrap an existing typed `MemoryService` instead.
`prepare_task_context(context, owner_id=..., query=...)` also remains directly
available. The runtime does not dispose a supplied service's database; stop workers
before calling `memory.service.store.db.dispose()` when your application shuts down.

## Task hooks and SDK integration

| Hook | Call | Result |
| --- | --- | --- |
| Task start/resume | Enter `memory.task(...)` | Persist context, objective, attribute sources |
| Before an action | `task.before_action()` or `task.execute(callback)` | Fresh context; refuse conflicts and unresolved/expired gaps |
| Missing information | `task.require_knowledge(gap)` | Typed resolved, needs-user-input, or unresolved result |
| Useful discovery | `task.record_finding(finding, event_id=...)` | Durable candidate and maintenance job |
| Task completion | `task.complete(outcome, findings=...)` | Immutable outcome and optional findings queued atomically |
| Whole task callback | `task.run(callback)` | Guard invocation; capture returned `TaskRunResult` automatically |

An SDK adapter should invoke these hooks at its actual action boundary. Merely
exposing tools does not guarantee their use. `execute` guards one action without
completing the task; `run` expects a typed `TaskRunResult` and completes the task.
The prepared task context contains preferences, conflicts, supporting memories,
resolved/unresolved knowledge by gap ID, the task objective, and a receipt.
`prepare_task` exposes this same operation to tools; `guard=true` enforces the action
checks. Pass it as structured context; preserve source references and do not treat
retrieved text as higher-priority system instructions.

Task identity is scoped to the authenticated owner. Resuming requires the same
objective/project and compatible explicit attributes. A different objective or
context needs a new task ID. Explicit attributes override resolver contributions;
conflicting inferred values raise an error. Register new fields through the schema
registry. Context resolvers receive the registered definitions; no language or tool
taxonomy is baked into the runtime. Provider modules define the callback protocols.

Context-manager exit does **not** imply success. An exception leaves the task
resumable. A host may explicitly report failed/cancelled outcomes. Supported
findings captured before failure survive. Completion replay is idempotent when the
same content is supplied; changed completion is rejected. For finding retries,
reuse both the event ID and the original finding, including evidence timestamps.

## Knowledge resolution

1. Persist the declared gap before making external calls.
2. Look up at most ten scoped active memory candidates.
3. Ask the evaluator whether available evidence satisfies the question/requirements.
4. If unresolved, call the configured researcher once, with a timeout and at most
   32 returned evidence records, then evaluate again.
5. Validate that every cited evidence object exactly matches a supplied object and
   satisfies the requested age limit. Future timestamps are rejected.
6. Persist a resolved answer and its candidate-memory job atomically. Otherwise
   retain an explicit gap which blocks guarded actions and successful completion.

Fresh resolved answers are reused within the task. Evidence age and referenced
memory revisions/status are checked again at the action boundary. There is no confidence threshold that magically establishes
truth: the evaluator must assess sufficiency, and the researcher must provide
trustworthy sources. MemoryCore checks provenance membership, not whether a quoted
source proves a claim. Model calls receive copies of bounded input snapshots.

No evaluator means an explicit unresolved result. No researcher means local memory
is the only source. Provider failures become unresolved results, not fabricated
answers. A `NeedsUserInput` result should be presented to the user by the host; the
host can make the user's answer available through its configured research provider
and retry the same gap. Already resolved findings do not wait for task success.
The system cannot identify every unknown unknown. Hosts must declare required gaps
and wire guards into dependent tools.

## Durable maintenance and review

Findings/outcomes first become durable candidates. They become searchable memories
when the worker reconciles them. Captured findings are observations scoped to the
task's project (or user when no project is set). They never automatically become
explicit user preferences. Outcomes without supporting evidence are labeled
"Reported task ...", rather than claiming verified success.

Events and jobs are inserted together. Workers claim jobs using SQLite write
transactions, expiring leases, and per-claim tokens. Memory mutations and job
acknowledgements commit together. An expired/replaced worker cannot commit changes.
A crash before commit leaves the job available after lease expiry; after commit,
the job is already acknowledged. Processing is at-least-once; effects are protected
by the transaction and stable IDs. Provider calls may repeat after a crash.

`MaintenancePolicy` controls polling, provider/job timeouts, lease duration, retry
backoff, attempts, and dream threshold. Lease duration must exceed job timeout.
Shutdown stops claiming jobs and awaits the bounded current job. Failed jobs remain
inspectable. Worker-loop failures are also exposed as `worker.last_error`.

The default policy applies new observations and exact duplicates automatically.
Semantic changes become typed proposals. `review_job` accepts/rejects a proposal;
acceptance checks the captured generation. `retry_job` resets a failed, proposed,
or rejected job for fresh reasoning and discards previous approval. Running/done
jobs cannot be reset. Stale approved work is never silently applied.

Dream jobs are enqueued within the memory-write transaction when the write threshold
is reached. Planning and application run off the foreground path. Exact duplicate
consolidation can be automatic; other proposals follow the same review policy.
Recommendations for embeddings/indexes remain advisory; no automatic model changes.

Use `maintenance_status` with status/limit/offset for owner-scoped inspection.
`worker.run_once(owner_id=...)` processes one available job; omit owner only in a
trusted worker process authorized to serve the entire database. `background=False`
lets an application run this from its existing scheduler. Multiple processes may
open the same database with consistent embedding and maintenance configuration.

## Tools share the Python dispatcher

```python
from memory_core.runtime import MemoryTools

tools = MemoryTools(memory, owner_id=authenticated_owner)
# Register input_schema/output_schema with your SDK or transport.
response = await tools.invoke(
    {
        "operation": "require_knowledge",
        "task_id": "task-123",
        "gap": {"id": "policy", "question": "What is the project policy?"},
    }
)
```

The tool adapter is transport-neutral and includes every typed core and lifecycle
operation in its discriminated schemas. The host binds identity; callers cannot
supply an owner in tool arguments. Only expose mutation/review operations to a
principal authorized for them. An MCP or SDK transport must register this adapter
and its hooks; no vendor SDK is monkey-patched or contacted automatically.

## Connect a reasoning model once

`StructuredReasoner` supplies conservative operation instructions, typed input,
output JSON schemas, and validated parsing for reconciliation, dreaming, and
knowledge evaluation. Implement one bridge to your chosen model client:

```python
from memory_core.runtime.reasoning import StructuredReasoner


async def generate(request):
    # Your client consumes request.instructions, request.input, and
    # request.output_schema. Return its JSON text or parsed JSON object.
    return await your_structured_model_call(request)


reasoner = StructuredReasoner(generate)
memory = MemoryCore.local(
    "memory.db",
    embedding_spec=embedder.spec,
    embedder=embedder,
    reconciler=reasoner.reconcile,
    dream_planner=reasoner.dream,
    knowledge_evaluator=reasoner.evaluate,
    researcher=researcher,
)
```

The bridge does not grant research access or authorize changes. The researcher
still supplies permitted sources; maintenance policy and revision checks still
control application. A generated answer with an invented evidence reference is
rejected. Instructions reduce mistakes but are not a semantic correctness guarantee;
evaluate your actual model on representative task fixtures before broad use.

## Run as a local MCP server

The optional `memory_core.mcpapp` module provides a real MCP stdio server. It exposes
one tool per typed operation, all routed through `MemoryCore.call`. Tool schemas
are generated from the existing Pydantic models and contain only the referenced
definitions needed by that operation. Tool annotations distinguish reads,
mutations, destructive operations, and idempotent event capture.

Install the optional transport:

```sh
uv sync --extra mcp --extra local
```

Create a trusted local factory, for example `my_memory.py`:

```python
from memory_core import MemoryCore
from memory_core.memory.embedding import LocalSentenceEmbedder
from memory_core.runtime import MaintenancePolicy


def create_memory():
    embedder = LocalSentenceEmbedder(
        "./models/my-embedding-model",
        model_id="my-embedding-model",
        revision="your-model-revision",
        dimensions=384,  # Must match your actual model.
    )
    return MemoryCore.local(
        "./memory.db",
        embedding_spec=embedder.spec,
        embedder=embedder,
        maintenance=MaintenancePolicy(dream_after_writes=100),
        # Add researcher=..., knowledge_evaluator=...,
        # reconciler=..., dream_planner=... for semantic reasoning.
    )
```

The factory may be synchronous or asynchronous. It must return `MemoryCore`. The
model directory must already exist; this example never downloads a model. Without
research/evaluation providers, unanswered gaps remain explicitly unresolved.

Launch from the directory containing your factory:

```sh
uv run --extra mcp --extra local python -m memory_core.mcpapp \
  --factory my_memory:create_memory --owner alice
```

Example MCP client configuration (use absolute project paths in your client):

```json
{
  "mcpServers": {
    "memory": {
      "command": "uv",
      "args": [
        "run", "--directory", "/absolute/path/to/project",
        "--extra", "mcp", "--extra", "local",
        "python", "-m", "memory_core.mcpapp",
        "--factory", "my_memory:create_memory", "--owner", "alice"
      ]
    }
  }
}
```

The owner is bound by the process configuration, never tool arguments. Use one
local process per owner. This is not remote authentication: do not expose a fixed
owner server to unauthenticated HTTP clients. No HTTP listener is started here.
The factory path is trusted configuration and can load Python code; tools cannot
change it. Keep factory/provider output on stderr so stdout remains the protocol
stream.

The server starts and stops the configured maintenance worker with its lifecycle.
The CLI closes the database on exit. For embedding in an existing authenticated
application, use `create_server(memory, owner_id=...)`; the application retains
ownership of the database.

To restrict the exposed operations, repeat `--allow`:

```sh
uv run --extra mcp python -m memory_core.mcpapp \
  --factory my_memory:create_memory --owner alice \
  --allow describe_schema --allow recall --allow resolve_prefs
```

The allowlist is enforced on execution as well as discovery. Sending a different
`operation` through an allowed tool cannot bypass it. Validation/domain errors
return an `is_error` tool result with a typed `failure` envelope. Validation errors
include field locations and error types, not a copy of submitted payloads.

A client can now call `begin_task`, `prepare_task` with `guard=true`,
`require_knowledge`, `record_finding`, and `complete_task` directly. MCP discovery
makes these tools available; the host must still invoke the task hooks at its
actual action boundary to guarantee that an agent uses memory before acting.

The transport uses the official MCP Python SDK 2.x low-level server API:
<https://py.sdk.modelcontextprotocol.io/advanced/low-level-server/>.

## Claude Agent SDK

The optional [Claude Agent SDK adapter](claude-agent-sdk.md) connects these task
hooks automatically to SDK runs, with bounded admission, isolated clients,
task-bound in-process tools, and explicit structured outcome capture.
