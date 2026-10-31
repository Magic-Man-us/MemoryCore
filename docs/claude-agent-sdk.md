# Claude Agent SDK integration

`memory_core.integrations.claude` attaches task memory to actual Claude Agent SDK
runs. It is optional: importing `memory_core` does not import the Claude SDK.
Python coordinates the lifecycle; the existing Rust engine performs retrieval.

## Install and run

```sh
uv sync --extra claude --extra local
```

Configure an existing local embedding model, permitted research sources, and a
knowledge evaluator as described in [Automatic task memory](task-lifecycle.md).
Claude authentication is configured through the SDK's normal mechanisms. MemoryCore
does not read, copy, or manage credentials. Using this adapter makes real Claude
calls when you invoke `run`; the optional dependency alone does not.

```python
from claude_agent_sdk import ClaudeAgentOptions

from memory_core.integrations.claude import ClaudeMemoryAgent, ClaudeMemoryPolicy
from memory_core.memory.models import TaskContext
from memory_core.runtime import KnowledgeGap

# `memory` is your configured MemoryCore. Own its lifespan once per application.
async with (
    memory,
    ClaudeMemoryAgent(
        memory,
        policy=ClaudeMemoryPolicy(
            max_concurrent_runs=4,
            max_pending_runs=16,
            run_timeout_seconds=300,
        ),
    ) as agent,
):
    report = await agent.run(
        owner_id=authenticated_user.id,
        context=TaskContext(
            task_id=task.id,
            project_id=project.id,
            attributes={"languages": ["python"], "activities": ["coding"]},
        ),
        objective="Implement this project's HTTP retry policy",
        required_knowledge=[
            KnowledgeGap(
                id="retry-policy",
                question="Which requests does this project permit retrying?",
                requirements=["Find the explicit project policy"],
            )
        ],
        options=ClaudeAgentOptions(
            cwd=project.path,
            allowed_tools=["Read", "Glob", "Grep"],
            max_turns=20,
        ),
    )
    if report.capture_error:
        # Execution may have happened. Retry outcome capture with the same task
        # result instead of automatically repeating the agent's external actions.
        handle_capture_failure(report)
```

The `allowed_tools` example is a permission allowlist, not a security sandbox.
Preserve your host's permission callback, sandbox, and working-directory policy.
The adapter preserves those settings and never switches to bypass-permissions mode.
For a read-only demonstration with an explicit tool set, see
[examples/claude_memory.py](../examples/claude_memory.py). Its `--help` describes the
required local model and project arguments. It opts into bounded read-result
snippets as evidence and makes real Claude calls when executed.

## Organization

| Module | Responsibility |
| --- | --- |
| `agent.py` | Admission limits, isolated SDK clients, streaming, completion, cleanup |
| `binding.py` | Owner/task/project scope, memory tools, evidence, context cache |
| `hooks.py` | Pre-tool guard and optional post-tool evidence extraction |
| `leases.py` | SQLite execution leases and token fencing across processes |
| `interfaces.py` | Small client, message observer, and evidence-extractor protocols |
| `models.py` | Typed policy, bounded task request, run report, errors |

## What happens during a run

1. Validate input and reserve a bounded admission slot. A full queue or an expired
   admission wait raises `AgentCapacityError` before launching an SDK client.
2. Claim an owner/task execution lease in SQLite. Another adapter/process using the
   same database cannot claim that pair until it is released or expires.
3. Start/resume the durable task, resolve host-declared knowledge gaps, and prepare
   context. An unresolved gap is presented to the agent so it can research or ask
   for clarification; it is not silently treated as resolved.
4. Create a new SDK client with a task-bound in-process MCP server and memory hooks.
   Owner, task, and project bindings are omitted from tool input schemas and cannot
   be supplied by the model. The host owns task completion and administrative tools.
5. Inject context into the initial prompt. Before each action, a `PreToolUse` hook
   refreshes/revalidates context and blocks unresolved gaps or preference conflicts.
6. Stream messages to an optional awaited `on_message` callback. No transcript is
   accumulated by this adapter. The callback provides backpressure.
7. Require `ResultMessage.structured_output` to validate as `TaskRunResult`. SDK
   errors, missing output, invented evidence, or unresolved gaps cannot produce a
   successful recorded outcome. A `Stop` event alone is never interpreted as success.
8. Disconnect the client, persist the outcome/findings, and release execution and
   admission capacity. Durable maintenance runs through the shared MemoryCore worker.

The `StructuredOutput` tool is the SDK's result mechanism, so it remains callable
when a gap blocks actions. Final success is separately checked against the task
state. Task-bound memory tools also remain callable for recovery. Other tools are
guarded by default, including shell tools and unknown MCP tools.

`research_tools` is an explicit host-selected set of tools permitted to gather
information while gaps remain unresolved. It defaults to empty. It never grants
SDK permissions: the host's normal permission rules still apply. Prefer the bound
`require_knowledge` tool with configured research/evaluation providers. If you allow
additional research tools, make their observations available through your research
provider before retrying the gap; recording a finding alone does not resolve it.

Existing SDK hooks are preserved. The memory guard returns context on success, not
an `allow` decision. Conflicting disallowed memory-tool rules are rejected rather
than silently overridden. Fresh SDK sessions are required: `resume`,
`continue_conversation`, `fork_session`, and explicit SDK `session_id` are rejected
because another session could contain another owner's context. Reuse durable memory
through task/project context instead. A completed task ID cannot be executed again.

SDK subagents inherit the same parent task scope; they are not separate durable
MemoryCore tasks. Use separate adapter runs with separate task IDs when independent
ownership, knowledge requirements, or outcomes are required.

## Evidence and outcomes

The agent may record findings only using exact evidence objects that this run
received from knowledge-resolution results or a host evidence extractor. The
per-run evidence cache is bounded. When it evicts an old item, later claims citing
that item are rejected; capture useful findings promptly.

An optional async `evidence_extractor(event) -> list[Evidence]` can turn successful
or failed tool events into selected evidence. It is a trusted host callback:
redact/filter output and return only the information appropriate to retain. Nothing
captures complete tool payloads or transcripts automatically. The post-tool hook
returns those evidence objects to the agent so it can cite them exactly. Extraction
failure stops the run from reporting verified success. Use stable evidence IDs and
contents when handling repeated events.

A success outcome without evidence remains a **reported outcome**, as in the core
runtime. Evidence membership establishes provenance, not semantic truth or proof
that every goal was satisfied. Use an application-specific verification step when
you need independently verified outcomes.

`ClaudeRunReport` distinguishes:

| Field | Meaning |
| --- | --- |
| `result` | Validated task outcome and findings, or an explicit failed outcome |
| `sdk_session_id` | SDK result session, when one was returned |
| `execution_error` | SDK/callback/validation failure, if any |
| `capture` | Durable job IDs created by task completion |
| `capture_error` | Outcome capture failed after execution; do not blindly rerun actions |
| `cleanup_error` | Client or lease cleanup failed; expiry still recovers a stale lease |

Cancellation and the overall run timeout disconnect the client and leave unfinished
tasks resumable. Findings already persisted remain. After an interrupted run, inspect
external effects before resuming: neither SDK cancellation nor SQLite can roll back
an external file edit or network mutation.

## Scaling boundaries

- **One runtime per application lifespan.** Run it outside the agent adapter's
  lifespan. Per-task MCP sessions do not start or stop shared maintenance workers.
- **One adapter per event loop.** Active and queued runs are bounded. Multiple tasks
  and owners get separate SDK clients, hooks, MCP bindings, and evidence caches.
- **One execution per owner/task in a shared database.** Token checks fence expired
  workers at hooks and in capture transactions. Leases expire after the configured
  run timeout plus cleanup allowance and a short grace period. Normal exits delete
  the lease; crashed processes recover by expiry.
- **Context reuse.** An unchanged generation/receipt avoids re-embedding the same
  objective. Knowledge freshness, task completion, and gaps are still checked.
  Any owner memory-generation change invalidates retrieval context conservatively.
- **Bounded owner index cache.** `MemoryCore.local(max_cached_owners=16, ...)`
  keeps the most recently used owners in memory. Evicted indexes reload from
  SQLite when needed. This bounds owner count, not total bytes: each resident
  owner still consumes memory proportional to its indexed memories.
- **Bounded context.** Optional supporting hits may be omitted to fit the character
  budget. Mandatory preferences and knowledge are never silently truncated: an
  oversized mandatory context blocks execution. This is a character budget, not a
  tokenizer-specific token guarantee.
- **Bounded memory tool output.** `max_recall_hits` defaults to 10 and is advertised
  in the task's tool schema. `max_tool_output_chars` defaults to 65,536; oversized
  results return an explicit error rather than silently truncating evidence.
- **Bounded callbacks.** Provider callbacks and message observers must cooperate
  with cancellation; Python cannot safely preempt blocking code on the event loop.
  Slow observers apply backpressure instead of creating an unbounded message queue.
- **SQLite is single-host storage.** Shared-file leases coordinate processes on
  that host. They are not a multi-host scheduler or a claim of unlimited write
  throughput. Admission limits are per adapter/process, not a global fleet limit.
  Separate databases do not coordinate task execution. Keep external actions
  idempotent where possible; execution leases are not exactly-once external effects.

Do not make knowledge/reconciliation model calls reacquire the same fully occupied
agent-run pool: that can deadlock. Use separate bounded provider capacity for
background reasoning and task-agent execution.

## Validation and references

The test suite exercises the installed Claude SDK's actual client protocol using a
deterministic transport, plus real in-process MCP calls. It also tests admission,
parallel owners, cancellation, leases, scope enforcement, context reuse, and typed
capture. These tests do not authenticate to Claude or establish live model quality.

Official interfaces used:

- <https://code.claude.com/docs/en/agent-sdk/hooks>
- <https://code.claude.com/docs/en/agent-sdk/structured-outputs>
- <https://github.com/anthropics/claude-agent-sdk-python>

For maintenance continuation, retained history, local diagnostics, and growth
benchmarks, see [Growing a local memory collection](local-memory-growth.md).
