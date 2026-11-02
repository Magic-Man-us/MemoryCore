# Memory context for an existing session

`SessionContext` prepares ordinary task data for an existing Claude, OpenAI, or local-model conversation. It does not create clients, replace system prompts, execute tools, or grant permissions. The application supplies the authenticated owner and trusted project/task attributes.

```python
from memory_core.integrations import SessionContext, SessionContextRequest
from memory_core.memory.models import TaskContext

context = SessionContext(
    memory,
    owner_id=authenticated_owner,
    count_tokens=lambda text: len(tokenizer.encode(text)),
)

# Reuse this adapter and the existing provider conversation across turns.
draft = await context.prepare(SessionContextRequest(
    context=TaskContext(task_id=session_task_id, project_id=project_id),
    query=user_text,             # omit for exact preferences only: no embedding
    max_tokens=2048,             # measured by your supplied tokenizer
    limit=5,
))
prompt = "\n\n".join(part for part in (draft.text, user_text) if part)
```

The budget covers the memory injection, including its framing, rather than the entire conversation. Supply the tokenizer for the chosen model and reserve room separately for system messages, conversation history, user input, and output. Applicable preferences and conflicts are mandatory: an insufficient budget raises `ContextBudgetError`. Optional supporting hits are dropped whole, with `omitted_supporting` reporting their count. Supporting hits already represented by mandatory preferences or another selected memory ID are deduplicated separately in `deduplicated_supporting`; they are not reported as omitted. `used_tokens` reports the supplied counter's measurement; no characters-to-tokens estimate is assumed.

Inject through the conversation's ordinary user-data path. For example, with an **already connected Claude SDK client**:

```python
from claude_agent_sdk import ResultMessage

await client.query(prompt)
async for message in client.receive_response():
    if isinstance(message, ResultMessage) and message.subtype == "success" and not message.is_error:
        context.commit(draft)
```

With an **existing OpenAI Responses client**, keep the conversation's response ID:

```python
response = await client.responses.create(
    model=model,
    previous_response_id=previous_response_id,
    input=[{"role": "user", "content": prompt}],
)
if response.status == "completed":
    context.commit(draft)
previous_response_id = response.id
```

For a local chat model, append `prompt` to the existing user-message history, generate using that same history, and commit after acceptance. Commit conversational turns only after a confirmed successful provider result. Returning from `client.query()` alone proves local submission, not acceptance by the model/session. Do not commit failed or uncertain delivery. The next preparation then replays the unacknowledged snapshot.

`initial` and `updated` responses contain a full current snapshot, including explicit withdrawn ID/revisions for removed or revised preferences and previously delivered supporting memories. An `unchanged` response has empty `text` and zero new tokens. Earlier text may remain in provider history; replacement/withdrawal records update its applicability rather than erase historical messages. The host can change trusted project/context attributes in a later request, and the new snapshot withdraws the old scope's material.

Call `context.reset()` when the provider reports compaction, replaces its context, or resumes a conversation whose injected state is unknown. The next turn reinjects a full snapshot without recreating the provider client or changing persistent memories.

Immediately before an action, the host's tool/action dispatcher can enforce:

```python
await context.before_action(draft)
```

This checks the latest committed preparation's owner/task/context receipt against current memory revisions and refuses unresolved preference conflicts. It is a memory freshness guard, not a substitute for host authentication, tool authorization, or confirmation requirements. For SDKs executing tools during generation, a provider/tool integration must acknowledge verified delivery before its action dispatcher uses this guard; local query submission alone is insufficient. This generic adapter does not infer such delivery from vendor events.

Use distinct `task_id` values for independently running sessions, even when owner and project are shared. Memory knowledge is shared by owner/project; context receipts retain the existing one-current-preparation-per-owner/task contract. Two sessions deliberately sharing a task must re-prepare before acting after the other refreshes it. For durable task outcomes, findings, and knowledge-gap enforcement, continue using `memory.task(...)` and `TaskSession.before_action()`.
