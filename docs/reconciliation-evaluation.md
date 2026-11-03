# Evaluating reconciliation decisions

Retrieval quality does not establish whether a reasoner should attach evidence,
replace a record, preserve an exception, or flag a conflict. Evaluate that decision
separately before configuring a reasoning callback for unattended use.

`memory_core.memory.reconciliation_evaluation.evaluate_reconciliation` accepts an
async reconciler and labeled `ReconciliationCase` objects. Each case contains an
incoming memory, supplied neighbors, acceptable typed decisions and a rationale.
The evaluator never opens a database or applies the proposed decision. Provider
callbacks receive deep copies; configuring an external provider can still send
those inputs to that provider.

```python
from memory_core.memory.reconciliation_evaluation import evaluate_reconciliation
from memory_core.runtime.reasoning import StructuredReasoner

reasoner = StructuredReasoner(generate)
report = await evaluate_reconciliation(
    reasoner.reconcile,
    cases,
    timeout_seconds=30,
)
print(report.model_dump_json(indent=2))
```

`generate` is your trusted async callback accepting `ReasoningRequest`. It configures
its provider with `request.instructions`, `request.input` and `request.output_schema`,
then returns a JSON object or JSON text. The existing structured reasoning bridge
validates the result. No provider, model, credential location or network service is
hardcoded into the evaluator.

## Read the report

- `decision_accuracy` requires a complete match with an acceptable decision, including
  target ID and expected revision. A correct action against the wrong memory fails.
- `action_accuracy` measures only the action name. Compare it with decision accuracy
  to detect target/revision mistakes.
- `invalid`, `timeouts` and `errors` remain in the denominator; provider failures
  cannot inflate accuracy by dropping difficult cases.
- Per-case outcomes retain the typed decision or error class, with elapsed time.
  Exception messages and provider response bodies are not included in failure results.
- Evaluation runs sequentially. Its asyncio timeout requires a cancellation-cooperative
  async provider; blocking synchronous code cannot be forcibly stopped. Cancelling
  the evaluation propagates cancellation to its caller.

A schema-valid, incorrect decision is still incorrect. A passing report also does
not bypass the storage engine's ownership, scope, revision or review checks.

## Reproduce the smoke evaluation

```sh
uv run python benchmarks/reconciliation_quality.py
uv run python benchmarks/reconciliation_quality.py --provider my_evaluation:generate
```

The first command uses an **always-create baseline**, with no AI calls or credentials.
The second imports a trusted host module and uses its async `generate` callback.
Run it in an environment where that module and its explicitly configured provider
are available. Imports execute Python code; do not use untrusted module paths.
Use `--fixture /path/to/labels.json` for your own labeled cases and `--timeout` to
bound each cooperative provider call. Scripts emit reports to stdout; the host
chooses where to retain potentially sensitive memory content and decisions.

The committed [fixture](../tests/fixtures/reconciliation.json) covers unrelated claims,
duplicates, paraphrases, uncertainty, explicit corrections, contradictory preferences,
project exceptions, observations versus preferences, embedded instructions, and
selection among distractor targets. The [baseline report](benchmarks/reconciliation-baseline.json)
is a reproducible mechanics check, not evidence of AI reasoning quality. The oracle
used in unit tests returns fixture labels solely to validate scoring and isolation.

These ten curated cases are not a statistically representative or independently
reviewed corpus. Before accepting a real reasoner, add independently labeled cases
from your application, record provider/model/prompt versions alongside reports,
and inspect harmful false merges separately. No universal accuracy threshold is
silently imposed. Live authenticated Claude behavior and real model reconciliation
quality remain unverified until those provider evaluations are actually run.
