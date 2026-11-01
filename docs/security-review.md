# Security review and fixes — 2026-11-01

Scope: MemoryCore Python/Rust library, task-bound Claude/MCP adapter, dependency
lockfiles, and website build configuration. Review baseline:
`ff8114c32ec9260409a04d746fbc2e9042c1eb0b` (reconciliation evaluation branch).
Tests used disposable local SQLite databases and synthetic users/projects. No live
provider, production service, credential store or other tenant was probed.

## Confirmed code issue: metadata discovery escaped the task's project scope

The task adapter bound recall to its task project, but `list_metadata_values` used
all active records owned by the same user. An agent assigned to project Alpha could
therefore discover values existing only in project Beta. A task without a project
could also discover project-only values. This was a constrained information disclosure
(CWE-200): it crossed the adapter's project boundary, not the authenticated owner
boundary. It required access to the owner's task-bound memory tools.

The regression `test_metadata_discovery_respects_task_project` failed on both task
configurations before the fix, exposing the synthetic `private-beta` value. It passes
after the fix, including hidden-prefix queries, pagination and override attempts.

`ListValuesRequest.scope` now uses the existing discriminated `Scope` type. The
catalog filters eligible records **before** calculating pages. The Claude adapter
supplies and validates the scope, and hides it from agent-controlled MCP arguments.
`UserScope` restricts discovery to global records; `ProjectScope` includes global
records and that project. Trusted host callers can still use `scope=None` for
owner-wide discovery. Field definitions remain an owner-wide schema by design.

## Dependency findings and updates

The audit of the exported Python lockfile reported 16 advisories across four
packages. Dependency presence is confirmed; exploitable use of every affected
function by MemoryCore was not demonstrated. All four packages enter through the
optional `ai` extra, including its FastMCP/OAuth and Google client dependencies.
The core local storage API does not expose an OAuth server.

| Package | Previous | Updated | Advisory count |
| --- | --- | --- | ---: |
| authlib | 1.7.2 | 1.9.1 | 9 |
| dnspython | 2.8.0 | 2.9.1 | 1 |
| joserfc | 1.7.4 | 1.7.7 | 2 |
| pyasn1 | 0.6.4 | 0.6.5 | 4 |
| nanoid (website build dependency) | 3.3.16 | 3.3.20 | 1 |

Python findings included Authlib signature/session-binding issues and parser/resource
exhaustion issues. The website finding was GHSA-2v37-7h3g-55p8 in nanoid's custom
size handling. No application-controlled zero-size custom generator call was found;
this was a vulnerable build dependency, not a demonstrated public website attack.
Updates stay within dependency constraints. No advisories were ignored and no major
website dependency was force-upgraded.

After the updates, pip-audit and npm audit report no known vulnerabilities in their
checked graphs. Cargo audit reports zero vulnerabilities or warnings in 15 locked
Rust dependencies. These are point-in-time advisory results, not proof of absence of
unknown vulnerabilities. Python platform markers are evaluated for the audit runner;
other deployment platforms need their own checks.

## Additional cleanup

The Pages workflow's build job no longer receives deployment write permissions or
OIDC token issuance. Those permissions are assigned only to the deployment job.
The build retains read access needed for checkout and Pages configuration. This is
least-privilege hardening, not a demonstrated workflow exploit. GitHub-hosted Pages
deployment was not executed as part of this review.

## Coverage and remaining boundaries

- **Identity and scope:** owner is supplied by trusted host code, never tool arguments.
  Reviewed task binding, recall, discovery, capture, evidence and queue ownership.
  Targeted MCP scope regressions and the existing isolation/replay tests were run.
- **Untrusted model output:** typed parsing, supplied evidence checks, revision checks
  and review boundaries remain in force. Prompt instructions are not treated as an
  authorization mechanism. Real model resistance to prompt injection remains untested.
- **Queries and serialization:** Bandit 1.9.4 scanned `memory_core`. Its three B608
  alerts in runtime status/queue queries were reviewed as false positives: only fixed
  SQL clauses are concatenated and all owner/status values use bound parameters.
  No suppression was added. Local HNSW snapshots remain trusted database artifacts;
  this review does not make native graph loading safe for untrusted imports.
- **Secrets:** a limited tracked-file scan for private keys, GitHub token formats and
  AWS access-key formats found no matches. This was not an exhaustive secret scan;
  history and external credential stores were not inspected.
- **Dependencies:** pip-audit 2.10.1, npm audit (npm 11.9.0), cargo-audit 0.22.2.
  No private source was uploaded to scanning services. Dependency name/version
  queries were sent to the standard advisory registries.
- **Compatibility:** full Python 3.12/3.13 suites run with frozen lockfile environments
  including `dev`, `claude`, `ann` and `ai`; website production build also verified.
  Hosted-provider authentication and optional AI provider behavior are not validated
  merely by installing their dependencies.

Reproduce dependency checks without installing the entire optional model stack:

```sh
uv export --frozen --all-extras --no-emit-project --no-hashes -o /tmp/memorycore-audit.txt
uv tool run pip-audit==2.10.1 -r /tmp/memorycore-audit.txt --no-deps --disable-pip
# In website/:
npm audit
# At repository root, with cargo-audit installed:
cargo audit
```

The stdio MCP server is for a trusted local host or an already-authenticated adapter.
Do not expose its fixed owner binding as a shared unauthenticated HTTP endpoint.
