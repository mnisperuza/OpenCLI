# Fenrir Agent Enterprise Harness Plan

Status: core implementation complete; advanced orchestration gated, version 3

## Implementation Status

Implemented in the 1.5 development line:

- versioned `RunState`, ledger events, tool outcomes, error taxonomy, manifests,
  execution receipts, memory records, compaction capsules, and completion decisions;
- backwards-compatible SQLite migration, immutable run ledger, cached snapshots,
  writer leases, pause/resume, cooperative cancellation, abandoned-run recovery,
  conservative effect reconciliation, and content-addressed artifacts;
- typed ReAct failure handling, evidence IDs, mutation verification, host completion
  validation, deterministic progress checks, and semantic stagnation scoring;
- trusted memory lineage, supersession/deletion/correction, structured compaction,
  raw-history artifacts, token-pressure hysteresis, and untrusted-memory isolation;
- tool policy/manifests, protected path and symlink controls, injection tagging,
  secret redaction, network destination allowlists, and optional artifact encryption;
- provider capability reports, bounded transport retry with jitter, circuit breaking,
  fallback policy contracts, strict/repaired structured output, and optional constrained
  decoding dependency boundary;
- privacy-first OpenTelemetry facade, redacted debug bundles, deterministic parallel
  read executor, cross-platform CI gates, and enterprise failure/recovery scenarios.

Operational commands:

```text
/harness status
/harness runs
/harness reconcile RUN_ID
/harness resume RUN_ID
/harness debug RUN_ID
/memory records | correct | delete | export
```

Still gated behind measurement or a separately approved product phase: automatic
cross-provider fallback credentials, an Outlines backend adapter, encrypted-artifact
key management UX, isolated subagent execution, alternative LangGraph checkpoints,
and state replay/fork UI. These are not silently enabled by the core migration.

## North Star

Fenrir Agent becomes a deterministic control plane around non-deterministic models.
The model proposes actions; the harness owns state, transitions, policy,
execution, evidence, memory, recovery, and completion.

```text
model proposes -> harness validates -> policy decides -> tool executes
      -> evidence commits -> progress checks -> continue / finish / pause
```

The harness must never infer success from confident prose. Completion requires
typed outcomes and evidence.

## Existing Baseline

Fenrir Agent already has foundations that should be hardened rather than rebuilt:

- explicit ReAct phases, step budgets, repeat detection, and failure limits;
- mandatory `/react on` dispatch and critique transitions;
- SQLite-backed conversation and tool-event persistence using WAL;
- macro conversation compaction and micro tool-result compaction;
- bounded tool-result archives and durable user notes;
- model-aware context accounting and tokenizer support;
- persistent task plans;
- workspace-scoped permissions, protected paths, and sandbox backends;
- streaming UI-neutral agent events.

The next work should migrate these pieces behind stable contracts instead of
adding parallel implementations.

## System Boundaries

```text
Cognition   model proposes a route, action, or bounded public critique
Control     harness validates state, policy, budgets, and transitions
Execution   tool runtime performs an approved operation
Evidence    append-only ledger records what actually occurred
Memory      trusted, scoped facts survive context reduction
Evaluation  traces and scenarios measure outcome and trajectory quality
```

### Model-facing operations

Keep the exposed control surface small:

- `react_dispatch`
- `critique_and_plan`
- `ask_user` when an explicit durable pause is required
- workspace, search, file, command, web, and future domain tools

### Harness-owned transitions

Do not expose these as selectable model tools:

- retry and structured-output repair;
- timeout, cancellation, and circuit breaking;
- fallback model selection;
- checkpointing and resumption;
- stagnation detection and strategy escalation;
- policy enforcement and approval routing;
- completion validation;
- halt and terminal-state selection.

This prevents tool-schema growth from making weaker models less reliable.

## Core Contracts

### Typed run state

Create one versioned `RunState` containing:

- run, session, turn, and current step IDs;
- goal, user constraints, active plan, and success criteria;
- current phase and allowed next transitions;
- model, tool, token, time, and cost budgets;
- active proposal, policy decision, approval, and tool receipt;
- verified facts, evidence IDs, artifacts, and changed resources;
- failure counters, stagnation score, cancellation state, and stop reason;
- memory and compaction checkpoint references.

Models may propose data for this state but may not mutate it directly.

### Typed tool outcome

All tools return one host-owned envelope rather than relying on prose markers:

```text
status: success | partial | denied | retryable_error | fatal_error | cancelled
summary: bounded public description
evidence_ids: stable references
artifact_ids: large or binary output references
changed: whether an external effect occurred
receipt: idempotency/reconciliation data
retry_after: optional provider guidance
error_code: stable machine-readable classification
```

ReAct logic must use the typed status. It must not scan summaries for words
such as “failed,” “error,” or “unchanged.”

### Tool manifest

Every tool registration defines:

- stable name, version, input schema, and output schema;
- capability class: read, write, execute, network, or sensitive;
- risk and approval policy;
- scope resolver and protected-resource rules;
- timeout and cancellation behavior;
- retry classification;
- idempotency and reconciliation semantics;
- output/context limits and artifact behavior;
- audit and redaction policy;
- compensation support, if safe and possible.

## Durable Execution

### Append-only run ledger

Use immutable, schema-versioned events as the source of truth:

```text
run.started
model.requested
model.responded
tool.proposed
policy.decided
approval.requested
approval.decided
tool.started
tool.completed
state.transitioned
memory.compacted
run.paused
run.resumed
run.finished
run.failed
```

Each event includes a monotonic sequence number, run/turn/step/event IDs,
parent event ID, timestamps, model/provider identity, hashes, policy result,
redaction metadata, and relevant evidence or artifact references.

Snapshots are cached projections for fast startup; the event ledger remains
authoritative.

### Atomic effect boundaries

Persist around side effects:

```text
proposal saved -> policy saved -> approval saved -> execution started
      -> effect receipt saved -> observation and state committed
```

Where one database transaction cannot cover the external effect, use an
execution receipt and reconciliation step on resume.

### Execution guarantees

Do not promise universal exactly-once execution:

- reads should be safely repeatable;
- local mutations should be transactional where possible;
- external actions use at-least-once delivery plus reconciliation;
- resumed actions inspect prior receipts before repeating;
- uncertain side effects pause for user review.

### Runtime lifecycle

Support explicit states:

```text
pending -> running -> waiting_approval | waiting_user
        -> cancelling -> cancelled
        -> recovering -> running
        -> completed | failed
```

Add cooperative cancellation, subprocess termination, abandoned-run recovery,
one active writer per workspace/session, and deterministic ordering when reads
eventually run in parallel.

## Memory and Compaction

### Memory layers

1. **Working context**: current goal, hot conversation, active plan, recent
   evidence, pending approvals, and unresolved errors.
2. **Task diary**: append-only decisions, actions, effects, failures, and
   evidence lineage for the current mission.
3. **Project memory**: stable workspace rules, conventions, user-confirmed
   facts, and reusable verified knowledge.

Every durable memory item needs provenance, trust class, namespace, scope,
creation time, optional TTL, sensitivity, source event IDs, and supersession
links. Distinguish user-confirmed facts from model-inferred candidates.

Retrieved repository text, web pages, tool results, historical transcripts,
and model summaries remain untrusted data and never gain instruction priority.

### Structured compaction capsule

A compacted checkpoint preserves these typed slots:

```text
goal
user_constraints
success_criteria
decisions
completed_work
changed_resources
verified_facts
failures_and_rejected_approaches
open_questions
active_plan
next_action
evidence_and_artifact_references
```

Compaction requirements:

- trigger from token pressure with hysteresis, not character count alone;
- preserve a complete recent hot window;
- retain source event ranges and raw archived history;
- validate required slots after model summarization;
- prevent recursive summary drift by referencing authoritative prior evidence;
- fall back to deterministic extractive compaction on summarizer failure;
- support user inspection, correction, export, and deletion.

## ReAct Control Plane V2

```text
dispatch -> plan -> act -> observe -> progress_check
         -> critique when needed -> continue | finish | ask_user | halt
```

### `react_dispatch`

Typed fields:

- `decision`: `answer | act | ask_user`;
- concrete goal and optional workspace scope for `act`;
- requested budget, capped by host policy;
- concise public route summary;
- declared risk hint, treated only as a model suggestion.

The harness computes actual risk and permissions.

### `critique_and_plan`

Typed fields:

- verified progress;
- stable evidence IDs, not copied evidence prose;
- blocker and failure classification;
- confidence calibrated to evidence coverage;
- next intended action;
- terminal proposal: `continue | finish | ask_user | halt`.

The harness validates the proposed transition and rejects unsupported
completion.

### Adaptive critique

Run a cheap deterministic progress check after every action. Require a model
critique only when:

- a tool fails or returns partial/ambiguous evidence;
- a mutation occurs;
- the plan crosses a milestone;
- semantic repetition or stagnation rises;
- the model changes strategy;
- completion is proposed.

This retains self-reflection without doubling every trivial read operation.
Never request or persist private chain-of-thought.

### Stagnation detection

Detect more than identical tool arguments:

- semantically equivalent actions with cosmetic argument changes;
- repeated reads with no new evidence;
- alternating failure/success cycles that do not advance the goal;
- plan steps repeatedly reopened;
- critiques that claim progress without new evidence;
- context compactions that lose the current objective.

Escalation order:

```text
repair arguments -> choose different evidence source -> re-plan
                 -> safe model fallback -> ask user -> halt
```

### Completion validator

Completion is a host transition, not a persuasive model answer. Validate:

- all required success criteria;
- evidence coverage for claimed work;
- expected mutation receipts;
- targeted verification after changes;
- no pending approval, tool call, or unresolved fatal error;
- clear disclosure of skipped or unverified work.

## Tool Runtime

Every invocation passes through one middleware pipeline:

```text
schema -> scope -> injection defense -> permission -> budget -> approval
       -> checkpoint -> timeout/cancel -> execute -> redact -> archive
       -> evidence -> state transition
```

### Evidence tools

- scoped workspace inspection with ignore, depth, and result budgets;
- line-range file reads with content hashes;
- literal/regex code search with stable match IDs;
- Git status/diff evidence separated from model narration;
- bounded test/lint/check execution with normalized exit status;
- artifact inspection by bounded range.

### Mutation tools

- proposal/preview separated from execution for risky changes;
- pre-image and post-image hashes;
- atomic replacement where supported;
- patch-conflict detection;
- mandatory targeted verification after mutation;
- compensation only for effects owned by the run and known reversible.

A generic “revert anything” tool must not exist. Reversal must be scoped,
evidence-backed, and permission checked.

### Content-addressed artifacts

Large or binary results become artifacts addressed by a content hash. Store:

- artifact ID, media type, size, origin, and creating event;
- full/partial/truncated status;
- sensitivity and redaction state;
- retention and deletion policy;
- bounded excerpts for model context.

## Security Control Plane

Add explicit controls for:

- prompt injection in repositories, web results, logs, and memory;
- secret detection and redaction before provider transmission;
- network egress allowlists and destination disclosure;
- symlink, path traversal, and path-race protection;
- sanitized subprocess environment and argv execution;
- sensitive artifact encryption where configured;
- hashed approval and execution receipts;
- dependency, plugin, MCP server, and tool provenance;
- workspace-scoped least-privilege grants;
- memory poisoning, contradiction, and unauthorized persistence;
- audit retention, export, and deletion controls.

## Provider Adaptation

Maintain declared capability profiles, then verify unstable capabilities with
cached runtime probes:

- native and parallel tool calling;
- strict JSON Schema support;
- grammar-constrained decoding;
- streaming tool-argument correctness;
- actual context/output limits and tokenizer availability;
- reasoning and vision input support;
- cancellation behavior;
- malformed-response and transient-failure rates;
- rate-limit and `Retry-After` handling.

Structured-output ladder:

```text
provider-native strict schema
-> backend-native grammar (for example llama.cpp)
-> optional Outlines adapter for compatible local backends
-> bounded parse/repair fallback
-> ask user or halt
```

Add transport retry with jitter, provider circuit breakers, bounded model
fallback chains, and policy checks before sensitive context reaches a fallback.

## Observability and Evaluation

Emit OpenTelemetry-compatible spans for model calls, ReAct transitions, policy
checks, approvals, tools, artifacts, compaction, recovery, and cancellation.
Centralize semantic attribute mapping because GenAI conventions can evolve.

Redact prompts, arguments, outputs, and secrets by default. Full-content tracing
must be an explicit local opt-in.

### Evaluation suites

- deterministic unit/property tests for state transitions and schemas;
- golden task scenarios with expected evidence and final state;
- trajectory tests for tool selection, arguments, and ordering;
- model matrix: weak local, llama.cpp, API chat models, and reasoning models;
- malformed tool call and structured-output repair tests;
- compaction fidelity and memory-poisoning tests;
- crash tests at every side-effect boundary;
- timeout, cancellation, denial, provider outage, and disk-full tests;
- prompt-injection and secret-exfiltration adversarial tests;
- recovery/resume and duplicate-effect tests.

### Initial release metrics

- task completion with evidence coverage;
- dispatch and tool-argument validity;
- unsupported completion rejection rate;
- stalled/repeated-action rate;
- repair and fallback success rate;
- compaction fidelity and constraint retention;
- crash recovery and resume success;
- duplicate/uncertain external effects;
- permission bypass and secret leakage rate;
- latency, tokens, and cost per completed task.

Define thresholds by task class and provider. Never combine all quality into one
opaque score.

## Library Decisions

### Use existing capabilities first

- Continue using Pydantic/PydanticAI schemas and validation.
- Continue using standard `sqlite3` while access is small and synchronous.
- Continue using existing tokenizers and context accounting.

### Add when its phase begins

- `tenacity`: transport retry/backoff only, with separate retry budgets from
  agent repair and model fallback;
- OpenTelemetry API/SDK/exporters: portable traces and metrics;
- `pydantic-evals`: code-first scenario and trajectory evaluation.

### Add only after measurement

- `aiosqlite`: if SQLite work demonstrably blocks the async/TUI runtime;
- `orjson`: if event/artifact serialization is a measured bottleneck;
- `structlog`: if it materially improves the logging pipeline beyond standard
  logging plus OpenTelemetry.

### Optional adapters

- Outlines: constrained generation for compatible local inference backends;
- backend-native grammar/schema implementations when available.

### Defer behind an interface and decision gate

- LangGraph: compare against Fenrir Agent's checkpoint protocol after durable-state
  contracts exist. Adopt only if pause/resume, pending writes, replay, forks,
  or operational maintenance are clearly better than the custom runtime.
- LlamaIndex ReActAgent: experimental compatibility backend only; do not make it
  the primary harness.
- Instructor: unnecessary by default while PydanticAI already owns structured
  validation; reconsider only for unsupported provider clients.

## Implementation Phases

### Phase 1 — Normalize contracts

- typed tool outcomes and stable error taxonomy;
- versioned `RunState`, events, IDs, and tool manifests;
- replace summary-string failure detection;
- migration compatibility for existing SQLite sessions.

Exit gate: existing behavior passes through typed contracts with no regression.

### Phase 2 — Durable execution

- append-only ledger and snapshot projector;
- atomic state/event commits and execution receipts;
- pause/resume, cancellation, recovery, and reconciliation;
- approval state integrated with checkpoints.

Exit gate: crash-injection tests recover without duplicate known effects.

### Phase 3 — Memory and compaction V2

- trusted memory records with lineage and lifecycle;
- structured compaction capsules and validators;
- contradiction/supersession handling;
- user inspect, correct, export, and delete operations.

Exit gate: long-task evals retain constraints and evidence across compactions.

### Phase 4 — ReAct V2

- evidence IDs and host completion validator;
- adaptive critique and deterministic progress checks;
- semantic stagnation detection and escalation policy;
- safe structured-output ladder for weak/local models.

Exit gate: target models complete scenario suites within bounded steps.

### Phase 5 — Tool and security platform

- middleware pipeline, artifacts, redaction, and capability grants;
- mutation receipts and targeted verification;
- injection, memory-poisoning, egress, and secret controls.

Exit gate: adversarial suites show no critical policy bypass.

### Phase 6 — Provider reliability

- capability probes, transport retry, circuit breakers, and fallback rules;
- normalized streaming and cancellation behavior;
- provider/model compatibility reporting.

Exit gate: supported providers meet defined reliability thresholds.

### Phase 7 — Observability and release gates

- OpenTelemetry spans and privacy controls;
- Pydantic Evals datasets and trajectory evaluators;
- regression dashboards and CI quality gates;
- documented SLOs and debug-bundle export.

Exit gate: release cannot proceed when critical harness metrics regress.

### Phase 8 — Advanced orchestration

- deterministic parallel reads;
- isolated subagents with scoped capabilities and hierarchical budgets;
- optional alternative checkpoint/orchestration backend;
- replay and safe state branching for debugging.

Exit gate: advanced execution cannot weaken baseline permissions or durability.

## Enterprise Definition of Done

The harness is enterprise-ready only when it can demonstrate:

- bounded, inspectable behavior across supported model classes;
- durable recovery from interruption and process failure;
- evidence-backed completion rather than prose-backed completion;
- least-privilege execution and durable human approval;
- auditable memory provenance and compaction lineage;
- privacy-aware traces and artifacts;
- measured reliability enforced by release gates;
- backwards-compatible migrations and documented recovery procedures.
