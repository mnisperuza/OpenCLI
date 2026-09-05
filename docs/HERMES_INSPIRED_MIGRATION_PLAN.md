# Hermes-Inspired Fenrir Agent Migration Plan

## Goal

Strengthen Fenrir Agent's local-first coding harness by adapting small, proven
patterns from Hermes Agent. Fenrir Agent keeps its own Textual UI, llama.cpp
integration, provider registry, permission model, and durable run ledger.

This plan is command-first. Capabilities are visible, inspectable, and changed
through slash commands; no new button-heavy workflow is introduced.

## Source Boundary

Hermes Agent is MIT licensed. Prefer independent Fenrir Agent implementations based
on its public architecture. If source is copied or adapted directly, preserve
the relevant copyright and MIT license notice. Do not import its gateway,
desktop, billing, bot, voice, or cloud-platform stack.

## Completed Foundation

| Fenrir Agent capability | Command surface | Status |
| --- | --- | --- |
| FTS5 durable-memory recall | `/memory search QUERY` | Complete |
| Model-accessible trusted memory recall | `search_memory` tool | Complete |
| Capability groups | `/tools`, `/tools enable NAME`, `/tools disable NAME`, `/tools reset` | Complete |
| Portable user/workspace skills | `/skills`, `/skill NAME [TASK]` | Complete |
| Safe turn recovery | `/retry`, `/undo [N]` | Complete |
| Sandbox verification evidence | `/verify`, `/verify status`, `/verify recipes` | Complete |
| Compaction failure cooldown | `/compact status` | Complete |
| Read-only isolated delegation | `/delegate`, `/delegates` | Complete |
| Correct chat/mutation routing | Internal host guard | Complete |
| TUI activity timer teardown safety | Internal | Complete |

Current toolsets: `workspace`, `web`, `memory`, `planning`, `sandbox`, and
`session`. A disabled toolset is removed from the model's tool schema; the
model cannot re-enable it.

## Phase 2 — Portable Skills (Foundation Complete)

**Why now:** Hermes' strongest reusable concept is procedural skills stored as
small `SKILL.md` directories. This gives Fenrir Agent repeatable workflows without
forcing every capability into the system prompt.

### Command design

| Command | Behavior |
| --- | --- |
| `/skills` | List enabled workspace and user skills. |
| `/skills show NAME` | Show metadata and a bounded skill preview. |
| `/skill NAME [TASK]` | Load one named skill for the next request. |
| `/skills enable NAME` | Enable a discovered skill for this workspace. |
| `/skills disable NAME` | Disable it without deleting user files. |
| `/skills reload` | Rescan skill directories. |
| `/skills path` | Show workspace and user skill roots. |

### Implementation shape

- Add `main/skills.py` with a strict `SkillManifest` and bounded loader.
- Read only `SKILL.md` from two roots: workspace `.fenrir/skills/` and user
  `~/.fenrir/skills/`.
- Parse small frontmatter fields: `name`, `description`, `version`, and optional
  `platforms`.
- Treat loaded skill text as untrusted reference material, never as higher
  priority than system, user, or permission policy.
- Limit loaded skill text, reject path escapes, and never execute inline shell
  snippets from a skill.
- Start with user-selected skills. Add `/learn` only after provenance, review,
  and archival controls exist.

### Acceptance gates

- Skills cannot access files outside their root.
- Disabled skills are never injected.
- A malformed skill cannot break the chat session.
- Skill loading has token/context accounting and visible status events.

The safe manual foundation is complete. Autonomous `/learn`, remote skill
browsing, and installation remain intentionally deferred until provenance,
review, and archival controls are implemented.

## Phase 3 — Recovery and Verification

**Status:** Recovery foundation complete.

**Why:** Hermes separates retry, empty-response handling, iteration budgets,
and verification evidence. Fenrir Agent already has the run ledger and permissions
needed to add this cleanly.

### Command design

| Command | Behavior |
| --- | --- |
| `/retry` | Re-run the last safe user turn with the same constraints. |
| `/undo [N]` | Remove the last N conversation turns after confirmation. |
| `/verify` | Run the selected verification recipe for the latest change. |
| `/verify status` | Show verification evidence and unresolved failures. |
| `/compact [FOCUS]` | Keep existing compaction command; add evidence-aware status. |

### Implementation shape

- Add provider-neutral empty-response and malformed-tool-call classifications.
- Keep transport retries separate from agent/tool retries.
- Store retry reason, attempt count, and final outcome in `RunLedger`.
- Add verification recipes for common project signals: changed file inspection,
  syntax check, targeted test, and sandbox command.
- Require explicit user approval before any verification recipe performs writes
  or network access.
- Preserve the last user request and recent evidence when compacting context.
- Add a compaction cooldown after repeated ineffective or failed compactions.

### Acceptance gates

- `/retry` never repeats an uncertain mutation receipt.
- `/undo` never deletes durable memory or project files.
- `/verify` reports evidence, not inferred success.
- Cancellation ends a retry/verification run promptly.

Implemented in this migration:

- `/retry` reconstructs the last real user request, retains selected-skill
  context, and refuses replay while any incomplete run has an uncertain effect.
- `/undo [N]` removes conversation turns only; durable memory and workspace
  files remain untouched.
- `/verify` detects Python, Node, and Rust test recipes and runs them read-only
  through the active Docker or E2B sandbox with hashed evidence.
- Empty model responses produce an actionable retry status instead of a silent
  turn, and repeated ineffective compactions enter a visible cooldown.

Deeper provider-specific transport classification and per-attempt retry
telemetry remain follow-up hardening; they are not required for safe manual
replay and should not be mixed with tool-effect recovery.

## Phase 4 — Sandbox Provider Contract

**Why:** Fenrir Agent already supports Docker and E2B. Hermes' provider-registry
pattern can make those backends consistent without adding every Hermes backend.

### Command design

| Command | Behavior |
| --- | --- |
| `/sandbox status` | Show backend capabilities, sync state, and network policy. |
| `/sandbox backends` | List supported local/cloud backends. |
| `/sandbox docker [IMAGE]` | Select isolated Docker backend. |
| `/sandbox e2b ...` | Keep existing E2B lifecycle commands. |
| `/verify sandbox` | Use the active sandbox for a read-only verification recipe. |

### Implementation shape

- Define a capability-based sandbox provider protocol.
- Normalize availability, network, write-sync, cancellation, and timeout data.
- Retain Docker and E2B first; defer SSH, Modal, Daytona, and Singularity.
- Persist only user-approved connection metadata; never persist cloud secrets in
  run events or memory.

## Phase 5 — Delegation, Only After Stability

**Status:** Safe command-first foundation complete; implemented before Phase 4
by explicit project decision.

**Why later:** Hermes has durable asynchronous delegation, but it adds concurrency,
state recovery, and cancellation complexity.

### Command design

| Command | Behavior |
| --- | --- |
| `/delegate TASK` | Propose an isolated subtask; requires user approval. |
| `/delegates` | Show active, completed, failed, and cancelled work. |
| `/delegate stop ID` | Cancel one delegated task. |

### Constraints

- Each delegate receives an isolated workspace view and bounded budget.
- No delegate receives write access by default.
- Parent run receives only structured result/evidence, not hidden reasoning.
- Do not start this phase until Phases 2–4 pass release gates.

Implemented foundation:

- `/delegate TASK` creates one bounded background job after explicit approval.
- `/delegates [ID]` exposes durable state, final result, and evidence IDs.
- `/delegate stop ID` cooperatively cancels queued or running work.
- Each job receives a secret-filtered disposable workspace snapshot, six ReAct
  steps, no network, and denied write permissions. Snapshot changes never merge
  into host workspace.
- Interrupted jobs become failed records on restart instead of silently
  resuming uncertain work.

Multi-model pools, writable delegates, automatic result injection, and parallel
fan-out remain deferred. They require Phase 4 provider isolation first.

## Deferred Features

- Messaging gateway and social channels.
- Voice, browser automation, image generation, billing, and subscriptions.
- Autonomous cron jobs.
- Full Hermes session database replacement.
- Full-text indexing of every transcript. Revisit after skill and memory usage
  proves the need; current Fenrir Agent FTS memory recall is intentionally small.

## Priority Order

1. Retry, undo, verification, and stronger compaction telemetry. Complete.
2. Read-only isolated delegation foundation. Complete.
3. Capability-based sandbox provider contract.
4. Optional `/learn` with provenance and review controls.
5. Optional cron/MCP integrations after the core remains reliable.

## Release Discipline

- One phase per commit series.
- Add focused regression tests for every new command and failure mode.
- Run command lint, targeted harness tests, and UI tests before merging.
- Keep external capabilities opt-in and visible in `/tools`, `/harness`, or the
  relevant slash command.
