# Harness Release Gates

These are hard gates for releases that modify cognition, control, execution,
evidence, memory, provider, or security behavior. A release fails when any
critical threshold regresses.

| Gate | Required threshold |
|---|---:|
| Typed tool outcome and event contract validity | 100% |
| Unsupported completion rejection in deterministic scenarios | 100% |
| Known local-effect recovery without duplicate execution | 100% |
| Permission, scope, protected-path, and symlink bypasses | 0 |
| Secrets persisted in ledger events or default traces | 0 |
| Compaction required-slot validation | 100% |
| Constraint and evidence retention in golden long tasks | 100% |
| Unbounded retry, model fallback, or ReAct trajectories | 0 |
| Open execution receipts incorrectly reported complete | 0 |
| Full regression suite | 100% pass |

The cross-platform CI matrix runs the enterprise harness scenarios before the
complete suite. Model-matrix evaluations remain provider-specific and must be
attached to a release when supported-provider behavior changes; deterministic
tests cannot substitute for weak/local/API model trajectory evaluations.

Debug bundles are redacted by default. Prompt/output tracing requires explicit
local opt-in. Sensitive artifacts are encrypted when an artifact key is
configured; keys are never persisted in the state database.

