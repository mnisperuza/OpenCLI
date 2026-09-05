# Changelog

All notable user-facing changes are documented here. Fenrir Agent follows semantic
versioning for released versions.

## 2.0.0

- Reworked deep web research into a bounded multi-angle pass over core facts,
  primary sources, independent evidence, and limitations. Evidence packets now
  report coverage and unresolved research angles alongside citation-ready
  excerpts.
- Made the session's `/search deep` setting apply to explicit automatic web
  grounding, while preserving the compact fast-search workflow.
- Hardened Gemini native tool continuations and malformed tool-call recovery
  without changing the wire format used by other API providers.
- Added Codex's native OS sandbox as Fenrir's default sandbox backend while
  keeping sandbox activation off at startup.
- Added `/sandbox on` and `/sandbox codex`; retained Docker and E2B as explicit
  alternatives with no host-shell fallback.
- Added built-in read-only/workspace-write profiles, network-off reporting,
  elevated-to-unelevated Windows setup fallback, sanitized command
  environments, and workspace-relative cwd validation.
- Renamed the product from OpenCLI to **Fenrir Agent**.
- Renamed the install distribution to `fenrir-agent`, the Python package to
  `fenrir_agent`, and the terminal command to `fenrir`.
- Moved workspace state and environment-variable names from `.opencli` /
  `OPENCLI_*` to `.fenrir` / `FENRIR_*`. Existing `.opencli` state is left
  untouched; copy the data you want to retain into `.fenrir`.
- Moved the canonical GitHub repository, installer sources, and package links
  to `mnisperuza/Fenrir-Agent`.

## 1.7.0

- Hardened ReAct execution, workspace recovery, verification, session memory,
  and provider reliability controls.
- Added Qwen Cloud OpenAI-compatible API support with regional endpoint
  configuration.
- Made Escape stop active generation and model loading reliably.
- Added FreeLLMAPI, LiteLLM, DS2API, Cerebras, Mistral AI, Fireworks AI, and
  Together AI provider integrations through Fenrir Agent's shared API workflow.
- Added OpenAI, DeepSeek, xAI, and NVIDIA NIM direct-provider integrations
  with model discovery, streamed chat, and native tool-call support.
- Restored llama.cpp as the local-first default model workflow.
- Refreshed documentation, packaging checks, and GitHub-source installers in
  preparation for versioned public releases.

## 1.6.0

- Added `curl` and PowerShell `irm` quick-install wrappers backed by isolated
  `uv` tool environments.
- Added a stable Python import surface for integrations (superseded by
  `fenrir_agent` in 2.0.0).
- Added macOS installation guidance, Homebrew llama.cpp discovery, Apple
  clipboard support, and macOS continuous-integration coverage.
- Reworked repository documentation for release, contribution, and security
  readiness.

## 1.5.3

- Stabilized model-aware context accounting, session memory, Textual workspace,
  typed harness outcomes, evidence-based verification, sandbox controls, and
  bounded web research.
