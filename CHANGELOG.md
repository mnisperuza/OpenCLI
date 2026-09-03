# Changelog

All notable user-facing changes are documented here. OpenCLI follows semantic
versioning for released versions.

## 1.7.0

- Hardened ReAct execution, workspace recovery, verification, session memory,
  and provider reliability controls.
- Added Qwen Cloud OpenAI-compatible API support with regional endpoint
  configuration.
- Made Escape stop active generation and model loading reliably.
- Added FreeLLMAPI, LiteLLM, DS2API, Cerebras, Mistral AI, Fireworks AI, and
  Together AI provider integrations through OpenCLI's shared API workflow.
- Added OpenAI, DeepSeek, xAI, and NVIDIA NIM direct-provider integrations
  with model discovery, streamed chat, and native tool-call support.
- Restored llama.cpp as the local-first default model workflow.
- Refreshed documentation, packaging checks, and GitHub-source installers in
  preparation for versioned public releases.

## 1.6.0

- Added `curl` and PowerShell `irm` quick-install wrappers backed by isolated
  `uv` tool environments.
- Added a stable `opencli` Python import surface for integrations.
- Added macOS installation guidance, Homebrew llama.cpp discovery, Apple
  clipboard support, and macOS continuous-integration coverage.
- Reworked repository documentation for release, contribution, and security
  readiness.

## 1.5.3

- Stabilized model-aware context accounting, session memory, Textual workspace,
  typed harness outcomes, evidence-based verification, sandbox controls, and
  bounded web research.
