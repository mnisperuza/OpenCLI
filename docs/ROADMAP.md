# OpenCLI Roadmap

This roadmap describes planned product work. It is not a release guarantee.
Items move into the changelog only after they ship.

## Release foundation

OpenCLI 1.7.0 establishes the release baseline: one package version source,
changelog/version parity, GitHub-source installers, automated tag builds,
wheel/source-distribution validation, checksums, GitHub Release assets, and
PyPI Trusted Publishing support. The remaining release task is repository-owner
setup for the PyPI project and GitHub `pypi` environment, followed by the first
public tag release.

## Provider target: 15 selectable providers

OpenCLI exposes 15 provider choices:

- Groq
- OpenAI
- Cerebras
- DeepSeek
- xAI
- NVIDIA NIM
- Gemini
- OpenRouter
- Qwen Cloud
- Mistral AI
- Fireworks AI
- Together AI
- FreeLLMAPI
- LiteLLM
- DS2API

The 15-provider target is complete while preserving the local-first llama.cpp
workflow. Each direct integration uses environment-key discovery, model-list
discovery, streamed chat, safe error redaction, and shared tool-call handling.

Anthropic is intentionally not part of this four-provider target. It uses a
native Messages API and requires a dedicated adapter, capability model, and
contract tests rather than a misleading generic OpenAI-compatible entry.

## Local sandbox

After release foundation and provider target, OpenCLI will add a local sandbox
backend with read-only and workspace-write modes, network disabled by default,
resource limits, process-tree cancellation, and clear platform-specific
guarantees. Docker and E2B remain optional backends.

## Later product layers

- Headless JSON and stream-JSON execution for scripts and CI.
- Stable MCP client and installable skill contracts.
- Lightweight hosted-only dependency set.
- Desktop app, IDE integrations, remote sessions, and hosted services.
