<div align="center">

# Fenrir Agent

### A local-first coding and research agent for trusted terminal workspaces.

[![CI](https://github.com/mnisperuza/OpenCLI/actions/workflows/harness-gates.yml/badge.svg)](https://github.com/mnisperuza/OpenCLI/actions/workflows/harness-gates.yml)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache--2.0-0A66C2?logo=apache&logoColor=white)](LICENSE)
[![Platforms](https://img.shields.io/badge/Platforms-Windows%20%7C%20macOS%20%7C%20Linux-4C8BF5)](#platform-setup)
[![Local-first](https://img.shields.io/badge/Inference-Local--first-2E8B57)](#models-without-lock-in)
[![Providers](https://img.shields.io/badge/Providers-15-7B61FF)](#direct-api-providers)
[![Release](https://img.shields.io/badge/Release-2.0.0-0A66C2)](CHANGELOG.md)
[![GitHub last commit](https://img.shields.io/github/last-commit/mnisperuza/OpenCLI?logo=github)](https://github.com/mnisperuza/OpenCLI/commits/main)

[Install](#install) · [First session](#first-session) · [Models](#models-without-lock-in) · [Safety](#designed-for-trusted-workspaces) · [Roadmap](docs/ROADMAP.md) · [Contributing](#development)

</div>

![Fenrir Agent workspace preview](assets/preview.png)

Fenrir Agent is a terminal workspace for getting real work done with an agent while keeping control over the model, workspace, tools, and execution boundary. It runs local GGUF models through llama.cpp by default, connects to hosted providers only when you choose to, and treats permissions, evidence, and recovery as product features—not afterthoughts.

> Fenrir Agent works inside a workspace you trust. It asks before sensitive actions, keeps web and tool content untrusted, and never turns a generated answer into an automatic deployment.

## Start here

```bash
# Install, then enter the repository or folder you want the agent to inspect.
fenrir
```

Choose one path after launch:

| Path | First move | Best for |
|---|---|---|
| Local | Select `/model`; Fenrir Agent starts or connects to llama.cpp. | Private, offline-capable GGUF work. |
| Hosted | Run `/api`, choose a provider, then choose a discovered model. | Fast access to a managed model. |
| Gateway | Run `/api`, choose FreeLLMAPI, LiteLLM, or DS2API. | An existing local or organization-managed route. |

API keys exist only in the current process. Fenrir Agent stores provider/model
profiles, never provider secrets.

## A capable agent with a visible boundary

| What you need | What Fenrir Agent does |
|---|---|
| Keep work local | Starts with a llama.cpp-backed GGUF workflow; sessions, plans, notes, and receipts stay on your machine. |
| Use the model that fits | Switch between local models, direct cloud APIs, or optional gateways without changing the agent workflow. |
| Make changes safely | Gates files, web access, and sandbox execution with explicit permissions and trusted-workspace boundaries. |
| Finish long tasks reliably | Keeps a durable ledger, bounded agent turns, recovery controls, context accounting, and evidence-aware completion. |
| Research without noise | Turns web research into a compact, sourced evidence packet instead of flooding the model context with raw pages. |

## Built for the terminal, not bolted onto it

- **A real workspace UI.** Streaming output, diffs, approvals, plan state, context usage, slash completion, and a classic line CLI when you want less interface.
- **Agentic without being reckless.** Multi-tool ReAct turns, recovery after interruption, repeated-action detection, and hard limits on runaway work.
- **Security that stays legible.** Workspace-scoped file access, explicit web approval, sandbox-only command execution, secret-aware exclusions, and no host-shell fallback.
- **Memory with hygiene.** Local session archives, user notes, task plans, and compact recovery capsules—without treating stale tool errors as future instructions.
- **Model choice without lock-in.** Local llama.cpp first; direct cloud APIs and OpenAI-compatible gateways are opt-in, session-scoped alternatives.

## Install

Fenrir Agent supports Python 3.10–3.12 on Windows, macOS, and Linux. The recommended installers use `uv` and install the current public source from GitHub into an isolated tool environment.

### Quick install

macOS and Linux:

```bash
curl -LsSf https://raw.githubusercontent.com/mnisperuza/OpenCLI/main/scripts/install.sh | sh
```

Windows PowerShell:

```powershell
irm https://raw.githubusercontent.com/mnisperuza/OpenCLI/main/scripts/install.ps1 | iex
```

Run either command again to update your installed tool. If the new command is not visible immediately, open a new terminal so its tool directory is on `PATH`.

### Install with pip

After the first Fenrir Agent release is published, install it from PyPI with
`python -m pip install --upgrade fenrir-agent`. Until then, install the current
public source directly from GitHub:

```bash
python -m pip install --upgrade "git+https://github.com/mnisperuza/OpenCLI.git"
fenrir
```

The Textual workspace is the default. For a simple line-oriented session:

```bash
fenrir --cli
# equivalent module entry point
python -m fenrir_agent --cli
```

### Platform setup

| Platform | Local GGUF setup | Notes |
|---|---|---|
| Windows | `winget install llama.cpp` | Native PowerShell and the Textual workspace are supported. Docker sandboxing needs Docker Desktop. |
| macOS | `brew install llama.cpp` | llama.cpp is the recommended GGUF path; direct model loading can use Apple MPS. |
| Linux | Install llama.cpp and put `llama-server` on `PATH` | Docker sandbox support is available when Docker is installed. |

Point Fenrir Agent at an already-running local server when you prefer to manage it yourself:

```bash
fenrir --llama-cpp-url http://127.0.0.1:8080/v1
```

## First session

```text
$ fenrir
Trust this workspace? [y/N] y

You > inspect this project and tell me how to run its tests
```

Fenrir Agent opens in its local model workflow. The `auto` profile uses a GGUF model through llama.cpp; APIs and gateways are explicit choices through `/api` or `--api start` and never replace the local default.

Try these next:

```text
/model                         # choose a local model
/api                            # choose an API provider and model
/search deep                   # make sourced research the default search mode
/status                         # inspect model, tools, memory, and sandbox state
/plan add Add a regression test # create a visible, persistent task plan
```

Good first prompts are concrete and reviewable:

```text
Explain this repository's test command, then run it in the configured sandbox.
Find the error path for API model selection. Propose a small fix; wait for approval before editing.
Read CHANGELOG.md and summarize only the changes that affect provider behavior.
```

## Models without lock-in

### Local by default

Fenrir Agent connects to a local llama.cpp OpenAI-compatible server, discovers supported local installations, and can start `llama-server` for a chosen GGUF model. Local inference remains the default path even after you configure hosted providers.

### Direct API providers

Use `/api` to select a provider and discover its selectable models. API keys are read from the environment or requested for the current session; Fenrir Agent never saves them in its profile store.

| Provider family | Providers |
|---|---|
| Fast hosted inference | Groq, Cerebras, Fireworks AI, Together AI, NVIDIA NIM |
| General hosted models | OpenAI, DeepSeek, xAI, Gemini, Mistral AI, OpenRouter, Qwen Cloud |
| Local or self-hosted gateways | FreeLLMAPI, LiteLLM, DS2API |

All providers use streamed OpenAI-compatible chat and native tool calls where the selected model supports them.

### FreeLLMAPI: one local gateway for configured free tiers

[FreeLLMAPI](https://github.com/tashfeenahmed/freellmapi) can consolidate the free tiers of upstream providers you configure. Start it locally, add your upstream keys in its dashboard, then give Fenrir Agent only its unified key:

```powershell
$env:FREELLMAPI_API_KEY = "freellmapi-your-unified-key"
# Optional for a different local port or a remote HTTPS deployment:
$env:FREELLMAPI_BASE_URL = "http://127.0.0.1:3001/v1"
```

Choose **FreeLLMAPI Gateway** in `/api`. Fenrir Agent asks it for ready-only models, so the picker excludes models without an enabled upstream key. For steady agent behavior, select an exact logical model; choose `auto` only when cross-model routing is intentional.

### LiteLLM and DS2API

LiteLLM remains available as an optional gateway for authorized cloud or local providers:

```powershell
$env:LITELLM_API_KEY = "sk-opencli"
$env:LITELLM_BASE_URL = "http://127.0.0.1:4000" # optional override
```

DS2API is available for compatibility testing and is intentionally kept separate while its behavior is reviewed:

```powershell
$env:DS2API_API_KEY = "your-config-key"
$env:DS2API_BASE_URL = "http://127.0.0.1:6011/v1" # Docker Compose host mapping
```

Plain HTTP is accepted only for loopback gateways. Remote gateways must use HTTPS.

## Designed for trusted workspaces

Fenrir Agent gives the agent useful tools without granting ambient authority.

| Surface | Boundary |
|---|---|
| Files | Reads, writes, and bounded edits stay inside the trusted workspace; protected and secret-like paths are blocked. |
| Web | Search and fetch require explicit approval; every result is treated as untrusted data. |
| Shell work | Commands run only inside a user-selected Docker or E2B sandbox—never by silently falling back to the host shell. |
| Changes | Mutations carry receipts; recovery, verification, and task-plan completion are evidence-aware. |
| Sessions | Historical archives are text, not executable instructions; raw tool and validation errors are kept out of durable model context. |

Docker sandboxes use ephemeral containers with network disabled, a read-only root filesystem, dropped capabilities, no privilege escalation, and CPU/RAM/PID limits. The workspace mounts read-only unless you grant a write approval.

```text
/sandbox docker python:3.12-slim
!python -V
!!python -m pytest -q
```

E2B sandboxes are explicitly connected or created by you. Fenrir Agent does not create, stop, push, or pull a remote sandbox on the agent’s behalf.

## Research that respects context

`/search fast` returns compact ranked results. `/search deep` builds a bounded evidence packet from general web, news, instant answers, and arXiv; it deduplicates sources, preserves citations, and keeps sourced facts separate from inference and uncertainty.

Deep research is deliberately bounded to six sources and 12,000 characters of evidence. The point is useful research inside an agent run—not unlimited browsing disguised as context.

## Command map

Type `/` in the Textual workspace for filtered command completion. Invalid slash commands are handled locally and do not consume a model turn.

| Area | Commands |
|---|---|
| Help and state | `/help`, `/status`, `/context`, `/usage`, `/prompt-size` |
| Models | `/model`, `/model-add`, `/model-rm`, `/api`, `/api-md`, `/api-del`, `/endserver` |
| Agent | `/tools`, `/tools-on`, `/tools-off`, `/tool-auto on\|off`, `/react on\|off`, `/harness status` |
| Research | `/web on\|off`, `/web always`, `/web ask`, `/search fast\|deep\|status` |
| Workspace | `/pwd`, `/cd PATH`, `/roots`, `/permissions`, `/permissions reset` |
| Plans and memory | `/plan`, `/plan add STEP`, `/memory`, `/remember TEXT`, `/session-name TEXT` |
| Sandboxes | `/sandbox docker [IMAGE]`, `/sandbox e2b connect ID`, `/sandbox push`, `/sandbox pull`, `/sandbox off` |
| Session | `/new`, `/history`, `/clear`, `/exit` |

Use `!<argv>` for a read-only sandbox command and `!!<argv>` for a write-approved one. Commands are parsed as argv, not passed through a host shell.

## Configuration

| Setting | Purpose |
|---|---|
| `.fenrir/config.toml` | Workspace-local model capability overrides; the agent cannot modify it with file tools. |
| `~/.fenrir/sessions/` | Local Markdown session archives scoped by workspace. |
| `FENRIR_LLAMA_CPP_URL` | Override the local llama.cpp endpoint. |
| `FENRIR_LLAMA_CPP_STARTUP_TIMEOUT` | Startup timeout in seconds; default `900`. |
| `GROQ_API_KEY`, `OPENAI_API_KEY`, `CEREBRAS_API_KEY`, `DEEPSEEK_API_KEY`, `XAI_API_KEY`, `NVIDIA_API_KEY`, `GEMINI_API_KEY` | Direct-provider keys, read for the current process only. |
| `MISTRAL_API_KEY`, `FIREWORKS_API_KEY`, `TOGETHER_API_KEY` | Additional direct-provider keys, read for the current process only. |
| `OPENROUTER_API_KEY`, `DASHSCOPE_API_KEY` or `QWEN_API_KEY` | OpenRouter and Qwen Cloud keys. |
| `FREELLMAPI_API_KEY`, `LITELLM_API_KEY`, `DS2API_API_KEY` | Optional gateway keys. |
| `FENRIR_HARNESS_MODE` | `v2` (default) or `legacy` compatibility harness. |

## Development

```bash
git clone https://github.com/mnisperuza/OpenCLI.git
cd OpenCLI
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[enterprise]" "pytest>=8,<10"
python -m compileall -q fenrir_agent
python -m pytest -q
```

The release workflow covers Python 3.10, 3.11, and 3.12 on Ubuntu, Windows, and macOS, including a clean built-wheel smoke test on every platform. Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request, [docs/HARNESS_RELEASE_GATES.md](docs/HARNESS_RELEASE_GATES.md) for agent-harness gates, [docs/RELEASING.md](docs/RELEASING.md) for tag and PyPI setup, and [SECURITY.md](SECURITY.md) for responsible disclosure.

## Scope and support

Fenrir Agent is designed for trusted local development and research workflows. MCP servers, third-party plugins, background cloud agents, messaging gateways, editor integrations, desktop installers, and automatic updates are not yet presented as stable product surface.

Use [GitHub Discussions](https://github.com/mnisperuza/OpenCLI/discussions) for
setup and workflow help. Report reproducible defects through
[GitHub Issues](https://github.com/mnisperuza/OpenCLI/issues) with your Fenrir
Agent version, operating system, Python version, selected provider/model,
command, and a redacted error. Read [SUPPORT.md](SUPPORT.md),
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md), and [SECURITY.md](SECURITY.md) before
participating or reporting a security concern.

## License

Apache License 2.0. See [LICENSE](LICENSE).
