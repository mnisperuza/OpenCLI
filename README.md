# OpenCLI

**A local-first coding and research agent for trusted terminal workspaces.**

OpenCLI combines local GGUF inference through llama.cpp with optional hosted models, permission-aware tools, durable sessions, and an evidence-conscious agent harness. It is for people who want a capable terminal agent without giving up control of their workspace, context, or execution boundary.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/) [![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE) [![CI](https://github.com/mnisperuza/OpenCLI/actions/workflows/harness-gates.yml/badge.svg)](https://github.com/mnisperuza/OpenCLI/actions/workflows/harness-gates.yml)

> OpenCLI is an agent harness, not an autonomous deployment platform. It works in a workspace you trust, asks before sensitive actions, and keeps web/tool content as untrusted data. Review generated changes before shipping them.

![OpenCLI workspace preview](assets/preview.png)

## Why OpenCLI

- **Bring your model.** Use a local GGUF model with llama.cpp, or connect Groq, Gemini, OpenRouter, or Qwen Cloud for a session without storing API keys.
- **Useful, not reckless.** File access, edits, web use, and sandbox commands are permission-gated and constrained to the trusted workspace.
- **Real agent execution.** Natural multi-tool ReAct turns, evidence-backed completion, bounded retries, recovery after interruption, and clear plans.
- **Research without context bloat.** Fast search returns compact top results; deep research gathers and compresses diverse web, news, instant-answer, and arXiv evidence with source provenance.
- **Local, durable control.** Sessions, notes, plans, and run receipts stay local. Old tool errors are excluded from future memory context.
- **A terminal that respects attention.** A Textual workspace with streaming, approvals, diffs, plan status, context accounting, slash completion, and a classic line CLI when you prefer it.

## Install

OpenCLI supports Python 3.10 through 3.12 on macOS, Linux, and Windows.

### Quick install

macOS and Linux:

```bash
curl -LsSf https://raw.githubusercontent.com/mnisperuza/OpenCLI/main/scripts/install.sh | sh
```

Windows PowerShell:

```powershell
irm https://raw.githubusercontent.com/mnisperuza/OpenCLI/main/scripts/install.ps1 | iex
```

These wrappers install OpenCLI into an isolated tool environment with `uv`.
They install `uv` first when it is unavailable and can be run again to upgrade
an existing OpenCLI installation.

### Install with pip

```bash
python -m pip install --upgrade pip
python -m pip install opencli
opencli
```

The Textual workspace is the default. Use the traditional line interface when you need a simple terminal session:

```bash
opencli --cli
# equivalent module entry point
python -m opencli --cli
```

### Platform setup

| Platform | Local GGUF setup | Notes |
|---|---|---|
| macOS (Apple Silicon or Intel) | `brew install llama.cpp` | Apple Silicon uses PyTorch MPS when direct model loading is selected. llama.cpp is recommended for GGUF. |
| Linux | Install llama.cpp and put `llama-server` on `PATH` | Docker sandbox support is available when Docker is installed. |
| Windows | `winget install llama.cpp` | Native PowerShell and the Textual workspace are supported. Docker requires Docker Desktop. |

For a local model, start OpenCLI and choose `/model`, or let it start a local `llama-server`. You can also point at an existing compatible server:

```bash
opencli --llama-cpp-url http://127.0.0.1:8080/v1
```

Optional installs:

```bash
# E2B sandbox integration
python -m pip install "opencli[sandbox]"

# Full optional acceleration/sandbox set (platform-aware)
python -m pip install "opencli[full]"

# Contributor/release checks
python -m pip install "opencli[enterprise]" "pytest>=8,<10"
```

## First session

```text
$ opencli
Trust this workspace? [y/N] y

You > inspect this project and tell me how to run its tests
```

OpenCLI asks for workspace trust before enabling agent tools. It then asks for each sensitive capability unless you choose a session- or workspace-level approval. API keys are read from the process environment or requested for the current session; they are never saved in the profile store.

For Qwen Cloud, set `DASHSCOPE_API_KEY` (or `QWEN_API_KEY`) and choose **Qwen Cloud (Alibaba Model Studio)** from `/api`. International accounts default to Alibaba's Singapore-compatible endpoint. Set `QWEN_BASE_URL` to the OpenAI-compatible URL issued for your region or workspace when needed; API key and endpoint regions must match.

Try these next:

```text
/model                         # choose a local model
/api                            # connect Groq, Gemini, OpenRouter, or Qwen Cloud
/search deep                   # make research the default search depth
/status                         # inspect model, tools, memory, and sandbox
/plan add Add a regression test # create a visible task plan
```

## What it can do

### Model and provider support

| Capability | Details |
|---|---|
| Local inference | GGUF models through a local llama.cpp OpenAI-compatible server; automatic discovery on `PATH`, Homebrew macOS locations, and Windows WinGet locations. |
| Hosted inference | Groq, Gemini, OpenRouter, and Qwen Cloud with streamed chat and native tool calls where a provider supports them. |
| Model profiles | Context window, output reserve, tools, vision, and reasoning capabilities are model-aware and can be overridden per workspace. |
| Media | Bounded image normalization and supported system-clipboard image attachment on Windows and macOS. |
| Context | Prompt accounting, tool-result pruning, hot-turn retention, and structured compaction before context exhaustion. |

### Agent harness

OpenCLI’s ReAct runtime lets a model make ordinary tool calls and multiple independent calls in one response. The host keeps the system reliable without forcing a rigid synthetic loop:

- 24 model-call budget and an absolute 20 tool-step ceiling per request.
- Warning-first repeated-action and failure detection for exploratory work.
- Tool result classification before final-answer handling, so executable calls are not lost to a premature final response.
- Typed outcomes, append-only run ledger, mutation receipts, crash reconciliation, and resume/recover controls.
- Evidence requirements for task-plan completion and mutation claims.
- ReAct coaching can be disabled with `/react off`; safety, permissions, and runaway limits remain active.

OpenCLI does not request, expose, or persist private chain-of-thought. It shows only user-useful progress, tool state, and concise reasoning summaries.

### Workspace tools

| Tool family | What OpenCLI can do | Guardrail |
|---|---|---|
| Files | List, read, search, inspect, create directories, write, and make bounded text edits | Paths remain inside the trusted workspace; protected and secret-like paths are blocked. |
| Web | Search and fetch public sources | Explicit web approval; pages and search results are untrusted data. |
| Plans | Create, update, and inspect persistent task plans | Completion and dismissal are evidence-aware. |
| Sandboxes | Run argv commands in Docker or a connected E2B sandbox | No host-shell fallback; lifecycle and sync stay user-controlled. |
| Memory | Save user notes, resume sessions, import one compact historical capsule | Raw tool/validation error payloads are excluded from durable model context. |
| Delegation | Run bounded work against a disposable isolated snapshot | Output is returned as data and never silently merges workspace changes. |

### Web search and deep research

`/search fast` is the default: compact, ranked top results for a quick answer. `/search deep` builds a bounded evidence packet rather than pouring raw pages into the prompt:

| Source lane | Role in deep research |
|---|---|
| General web | Breadth and coverage |
| News | Recent or breaking developments |
| Instant answers | Fast, precise factual checks |
| arXiv | Academic and citable research; clearly labeled as preprints |

OpenCLI deduplicates sources, selects diverse pages where useful, preserves citations and excerpts, and asks the model to separate sourced facts, inference, uncertainty, and disagreement. Deep research is bounded to six sources and 12,000 characters of evidence; it is deliberately not unlimited browsing.

## Commands

Type `/` in the Textual workspace for filtered command completion. Invalid or mistyped slash commands are handled locally with the correct usage; they do not consume a model turn.

| Area | Commands |
|---|---|
| Help and state | `/help`, `/status`, `/context`, `/usage`, `/prompt-size` |
| Models | `/model`, `/model-add`, `/model-rm`, `/api`, `/api-md`, `/api-del`, `/endserver` |
| Agent | `/tools`, `/tools-on`, `/tools-off`, `/tool-auto on\|off`, `/react on\|off`, `/harness status` |
| Research | `/web on\|off`, `/web always`, `/web ask`, `/search fast\|deep\|status` |
| Workspace | `/pwd`, `/cd PATH`, `/roots`, `/permissions`, `/permissions reset` |
| Plans | `/plan`, `/plan add STEP`, `/plan set ID STATUS`, `/plan clear` |
| Memory | `/memory`, `/memory notes`, `/memory clear`, `/memory export`, `/remember TEXT`, `/session-name TEXT` |
| Sandboxes | `/sandbox docker [IMAGE]`, `/sandbox e2b connect ID`, `/sandbox push`, `/sandbox pull`, `/sandbox off` |
| Session | `/new`, `/history`, `/clear`, `/exit` |

Use `!<argv>` for a read-only sandbox command and `!!<argv>` for a write-approved one after selecting a sandbox. Commands are parsed as argv, not sent to a host shell.

## Docker and E2B

Docker uses ephemeral containers with network disabled, a read-only root filesystem, dropped capabilities, no privilege escalation, and CPU/RAM/PID limits. The workspace mounts read-only unless a write approval is granted.

```text
/sandbox docker python:3.12-slim
!python -V
!!python -m pytest -q
```

E2B is explicitly connected or created by the user. OpenCLI does not create, connect, stop, push, or pull an E2B sandbox on the agent’s behalf. Transfers are bounded, exclude secrets and environments, and reject local conflicts.

```text
/sandbox e2b connect YOUR_SANDBOX_ID
/sandbox push
!python -m pytest -q
/sandbox pull
```

## Configuration and data

| Location | Purpose |
|---|---|
| `.opencli/config.toml` | Workspace-local model capability overrides. The agent cannot modify it through file tools. |
| `~/.opencli/sessions/` | Local Markdown session archives scoped by workspace. |
| `~/.opencli/errors.log` | Local diagnostic log, never loaded as model memory. |
| `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` | Optional Hugging Face token for the current process. |
| `E2B_API_KEY` | Required only when using E2B. |
| `OPENCLI_LLAMA_CPP_URL` | Override local llama.cpp server URL. |
| `OPENCLI_LLAMA_CPP_STARTUP_TIMEOUT` | Startup timeout in seconds; default 900. |
| `OPENCLI_LLAMA_CPP_DOWNLOAD_FAILURE_LIMIT` | Explicit llama.cpp Hugging Face download failures tolerated before startup stops; default 1. |
| `OPENCLI_HARNESS_MODE` | `v2` (default) or `legacy` compatibility harness. |

Sessions are historical text, never executable instructions. Tool results are pruned after consumption; a bounded local archive preserves useful results without repeatedly burning model context. `/compact` creates a structured summary using the active model and retains recent complete turns.

## Python library

The CLI is OpenCLI’s primary interface. For integrations, import from the stable `opencli` namespace rather than the internal `main` package:

```python
from opencli import OpenCLI, OpenCLIEngine, __version__

app = OpenCLI(dry_run=True)
print(__version__)
```

`opencli` also exports the backend, tool-provider, permission, session, and sandbox contracts for typed integrations. The UI and model runtime load lazily so importing the package does not immediately initialize a model.

## Development and releases

```bash
git clone https://github.com/mnisperuza/OpenCLI.git
cd OpenCLI
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[enterprise]" "pytest>=8,<10"
python -m compileall -q main opencli
python -m pytest -q
```

The release gate covers Python 3.10, 3.11, and 3.12 on Ubuntu, Windows, and macOS. Full regression runs on Ubuntu. Changes to harness behavior must also pass the evidence, permission, recovery, redaction, and compaction gates in [docs/HARNESS_RELEASE_GATES.md](docs/HARNESS_RELEASE_GATES.md).

Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request and [SECURITY.md](SECURITY.md) for responsible disclosure.

## Current scope

OpenCLI is ready for trusted local development and research workflows. The following are intentionally not presented as stable product surface yet:

- MCP servers and third-party plugins.
- Background cloud agents, messaging gateways, and editor integrations.
- A desktop installer or automatic update service.
- Guaranteed tool calling from every local model; weak models may need `/tool-auto on` for deterministic routing.

## Support

Open an issue at [GitHub Issues](https://github.com/mnisperuza/OpenCLI/issues) with your OpenCLI version, OS and architecture, Python version, selected model/provider, command, and a redacted error. For security reports, use the process in [SECURITY.md](SECURITY.md).

## License

Apache License 2.0. See [LICENSE](LICENSE).
