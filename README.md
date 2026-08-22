# OpenCLI

OpenCLI is an educational, local-first AI assistant for trusted workspaces.
It runs GGUF models through llama.cpp and can use Groq, Gemini, or OpenRouter
for a single API-backed session. Pydantic AI manages tool loops and tool
validation.

## Release 1.5.1 status

This release adds model-aware context visibility on top of the stabilized 1.5
base. OpenCLI now resolves model capability profiles, accounts for prompt
components, tracks session usage, and shows context occupancy beside model name.

Supported now:

- Local GGUF inference through a local llama.cpp server.
- Hosted chat and native tool calls through Groq, Gemini, and OpenRouter.
- Permission-gated workspace files, web retrieval, session archives, and Docker commands.

Preview or incomplete:

- `/think` is a prompt hint, not a separate reasoning runtime.
- `/paste` and `/multiline` toggle input mode; large-paste ergonomics are still experimental.
- Model tool calling depends on selected local model. Enable `/tool-auto on` only for weaker models that need deterministic routing.
- Docker commands require Docker Desktop running. OpenCLI does not install Docker or provide host-shell fallback.
- MCP, plugins, subagents, plan mode, context compaction, and editor integration are not implemented in 1.5.1.

## Install

```bash
pip install opencli
opencli
```

Development on Windows:

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.lock
.venv\Scripts\python -m pip install --no-deps -e .
.venv\Scripts\python -m unittest discover -s tests -v
```

`pyproject.toml` owns package metadata. `requirements.lock` records tested
Python 3.12 runtime dependencies. llama.cpp startup waits up to 900 seconds;
set `OPENCLI_LLAMA_CPP_STARTUP_TIMEOUT` to change it.

## Commands

| Command | Behavior |
|---|---|
| `/help`, `/h` | Show built-in help. |
| `/status` | Show loaded model, mode, tools, web, sandbox, and session state. |
| `/context` | Show active profile, prompt breakdown, output reserve, and available input. |
| `/usage` | Show input/output estimates or provider-reported usage for current session. |
| `/prompt-size` | Show fixed instruction and tool-schema prompt cost. |
| `/agent`, `/agent status` | Show agent runtime state. |
| `/model` | Open local model picker. `/model <name>` loads built-in or saved profile. |
| `/model-add`, `/modeladd` | Add a Hugging Face GGUF repo or existing local `.gguf` profile. |
| `/model-rm`, `/modelrm` | Remove saved model profile; never deletes model file. |
| `/api` | Choose Groq, Gemini, or OpenRouter plus hosted model. |
| `/api-md` | Change hosted model while retaining current provider/key. |
| `/api-del` | Remove saved provider/model profile; keys are never saved. |
| `/tools` | List tools currently available to agent. |
| `/tools-on`, `/tools-off` | Enable or disable agent tools. |
| `/tool-auto on|off` | Toggle deterministic routing for local models; default off. |
| `/web on|off` | Enable or block web tools for session. |
| `/web always` | Persist web approval for current workspace. |
| `/web ask` | Restore web approval prompts. |
| `/sandbox`, `/sandbox on|off` | Show or toggle Docker-only command sandbox. |
| `!<argv>` | Run command only inside enabled Docker sandbox; shell syntax unsupported. |
| `/permissions`, `/permissions reset` | Show or reset workspace permissions. |
| `/history` | Show recent agent history. |
| `/new`, `/newchat` | Start clean session. |
| `/memory`, `/mem` | Select one prior workspace archive to load as untrusted context. |
| `/memory clear` | Clear active runtime conversation. |
| `/memory current` | Show active archive path. |
| `/remember TEXT` | Save user-controlled session note. |
| `/paste`, `/multiline` | Toggle experimental multiline input. |
| `/think TEXT` | Send prompt with thinking hint; preview behavior. |
| `/endserver` | Unload model and stop OpenCLI-owned llama.cpp server. |
| `/clear`, `/cls` | Clear terminal. |
| `/exit`, `/quit`, `/q` | Save session, stop server, exit. |

## Agent tools

Tools are enabled only after workspace trust confirmation. Every sensitive call
requests permission unless allowed for session or workspace.

| Tool | Permission | Behavior |
|---|---|---|
| `list_files`, `read_text_file`, `search_text`, `file_info` | `file_read` | Read trusted-workspace files only. Protected paths include `.git`, `.env`, keys, and secret-like names. |
| `write_text_file`, `edit_text_file`, `create_directory` | `file_write` | Create or change workspace files after approval; writes are size-limited. |
| `web_search`, `web_fetch` | `web` | Search and fetch public HTTP/S sources after approval. |
| `run_sandboxed_command` | `command`, optional `file_write` | Run argv in Docker with no network. Workspace remains read-only unless separately approved. |

API keys remain process-memory only. Before API requests, OpenCLI asks permission
to transmit prompt, conversation context, tool schemas, and tool results.
Sessions save under `~/.opencli/sessions/`, scoped by workspace. Loaded session
archives remain untrusted historical text, never executable instructions.

## Docker sandbox

Install and start Docker Desktop, then run:

```text
/sandbox on
!python -V
```

Each command starts an ephemeral container with network disabled, read-only root
filesystem, dropped Linux capabilities, no privilege escalation, CPU/RAM/PID
limits, and a read-only workspace mount by default. Docker image pulls may need
internet access during Docker setup; sandboxed commands do not receive network.

## Internal boundaries

`main.interfaces` declares stable internal contracts for model backends, tool
providers, permission gates, and session stores. Providers may differ, but CLI
code uses those shared boundaries. New features should extend these contracts
instead of binding UI code to a model vendor or tool implementation.

## Model context profiles

OpenCLI includes tested profiles for Ministral 3 14B, GPT-OSS 20B, Devstral
Small 2 24B, and Qwen 3.8 27B. Unknown models use a conservative 16,384-token
window and 2,048-token output reserve unless model metadata or workspace config
provides explicit limits. Counts use loaded tokenizer when available; otherwise
OpenCLI marks byte-based estimates with `~`.

Workspace overrides live in `.opencli/config.toml`. This file is protected from
agent file tools. Identifiers may be a built-in key, exact model ID, or
`provider:model-id`:

```toml
[models."openrouter:anthropic/claude-example"]
display_name = "Claude Example"
context_window = 200000
max_output_tokens = 8192
supports_tools = true
supports_vision = true
supports_reasoning = true
```

Invalid overrides are ignored and reported by `/context`.

## Requirements

| Component | Minimum | Recommended |
|---|---:|---:|
| RAM | 8 GB | 16 GB+ |
| VRAM | 3 GB | 6 GB+ |
| Python | 3.10 | 3.12+ |
| Storage | 30 GB | 40 GB |

## Support

Report issues at [GitHub Issues](https://github.com/mnisperuza/OpenCLI/issues).
Include OpenCLI version, OS, selected model/provider, command, and error text.
