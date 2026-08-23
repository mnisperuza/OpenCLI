# OpenCLI

OpenCLI is an educational, local-first AI assistant for trusted workspaces.
It runs GGUF models through llama.cpp and can use Groq, Gemini, or OpenRouter
for a single API-backed session. Pydantic AI manages tool loops and tool
validation.

## Release 1.5.2 status

This release stabilizes the model-aware context and Textual agent workspace on
top of the 1.5 base. OpenCLI now resolves model capability profiles, accounts
for prompt components, tracks session usage, and shows context occupancy beside
model name.

Supported now:

- Local GGUF inference through a local llama.cpp server.
- Hosted chat and native tool calls through Groq, Gemini, and OpenRouter.
- Permission-gated workspace files, web retrieval, session archives, and Docker commands.
- Textual coding-agent workspace with live events, approvals, diffs, and task plans.
- Per-turn English/Spanish response-language guard; ambiguous technical prompts use English.
- Two-level context control: consumed tool-result pruning plus loaded-model structured summaries with a retained hot window.
- Bounded, syntax-safe diff previews with added/removed line highlighting.
- Session titles, full-session resume, and non-stacking compact memory import.
- Model-visible task-plan status updates for completed and dismissed items.

Preview or incomplete:

- `/think` is a prompt hint, not a separate reasoning runtime.
- `/paste` and `/multiline` toggle input mode; large-paste ergonomics are still experimental.
- Model tool calling depends on selected local model. Enable `/tool-auto on` only for weaker models that need deterministic routing.
- Docker commands require Docker Desktop running. OpenCLI does not install Docker or provide host-shell fallback.
- MCP, plugins, subagents, agent-created detailed plans, and editor integration are not implemented in 1.5.2.

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

## Textual agent workspace

Textual is the default interface:

```bash
opencli
```

Use `opencli --cli` for the classic line-oriented interface. The former
`opencli --tui` flag remains accepted as a compatibility alias.

The TUI streams model and tool events live, gates sensitive tools with approval
modals, shows context and usage state, previews approved file diffs, switches
saved models/API profiles, adds/removes GGUF and API profiles, discovers hosted
model limits, imports prior session memory explicitly, and stores a task plan
outside the agent-writable workspace. The agent can inspect and mark existing
steps `completed` or `dismissed` when evidence supports it; it cannot create or
edit plan steps. Press `Enter` or click
Send to submit; use `Shift+Enter` for a newline. `Ctrl+Enter` remains supported
where the terminal distinguishes it. Use `Escape` to stop, `Ctrl+P` to add a
plan step, `Ctrl+M` for models, `Ctrl+R` for sessions, and `Ctrl+K` or Compact
button to shorten old history.

Sessions have stable timestamp/UUID filenames and optional human-readable titles.
The model may propose one short title through a validated tool; use
`/session-name TEXT` to set or replace it. In Sessions, choose **Resume full
session** to reopen retained runtime history, or **Import compact memory** to
replace one bounded historical capsule in current chat.

## Commands

| Command | Behavior |
|---|---|
| `/help`, `/h` | Show built-in help. |
| `/status` | Show loaded model, mode, tools, web, sandbox, and session state. |
| `/context` | Show active profile, prompt breakdown, output reserve, and available input. |
| `/usage` | Show input/output estimates or provider-reported usage for current session. |
| `/prompt-size` | Show fixed instruction and tool-schema prompt cost. |
| `/compact`, `/compact status` | Micro-prune tool results, summarize cold history with the loaded model, and retain recent complete turns. |
| `/compact auto on|off` | Toggle context-aware compaction before a request; default on. |
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
| `/session-name TEXT` | Set or replace active session title. |
| `/memory`, `/mem` | Select one prior archive to replace current imported memory capsule. |
| `/memory clear` | Clear active runtime conversation while keeping durable user notes. |
| `/memory notes`, `/memory forget`, `/memory list` | Review, remove, or list durable memory and workspace archives. |
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

Large tool results are pruned after the model consumes them; bounded payload
copies move to the local Markdown tool archive and stay outside future model context. A turn
can fetch at most three pages, each bounded to 8,000 characters. `/compact` uses
the currently loaded local or API model, without tools or active chat history,
to create a structured memory. Recent complete turns remain verbatim. If model
summarization fails, OpenCLI uses a deterministic excerpt. Automatic compaction
runs near 80% context occupancy and reserves space for tool observations.
`/remember` notes remain after `/memory clear` until `/memory forget`.

Hosted output is bounded twice: provider `max_tokens` plus an independent stream
character cap. The TUI batches streaming updates and caps rendered Markdown to
prevent oversized responses from exhausting UI memory.

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
Small 2 24B, and Qwen 3.8 27B. Unknown models use a 32,768-token window and
4,096-token output reserve unless model metadata, saved API profile, or workspace
config provides explicit limits. API discovery recognizes common flat, nested,
and camel-case provider limit fields. Reconnecting a saved TUI API profile
refreshes missing metadata and persists discovered limits. The TUI API form
exposes both values when a provider omits metadata. Output reserve is capped at
half the context window so a malformed profile cannot consume all input space.
Counts use the loaded tokenizer when available; otherwise OpenCLI marks
byte-based estimates with `~`.

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
