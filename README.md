# OpenCLI

OpenCLI is an educational, local-first AI assistant for trusted workspaces.
It runs GGUF models through llama.cpp and can use Groq, Gemini, or OpenRouter
for a single API-backed session. Pydantic AI manages tool loops and tool
validation.

## Release 1.5.3 status

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
- Bounded ReAct loops: one tool action per model step, repeated-action/failure guards, mutation evidence, persistent detailed plans.
- Selectable Docker/E2B sandbox backends with explicit, conflict-aware E2B workspace sync.

Preview or incomplete:

- `/think` is a prompt hint, not a separate reasoning runtime.
- `/paste` and `/multiline` toggle input mode; large-paste ergonomics are still experimental.
- Model tool calling depends on selected local model. Enable `/tool-auto on` only for weaker models that need deterministic routing.
- Docker commands require Docker Desktop. E2B requires optional SDK plus user API key. No host-shell fallback exists.
- MCP, plugins, subagents, and editor integration are not implemented in 1.5.3.

## Install

```bash
pip install opencli
opencli
```

E2B support:

```bash
pip install "opencli[sandbox]"
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

The keyboard-first TUI uses one full-width conversation stream with inline tool,
ReAct, plan, thinking-summary, error, and diff cards. Sensitive actions use
buttonless approval screens that default to denial. Context, usage, model, and
sandbox state remain visible in compact adaptive status lines. Type `/` for
filtered completion of existing commands; no separate UI command system exists.
Press `Enter` to submit, `Shift+Enter` for a newline, `Escape` to stop, and
`Ctrl+G` to resume following live output after scrolling upward. Model, session,
plan, memory, compact, and permission workflows remain available through their
documented slash commands.

Pillow normalizes existing model-bound visual context with EXIF correction,
RGB conversion, and bounded dimensions. The TUI intentionally has no image
picker, image preview, or terminal-image rendering yet.

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
| `/pwd` | Show trusted workspace root and logical current directory. |
| `/cd PATH` | Change logical directory inside trusted workspace; does not change host process directory. |
| `/roots` | Show filesystem roots available to the agent. |
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
| `/plan`, `/plan add STEP`, `/plan set ID STATUS`, `/plan clear` | Inspect or manually maintain persistent session plan. Model can also create/update it through tools. |
| `/react on|off`, `/react status` | Enable or inspect bounded model-requested ReAct tasks; default on. The model uses `start_react_task` only for genuine multi-step work. |
| `/web on|off` | Enable or block web tools for session. |
| `/web always` | Persist web approval for current workspace. |
| `/web ask` | Restore web approval prompts. |
| `/sandbox docker [IMAGE]` | Select ephemeral Docker backend and optional user-chosen image. `/sandbox on` remains alias. |
| `/sandbox e2b connect ID` | Connect user-owned E2B sandbox. API key comes only from `E2B_API_KEY`. |
| `/sandbox e2b create [TEMPLATE] [--network]` | User explicitly creates E2B sandbox; network disabled unless requested. |
| `/sandbox push`, `/sandbox pull` | Explicit bounded E2B upload and conflict-aware import. Secrets and remote deletions are excluded. |
| `/sandbox status`, `/sandbox stop`, `/sandbox off` | Inspect, kill, or detach active backend. |
| `!<argv>`, `!!<argv>` | Run read-only or write-approved argv in active sandbox; no host shell. |
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
| `get_working_directory`, `set_working_directory`, `list_allowed_roots` | none | Inspect or change the logical directory inside the trusted workspace. Path escape is rejected. |
| `list_files`, `read_text_file`, `search_text`, `file_info` | `file_read` | Read trusted-workspace files only. Protected paths include `.git`, `.env`, keys, and secret-like names. |
| `write_text_file`, `edit_text_file`, `create_directory` | `file_write` | Create or change workspace files after approval; writes are size-limited. |
| `web_search`, `web_fetch` | `web` | Search and fetch public HTTP/S sources after approval. |
| `get_task_plan`, `create_task_plan`, `add_task_plan_item`, `update_task_plan_item` | none | Maintain persistent plan; completion/dismissal requires evidence by instruction and loop policy. |
| `get_sandbox_status`, `run_sandboxed_command` | `command`, `file_write` for Docker writes or any E2B command | Run argv in selected backend. E2B cannot enforce per-command read-only state, so agent commands receive extra approval. Lifecycle and host sync remain user-only. |

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

## Docker and E2B sandboxes

Install and start Docker Desktop, then run:

```text
/sandbox docker python:3.12-slim
!python -V
!!python -m pytest -q
```

Each command starts an ephemeral container with network disabled, read-only root
filesystem, dropped Linux capabilities, no privilege escalation, CPU/RAM/PID
limits, and a read-only workspace mount by default. Docker image pulls may need
internet access during Docker setup; sandboxed commands do not receive network.

E2B uses user-controlled lifecycle and explicit transfer:

```text
# Set E2B_API_KEY in process environment first.
/sandbox e2b connect YOUR_SANDBOX_ID
# Or: /sandbox e2b create YOUR_TEMPLATE
/sandbox push
!python -m pytest -q
/sandbox pull
```

Push/pull skip `.git`, `.opencli`, environments, keys, and secret-like files;
enforce file/byte limits; reject local conflicts; ignore remote deletions. Agent
may execute approved commands but cannot create, connect, stop, push, or pull a
sandbox. Connected E2B sandboxes are not killed by `/sandbox off`; use
`/sandbox stop` first when shutdown is wanted.

## ReAct execution and plans

ReAct is enabled by default for local and API models. For genuinely multi-step
work, the model can call `start_react_task`; OpenCLI then owns the capped budget,
repeated-action and failure guards, and all normal permission checks. It cannot
restart a task to evade limits. Simple requests remain ordinary tool/chat turns.
OpenCLI never requests or stores private chain-of-thought. Persistent user-visible
task plan remains separate from loop state, so planning-only requests can produce
detailed plans without editing files.

## Internal boundaries

`main.interfaces` declares stable internal contracts for model backends, tool
providers, permission gates, sessions, sandboxes, and loop controllers. Providers may differ, but CLI
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
