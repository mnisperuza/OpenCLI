"""Professional optional Textual workspace for OpenCLI."""

from __future__ import annotations

from concurrent.futures import Future
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import re
import threading
from typing import TYPE_CHECKING

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.events import Key
from textual.screen import ModalScreen
from textual.widgets import (
    Button, Collapsible, Footer, Header, Input, Label, ListItem, ListView,
    Markdown, OptionList, RichLog, Static, TextArea,
)
from textual.widgets.option_list import Option

from main.permissions import PermissionDecision, PermissionRequest
from main.task_plan import PLAN_STATUSES, TaskPlanItem, TaskPlanStore
from main.ui_events import AgentEvent

if TYPE_CHECKING:
    from main.cli import OpenCLI


ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


class PermissionScreen(ModalScreen[PermissionDecision]):
    BINDINGS = [("escape", "deny", "Deny")]

    def __init__(self, request: PermissionRequest) -> None:
        super().__init__()
        self.request = request

    def compose(self) -> ComposeResult:
        with Vertical(id="permission-dialog"):
            yield Label("Permission required", classes="dialog-title")
            yield Static(
                f"Category: {self.request.category}\nAction: {self.request.action}\n"
                f"Target: {self.request.target}\n\nReason: {self.request.reason}",
                id="permission-details",
            )
            with Horizontal(classes="dialog-actions"):
                yield Button("Allow once", id="allow-once", variant="primary")
                yield Button("Session", id="allow-session")
                yield Button("Always", id="always-allow")
                yield Button("Deny", id="deny", variant="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        decisions = {
            "allow-once": PermissionDecision.ALLOW_ONCE,
            "allow-session": PermissionDecision.ALLOW_SESSION,
            "always-allow": PermissionDecision.ALWAYS_ALLOW,
            "deny": PermissionDecision.DENY,
        }
        self.dismiss(decisions[event.button.id or "deny"])

    def action_deny(self) -> None:
        self.dismiss(PermissionDecision.DENY)


class TextualPermissionBroker:
    """Bridge synchronous tool callbacks to async Textual modals."""

    def __init__(self, app: "OpenCLITui") -> None:
        self.app = app
        self._closed = False
        self._pending: set[Future[PermissionDecision]] = set()
        self._lock = threading.Lock()

    def request(self, request: PermissionRequest) -> PermissionDecision:
        future: Future[PermissionDecision] = Future()
        with self._lock:
            if self._closed:
                return PermissionDecision.DENY
            self._pending.add(future)
        self.app.call_from_thread(self._show, request, future)
        try:
            return future.result()
        finally:
            with self._lock:
                self._pending.discard(future)

    def _show(self, request: PermissionRequest, future: Future[PermissionDecision]) -> None:
        def resolve(decision: PermissionDecision | None) -> None:
            if not future.done():
                future.set_result(decision or PermissionDecision.DENY)
        self.app.push_screen(PermissionScreen(request), resolve)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            pending = list(self._pending)
        for future in pending:
            if not future.done():
                future.set_result(PermissionDecision.DENY)


class TextPromptScreen(ModalScreen[str | None]):
    def __init__(
        self,
        title: str,
        placeholder: str = "",
        value: str = "",
        password: bool = False,
    ) -> None:
        super().__init__()
        self.dialog_title = title
        self.placeholder = placeholder
        self.value = value
        self.password = password

    def compose(self) -> ComposeResult:
        with Vertical(id="text-dialog"):
            yield Label(self.dialog_title, classes="dialog-title")
            yield Input(
                value=self.value,
                placeholder=self.placeholder,
                password=self.password,
                id="dialog-input",
            )
            with Horizontal(classes="dialog-actions"):
                yield Button("Save", id="save", variant="primary")
                yield Button("Cancel", id="cancel")

    def on_mount(self) -> None:
        self.query_one(Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip() or None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        value = self.query_one(Input).value.strip() if event.button.id == "save" else ""
        self.dismiss(value or None)


class ChoiceScreen(ModalScreen[str | None]):
    def __init__(self, title: str, choices: list[tuple[str, str]]) -> None:
        super().__init__()
        self.dialog_title = title
        self.choices = choices

    def compose(self) -> ComposeResult:
        with Vertical(id="choice-dialog"):
            yield Label(self.dialog_title, classes="dialog-title")
            yield OptionList(
                *[Option(label, id=value) for value, label in self.choices],
                id="choice-list",
            )
            yield Button("Cancel", id="cancel")

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(str(event.option.id))

    def on_button_pressed(self, _event: Button.Pressed) -> None:
        self.dismiss(None)


class PromptTextArea(TextArea):
    """Multiline editor with reliable agent-style submit keys."""

    def on_key(self, event: Key) -> None:
        # Many terminals encode Ctrl+Enter as Enter or Ctrl+J. Enter therefore
        # submits consistently; Shift+Enter remains TextArea's newline action.
        if event.key == "shift+enter":
            event.prevent_default()
            event.stop()
            self.insert("\n")
            return
        if event.key in {"enter", "ctrl+enter", "ctrl+j"}:
            event.prevent_default()
            event.stop()
            self.app.action_submit()


class OpenCLITui(App[None]):
    TITLE = "OpenCLI"
    SUB_TITLE = "Agent workspace"
    CSS = """
    Screen { background: #0d1113; color: #d5dcda; }
    #status-line { height: 3; padding: 1 2; background: #151c1e; color: #b9d8c2; }
    #workspace { height: 1fr; }
    #plan-pane { width: 26; border-right: solid #344a4e; padding: 1; }
    #timeline { width: 1fr; padding: 1 2; }
    #inspector-pane { width: 36; border-left: solid #344a4e; padding: 1; }
    .pane-title, .dialog-title { color: #b9d8c2; text-style: bold; margin-bottom: 1; }
    .user-message { background: #172226; border-left: thick #6b9785; padding: 1; margin: 1 0; }
    .assistant-message { padding: 1; margin-bottom: 1; }
    .event-card { margin: 0 0 1 1; color: #aeb8b6; }
    #plan-list { height: 1fr; }
    #plan-help, #inspector-help { color: #7f8b89; height: auto; }
    #inspector { height: 1fr; border: round #344a4e; padding: 1; }
    #activity { height: 1; color: #d9c37a; padding: 0 2; }
    #prompt { height: 7; border: round #547a70; margin: 0 1; }
    TextArea:focus { border: round #9bd2bd; }
    #toolbar { height: 3; padding: 0 1; }
    #toolbar Button { margin-right: 1; min-width: 12; }
    #composer { height: 8; }
    #prompt { width: 1fr; }
    #send-button { width: 12; height: 5; margin: 1 1 0 0; }
    ModalScreen { align: center middle; background: rgba(0, 0, 0, 0.65); }
    #permission-dialog, #text-dialog, #choice-dialog {
        width: 72; max-width: 92%; height: auto; max-height: 85%;
        border: round #6b9785; background: #151c1e; padding: 1 2;
    }
    #permission-details { height: auto; }
    #choice-list { height: 18; }
    .dialog-actions { height: 3; align-horizontal: right; margin-top: 1; }
    .dialog-actions Button { margin-left: 1; }
    """
    BINDINGS = [
        ("ctrl+enter", "submit", "Send"), ("escape", "stop", "Stop"),
        ("ctrl+l", "clear_timeline", "Clear view"), ("ctrl+p", "add_plan", "Add plan"),
        ("ctrl+d", "cycle_plan", "Plan status"), ("ctrl+e", "edit_plan", "Edit plan"),
        ("delete", "delete_plan", "Delete plan"), ("alt+up", "plan_up", "Plan up"),
        ("alt+down", "plan_down", "Plan down"), ("ctrl+m", "models", "Models"),
        ("ctrl+r", "sessions", "Sessions"), ("ctrl+q", "quit", "Quit"),
    ]

    def __init__(
        self,
        cli: "OpenCLI",
        state_root: Path | None = None,
        api_start: bool = False,
    ) -> None:
        super().__init__()
        self.cli = cli
        self.state_root = state_root
        self.api_start = api_start
        self._busy = False
        self._assistant: Markdown | None = None
        self._assistant_text = ""
        self._events: list[AgentEvent] = []
        self.permission_broker = TextualPermissionBroker(self)
        self.plan_store: TaskPlanStore | None = None
        self.plan_items: list[TaskPlanItem] = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Static(id="status-line")
        with Horizontal(id="workspace"):
            with Vertical(id="plan-pane"):
                yield Label("TASK PLAN", classes="pane-title")
                yield ListView(id="plan-list")
                yield Static("Ctrl+P add · Ctrl+D advance", id="plan-help")
            yield VerticalScroll(id="timeline")
            with Vertical(id="inspector-pane"):
                yield Label("INSPECTOR", classes="pane-title")
                yield RichLog(id="inspector", wrap=True, markup=False)
                yield Static("Latest tool, result, or diff", id="inspector-help")
        yield Static("Ready", id="activity")
        with Horizontal(id="toolbar"):
            yield Button("Models", id="models-button")
            yield Button("Sessions", id="sessions-button")
            yield Button("Add plan", id="plan-button")
            yield Button("Clear", id="clear-button")
        with Horizontal(id="composer"):
            yield PromptTextArea(id="prompt")
            yield Button("Send", id="send-button", variant="primary")
        yield Footer()

    def on_mount(self) -> None:
        if self.cli.chat_session is None:
            self.cli.chat_session = self.cli.session_memory.create()
        self.plan_store = TaskPlanStore(
            Path.cwd(), self.cli.chat_session.session_id, root=self.state_root
        )
        self.plan_items = self.plan_store.load()
        self._sync_plan_context()
        self.cli.permission_manager.approval_callback = self.permission_broker.request
        self._mount_message(
            "OpenCLI agent workspace\nCtrl+Enter sends. Tools and permissions remain workspace-scoped.",
            "assistant-message",
        )
        self._refresh_plan()
        self._refresh_state()
        self.query_one("#prompt", PromptTextArea).focus()
        if self.api_start:
            self.call_after_refresh(self._start_default_api)

    def on_unmount(self) -> None:
        self.permission_broker.close()
        self.cli._save_chat_session()

    def on_resize(self, event) -> None:
        plan = self.query_one("#plan-pane", Vertical)
        inspector = self.query_one("#inspector-pane", Vertical)
        plan.styles.display = "none" if event.size.width < 105 else "block"
        inspector.styles.display = "none" if event.size.width < 78 else "block"

    def action_submit(self) -> None:
        if self._busy:
            return
        prompt = self.query_one("#prompt", PromptTextArea)
        message = prompt.text.strip()
        if not message:
            return
        prompt.text = ""
        if message.lower() in {"/model", "/models"}:
            self.action_models()
            return
        if message.lower() in {"/api", "/api-md"}:
            self.action_models()
            return
        if message.lower() in {"/sessions", "/memory"}:
            self.action_sessions()
            return
        if message.lower() in {"/clear", "/cls"}:
            self.action_clear_timeline()
            return
        self._mount_message(f"You\n{message}", "user-message")
        self._run_message(message)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        actions = {
            "send-button": self.action_submit,
            "models-button": self.action_models,
            "sessions-button": self.action_sessions,
            "plan-button": self.action_add_plan,
            "clear-button": self.action_clear_timeline,
        }
        action = actions.get(event.button.id or "")
        if action:
            action()

    def action_clear_timeline(self) -> None:
        self.query_one("#timeline", VerticalScroll).remove_children()

    def action_stop(self) -> None:
        if self._busy:
            self.cli._request_generation_stop()
            self.query_one("#activity", Static).update("Stopping generation…")

    def action_add_plan(self) -> None:
        self.push_screen(TextPromptScreen("Add task-plan item", "One concrete step"), self._add_plan_item)

    def _add_plan_item(self, text: str | None) -> None:
        if text and self.plan_store:
            TaskPlanStore.add(self.plan_items, text)
            self.plan_store.save(self.plan_items)
            self._sync_plan_context()
            self._refresh_plan()

    def action_cycle_plan(self) -> None:
        view = self.query_one("#plan-list", ListView)
        if view.index is None or not 0 <= view.index < len(self.plan_items):
            return
        item = self.plan_items[view.index]
        item.status = PLAN_STATUSES[(PLAN_STATUSES.index(item.status) + 1) % len(PLAN_STATUSES)]
        if self.plan_store:
            self.plan_store.save(self.plan_items)
        self._sync_plan_context()
        self._refresh_plan(view.index)

    def action_edit_plan(self) -> None:
        index = self._selected_plan_index()
        if index is None:
            return
        self.push_screen(
            TextPromptScreen("Edit task-plan item", value=self.plan_items[index].text),
            lambda text: self._edit_plan_item(index, text),
        )

    def _edit_plan_item(self, index: int, text: str | None) -> None:
        if not text or not 0 <= index < len(self.plan_items):
            return
        self.plan_items[index].text = " ".join(text.split())
        self._save_plan(index)

    def action_delete_plan(self) -> None:
        index = self._selected_plan_index()
        if index is not None:
            self.plan_items.pop(index)
            self._save_plan(min(index, len(self.plan_items) - 1))

    def action_plan_up(self) -> None:
        self._move_plan(-1)

    def action_plan_down(self) -> None:
        self._move_plan(1)

    def _move_plan(self, offset: int) -> None:
        index = self._selected_plan_index()
        if index is None:
            return
        target = index + offset
        if not 0 <= target < len(self.plan_items):
            return
        self.plan_items[index], self.plan_items[target] = self.plan_items[target], self.plan_items[index]
        self._save_plan(target)

    def _selected_plan_index(self) -> int | None:
        index = self.query_one("#plan-list", ListView).index
        return index if index is not None and 0 <= index < len(self.plan_items) else None

    def _save_plan(self, selected: int | None = None) -> None:
        if self.plan_store:
            self.plan_store.save(self.plan_items)
        self._sync_plan_context()
        self._refresh_plan(selected if selected is not None and selected >= 0 else None)

    def action_models(self) -> None:
        choices: list[tuple[str, str]] = []
        for key, value in self.cli.router_models().items():
            choices.append((key, f"{value.get('display_name', key)}  [{key}]"))
        for key, profile in self.cli.api_profiles.profiles.items():
            choices.append(
                (f"api::{key}", f"API · {profile['provider']} · {profile['model']}")
            )
        self.push_screen(ChoiceScreen("Select model or API profile", choices), self._select_model)

    def _start_default_api(self) -> None:
        profile = self.cli.api_profiles.default()
        if not profile:
            self._mount_message("No saved API profile. Use Ctrl+M to select one.", "event-card")
            return
        key = f"{profile['provider']}:{profile['model']}"
        self._select_model(f"api::{key}")

    def _select_model(self, key: str | None) -> None:
        if not key:
            return
        if key.startswith("api::"):
            profile = self.cli.api_profiles.profiles.get(key[5:])
            if not profile:
                return
            from main.api_providers import PROVIDERS
            definition = PROVIDERS[profile["provider"]]
            api_key = self.cli._api_key or os.environ.get(definition.environment_variable, "")
            if api_key:
                self._activate_api_profile(profile["provider"], profile["model"], api_key)
            else:
                self.push_screen(
                    TextPromptScreen(
                        f"{definition.name} API key",
                        f"Not saved; prefer {definition.environment_variable}",
                        password=True,
                    ),
                    lambda secret: self._activate_api_profile(
                        profile["provider"], profile["model"], secret
                    ) if secret else None,
                )
            return
        self._load_selected_model(key)

    @work(thread=True, exclusive=True)
    def _activate_api_profile(self, provider: str, model: str, api_key: str) -> None:
        self.call_from_thread(self._set_busy, True, f"Connecting {provider}…")
        output = io.StringIO()
        with redirect_stdout(output):
            success = self.cli._activate_api(provider, api_key, model)
        self.call_from_thread(
            self._mount_message,
            f"API {'ready' if success else 'failed'}: {provider} · {model}",
            "event-card",
        )
        self.call_from_thread(self._refresh_state)
        self.call_from_thread(self._set_busy, False)

    @work(thread=True, exclusive=True)
    def _load_selected_model(self, key: str) -> None:
        self.call_from_thread(self._set_busy, True, f"Loading {key}…")
        success = self.cli.load_model(key, self.cli.quant, show_picker=False, render=False)
        self.call_from_thread(self._mount_message, f"Model {'ready' if success else 'failed'}: {key}", "event-card")
        self.call_from_thread(self._refresh_state)
        self.call_from_thread(self._set_busy, False)

    def action_sessions(self) -> None:
        choices = [(str(path), path.stem) for path in self.cli.session_memory.list()]
        if not choices:
            self._mount_message("No previous workspace sessions.", "event-card")
            return
        self.push_screen(ChoiceScreen("Import session memory", choices), self._select_session)

    def _select_session(self, value: str | None) -> None:
        if value:
            self._import_session(Path(value))

    @work(thread=True, exclusive=True)
    def _import_session(self, path: Path) -> None:
        self.call_from_thread(self._set_busy, True, "Importing session memory…")
        try:
            content = self.cli.session_memory.load(path)
            if not self.cli.ensure_agent_runtime():
                raise RuntimeError("Agent runtime unavailable")
            self.cli.agent_runtime.load_memory(content, path.name)
            message = f"Imported session memory: {path.stem}"
        except Exception as error:
            message = f"Session import failed: {error}"
        self.call_from_thread(self._mount_message, message, "event-card")
        self.call_from_thread(self._refresh_state)
        self.call_from_thread(self._set_busy, False)

    @work(thread=True, exclusive=True)
    def _run_message(self, message: str) -> None:
        self.call_from_thread(self._set_busy, True)
        query, think_mode = self._parse_think_command(message)
        if query.startswith(("/", "!")):
            output = io.StringIO()
            try:
                if query.lower() in {"/model-add", "/modeladd", "/model-rm", "/modelrm", "/api-del"}:
                    result = True
                    output.write("Profile editing remains in classic REPL for this release.")
                else:
                    with redirect_stdout(output):
                        result = self.cli.handle_command(query)
                if result is False:
                    self.call_from_thread(self.exit)
            except Exception as error:
                output.write(f"Error: {error}")
            rendered = ANSI_ESCAPE.sub("", output.getvalue()).strip()
            if rendered:
                self.call_from_thread(self._mount_message, rendered, "event-card")
            self.call_from_thread(self._sync_session_plan)
        else:
            self.call_from_thread(self._begin_assistant)
            for event in self.cli.stream_turn(query, think_mode=think_mode):
                self.call_from_thread(self._handle_event, event)
        self.call_from_thread(self._refresh_state)
        self.call_from_thread(self._set_busy, False)

    @staticmethod
    def _parse_think_command(message: str) -> tuple[str, bool]:
        if message.lower().startswith("/think "):
            return message.split(" ", 1)[1].strip(), True
        return message, False

    def _begin_assistant(self) -> None:
        self._assistant_text = ""
        self._assistant = Markdown("_Thinking…_", classes="assistant-message")
        self.query_one("#timeline", VerticalScroll).mount(self._assistant)

    def _handle_event(self, event: AgentEvent) -> None:
        self._events.append(event)
        if event.type == "token":
            self._assistant_text += event.content
            if self._assistant:
                self._assistant.update(self._assistant_text)
        elif event.type == "status":
            self._mount_message(f"Status · {event.content}", "event-card")
        elif event.type == "tool":
            details = json.dumps(dict(event.arguments), indent=2, default=str)
            self._mount_collapsible(f"Tool · {event.name}", details)
            self._inspect(f"TOOL {event.name}\n\n{details}")
        elif event.type == "tool_result":
            text = f"Result · {event.name}: {event.summary or 'complete'}"
            self._mount_message(text, "event-card")
            self._inspect(text)
        elif event.type == "file_change":
            details = dict(event.details)
            diff = str(details.get("diff", "")) or "Diff unavailable"
            suffix = " (truncated)" if details.get("truncated") else ""
            title = f"Change · {details.get('path', event.summary)}{suffix}"
            self._mount_collapsible(title, diff)
            self._inspect(f"{title}\n\n{diff}")
        elif event.type == "error":
            self._mount_message(f"Error · {event.content}", "event-card")
            self._inspect(event.content)
        elif event.type == "done" and not self._assistant_text and event.content:
            self._assistant_text = event.content
            if self._assistant:
                self._assistant.update(self._assistant_text)
        self.query_one("#timeline", VerticalScroll).scroll_end(animate=False)

    def _mount_message(self, text: str, classes: str) -> None:
        self.query_one("#timeline", VerticalScroll).mount(Static(text, classes=classes))

    def _mount_collapsible(self, title: str, body: str) -> None:
        self.query_one("#timeline", VerticalScroll).mount(
            Collapsible(Static(body), title=title, collapsed=True, classes="event-card")
        )

    def _inspect(self, text: str) -> None:
        inspector = self.query_one("#inspector", RichLog)
        inspector.clear()
        inspector.write(text)

    def _set_busy(self, busy: bool, message: str | None = None) -> None:
        self._busy = busy
        self.query_one("#activity", Static).update(
            message or ("Agent working… Escape stops generation" if busy else "Ready")
        )

    def _refresh_state(self) -> None:
        snapshot = self.cli._context_snapshot()
        usage = self.cli.context_accounting.usage
        filled = min(16, max(0, round(snapshot.percent_used * 16 / 100)))
        meter = "█" * filled + "░" * (16 - filled)
        session = self.cli.chat_session.session_id if self.cli.chat_session else "not started"
        self.query_one("#status-line", Static).update(
            f"{snapshot.profile.display_name}  ctx [{meter}] {snapshot.percent_used:.0f}% "
            f"{snapshot.used_tokens:,}/{snapshot.profile.context_window:,}  tokens {usage.total_tokens:,}  "
            f"tools {'on' if self.cli.tools_enabled else 'off'}  "
            f"web {'on' if self.cli.permission_manager.web_enabled else 'off'}  "
            f"sandbox {'on' if self.cli.sandbox_enabled else 'off'}  session {session}"
        )

    def _refresh_plan(self, selected: int | None = None) -> None:
        view = self.query_one("#plan-list", ListView)
        view.clear()
        markers = {"pending": "○", "in_progress": "◐", "completed": "●"}
        for item in self.plan_items:
            view.append(ListItem(Label(f"{markers[item.status]} {item.text}")))
        if selected is not None and self.plan_items:
            view.index = min(selected, len(self.plan_items) - 1)

    def _sync_plan_context(self) -> None:
        self.cli.task_plan_context = "\n".join(
            f"- [{item.status}] {item.text}" for item in self.plan_items
        )

    def _sync_session_plan(self) -> None:
        if self.cli.chat_session is None:
            return
        expected_name = f"{self.cli.chat_session.session_id}.json"
        if self.plan_store is not None and self.plan_store.path.name == expected_name:
            return
        self.plan_store = TaskPlanStore(
            Path.cwd(), self.cli.chat_session.session_id, root=self.state_root
        )
        self.plan_items = self.plan_store.load()
        self._sync_plan_context()
        self._refresh_plan()


__all__ = ["ChoiceScreen", "OpenCLITui", "PermissionScreen", "TextualPermissionBroker"]
