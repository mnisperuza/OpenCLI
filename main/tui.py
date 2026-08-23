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
import time
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
from rich.text import Text

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


class ConfirmScreen(ModalScreen[bool]):
    def __init__(self, title: str, message: str) -> None:
        super().__init__()
        self.dialog_title = title
        self.message = message

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-dialog"):
            yield Label(self.dialog_title, classes="dialog-title")
            yield Static(self.message)
            with Horizontal(classes="dialog-actions"):
                yield Button("Confirm", id="confirm", variant="error")
                yield Button("Cancel", id="cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "confirm")


class FormScreen(ModalScreen[dict[str, str] | None]):
    """Small reusable Textual form; validation remains in domain registries."""

    def __init__(
        self,
        title: str,
        fields: list[tuple[str, str, str, bool]],
    ) -> None:
        super().__init__()
        self.dialog_title = title
        self.fields = fields

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="form-dialog"):
            yield Label(self.dialog_title, classes="dialog-title")
            for name, label, default, password in self.fields:
                yield Label(label, classes="form-label")
                yield Input(value=default, password=password, id=f"form-{name}")
            with Horizontal(classes="dialog-actions"):
                yield Button("Save", id="save", variant="primary")
                yield Button("Cancel", id="cancel")

    def on_mount(self) -> None:
        self.query_one(Input).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "save":
            self.dismiss(None)
            return
        self.dismiss(
            {
                name: self.query_one(f"#form-{name}", Input).value.strip()
                for name, *_ in self.fields
            }
        )


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
    MAX_RENDERED_RESPONSE_CHARS = 96_000
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
    #permission-dialog, #text-dialog, #choice-dialog, #confirm-dialog, #form-dialog {
        width: 72; max-width: 92%; height: auto; max-height: 85%;
        border: round #6b9785; background: #151c1e; padding: 1 2;
    }
    #form-dialog { width: 80; height: 90%; }
    .form-label { color: #aeb8b6; margin-top: 1; }
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
        ("ctrl+r", "sessions", "Sessions"), ("ctrl+k", "compact", "Compact"),
        ("ctrl+q", "quit", "Quit"),
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
        self._response_render_truncated = False
        self._events: list[AgentEvent] = []
        self.permission_broker = TextualPermissionBroker(self)
        self.plan_store: TaskPlanStore | None = None
        self.plan_items: list[TaskPlanItem] = []
        self._pending_api_client = None
        self._pending_api_key = ""

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
            yield Button("Compact", id="compact-button")
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
        self.cli.task_plan_store = self.plan_store
        self.plan_items = self.plan_store.load()
        if self.cli.agent_runtime is not None:
            self.cli.agent_runtime.task_plan_store = self.plan_store
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
        if message.lower() in {"/model-add", "/modeladd"}:
            self.action_add_model()
            return
        if message.lower() in {"/model-rm", "/modelrm"}:
            self.action_remove_model()
            return
        if message.lower() == "/api":
            self.action_add_api()
            return
        if message.lower() == "/api-md":
            self.action_models()
            return
        if message.lower() == "/api-del":
            self.action_remove_api()
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
            "compact-button": self.action_compact,
            "plan-button": self.action_add_plan,
            "clear-button": self.action_clear_timeline,
        }
        action = actions.get(event.button.id or "")
        if action:
            action()

    def action_compact(self) -> None:
        if not self._busy:
            self._run_message("/compact")

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
        choices: list[tuple[str, str]] = [
            ("manage::add-model", "+ Add local/Hugging Face GGUF profile"),
            ("manage::remove-model", "- Remove GGUF profile"),
            ("manage::add-api", "+ Add/connect API profile"),
            ("manage::remove-api", "- Remove API profile"),
        ]
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
        management = {
            "manage::add-model": self.action_add_model,
            "manage::remove-model": self.action_remove_model,
            "manage::add-api": self.action_add_api,
            "manage::remove-api": self.action_remove_api,
        }
        if key in management:
            management[key]()
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

    def action_add_model(self) -> None:
        self.push_screen(
            FormScreen(
                "Add GGUF model profile",
                [
                    ("name", "Display name", "", False),
                    ("source", "Source: huggingface or local", "huggingface", False),
                    ("path", "Repository owner/repo[:quant] or local .gguf path", "", False),
                    ("file", "Exact GGUF filename (optional)", "", False),
                    ("context", "Context window", "32768", False),
                    ("output", "Max output tokens", "8192", False),
                    ("temperature", "Temperature", "0.7", False),
                    ("thinking", "Thinking support: yes/no", "no", False),
                    ("vision", "Vision support: yes/no", "no", False),
                ],
            ),
            self._create_model_profile,
        )

    @work(thread=True, exclusive=True)
    def _create_model_profile(self, values: dict[str, str] | None) -> None:
        if not values:
            return
        self.call_from_thread(self._set_busy, True, "Saving model profile…")
        from main.model_registry import ModelRegistryError

        try:
            key = self.cli.model_registry.add(
                name=values["name"],
                source_type=values["source"],
                path=values["path"],
                llama_file=values["file"],
                context=values["context"],
                max_tokens=values["output"],
                temperature=values["temperature"],
                has_thinking=values["thinking"].casefold() in {"yes", "y", "true", "1"},
                supports_vision=values["vision"].casefold() in {"yes", "y", "true", "1"},
                reserved_keys=set(self.cli.MODELS) | set(self.cli.BUILTIN_MODELS),
            )
            if self.cli.engine is not None:
                self.cli.engine.register_models(self.cli.model_registry.engine_models())
            self.cli.model_selection_mode = "manual"
            self.cli.manual_model_key = key
            self.cli.agent_runtime = None
            success = self.cli.load_model(key, self.cli.quant, show_picker=False, render=False)
            message = f"Model profile {'added and loaded' if success else 'saved; load failed'}: {key}"
        except (ModelRegistryError, OSError, ValueError) as error:
            message = f"Model profile not added: {error}"
        self.call_from_thread(self._mount_message, message, "event-card")
        self.call_from_thread(self._refresh_state)
        self.call_from_thread(self._set_busy, False)

    def action_remove_model(self) -> None:
        choices = [
            (key, f"{value.get('display_name', key)} · {value.get('path', '')}")
            for key, value in self.cli.custom_models().items()
        ]
        if not choices:
            self._mount_message("No user-added model profiles.", "event-card")
            return
        self.push_screen(
            ChoiceScreen("Remove GGUF profile", choices), self._confirm_remove_model
        )

    def _confirm_remove_model(self, key: str | None) -> None:
        if key:
            self.push_screen(
                ConfirmScreen(
                    "Remove model profile",
                    "Profile will be removed. GGUF file will not be deleted.",
                ),
                lambda confirmed: self._remove_model_profile(key) if confirmed else None,
            )

    def _remove_model_profile(self, key: str) -> None:
        try:
            if self.cli.mode == key:
                self.cli.stop_server(mark_stopped=False)
                self.cli.mode = "auto"
                self.cli.manual_model_key = None
            removed = self.cli.model_registry.remove(key)
            if self.cli.engine is not None:
                self.cli.engine.MODELS.pop(key, None)
            self.cli.agent_runtime = None
            message = f"Removed model profile: {removed.get('display_name', key)}"
        except Exception as error:
            message = f"Model profile removal failed: {error}"
        self._mount_message(message, "event-card")
        self._refresh_state()

    def action_add_api(self) -> None:
        from main.api_providers import PROVIDERS

        self.push_screen(
            ChoiceScreen(
                "Select API provider",
                [(key, definition.name) for key, definition in PROVIDERS.items()],
            ),
            self._select_api_provider,
        )

    def _select_api_provider(self, provider: str | None) -> None:
        if not provider:
            return
        from main.api_providers import PROVIDERS

        definition = PROVIDERS[provider]
        api_key = self.cli._api_key or os.environ.get(
            definition.environment_variable, ""
        )
        if api_key:
            self._discover_api_models(provider, api_key)
            return
        self.push_screen(
            TextPromptScreen(
                f"{definition.name} API key",
                f"Not saved; prefer {definition.environment_variable}",
                password=True,
            ),
            lambda secret: self._discover_api_models(provider, secret) if secret else None,
        )

    @work(thread=True, exclusive=True)
    def _discover_api_models(self, provider: str, api_key: str) -> None:
        from main.api_providers import ApiProviderError, OpenAICompatibleClient

        self.call_from_thread(self._set_busy, True, f"Discovering {provider} models…")
        client = OpenAICompatibleClient(provider, api_key)
        models: list[str] = []
        try:
            allowed = self.cli.permission_manager.request(
                "api",
                "list_api_models",
                client.provider_name,
                "Request provider model list and context metadata",
            )
            if allowed:
                models = client.list_models()
        except ApiProviderError as error:
            self.call_from_thread(
                self._mount_message, f"Model discovery failed: {error}", "event-card"
            )
        self._pending_api_client = client
        self._pending_api_key = api_key
        choices = [(model, model) for model in models[:100]]
        choices.append(("__manual__", "Enter model ID and limits manually"))
        self.call_from_thread(self._show_api_model_choices, choices)
        self.call_from_thread(self._set_busy, False)

    def _show_api_model_choices(self, choices: list[tuple[str, str]]) -> None:
        self.push_screen(
            ChoiceScreen("Select API model", choices), self._choose_discovered_api
        )

    def _choose_discovered_api(self, model: str | None) -> None:
        if not model or self._pending_api_client is None:
            return
        metadata = (
            {} if model == "__manual__"
            else self._pending_api_client.model_metadata(model)
        )
        self._open_api_form(
            self._pending_api_client.provider,
            "" if model == "__manual__" else model,
            str(metadata.get("context", 32768)),
            str(metadata.get("max_tokens", 4096)),
        )

    def _open_api_form(
        self, provider: str, model: str, context: str, output: str
    ) -> None:
        self.push_screen(
            FormScreen(
                "Add or connect API profile",
                [
                    ("provider", "Provider", provider, False),
                    ("model", "Exact provider model ID", model, False),
                    ("key", "API key (never saved; blank uses environment)", "", True),
                    ("context", "Context window", context, False),
                    ("output", "Max output tokens", output, False),
                ],
            ),
            self._create_api_profile,
        )

    @work(thread=True, exclusive=True)
    def _create_api_profile(self, values: dict[str, str] | None) -> None:
        if not values:
            self._pending_api_key = ""
            self._pending_api_client = None
            return
        from main.api_providers import PROVIDERS

        provider = values["provider"].casefold()
        if provider not in PROVIDERS:
            self.call_from_thread(
                self._mount_message, "Unknown API provider.", "event-card"
            )
            return
        api_key = values["key"] or self._pending_api_key or os.environ.get(
            PROVIDERS[provider].environment_variable, ""
        )
        self._pending_api_key = ""
        self._pending_api_client = None
        self.call_from_thread(self._set_busy, True, f"Connecting {provider}…")
        try:
            context = int(values["context"])
            output = int(values["output"])
            if not 512 <= context <= 1_000_000 or not 64 <= output <= context // 2:
                raise ValueError("Invalid context or output token limit")
            success = self.cli._activate_api(
                provider,
                api_key,
                values["model"],
                context_window=context,
                max_output_tokens=output,
            )
            message = (
                f"API ready: {provider} · {values['model']} · {context:,} context"
                if success else "API activation failed."
            )
        except (OSError, ValueError) as error:
            message = f"API profile not added: {error}"
        self.call_from_thread(self._mount_message, message, "event-card")
        self.call_from_thread(self._refresh_state)
        self.call_from_thread(self._set_busy, False)

    def action_remove_api(self) -> None:
        choices = [
            (key, f"{value['provider']} · {value['model']}")
            for key, value in self.cli.api_profiles.profiles.items()
        ]
        if not choices:
            self._mount_message("No saved API profiles.", "event-card")
            return
        self.push_screen(
            ChoiceScreen("Remove API profile", choices), self._confirm_remove_api
        )

    def _confirm_remove_api(self, key: str | None) -> None:
        if key:
            self.push_screen(
                ConfirmScreen("Remove API profile", "API key is not stored."),
                lambda confirmed: self._remove_api_profile(key) if confirmed else None,
            )

    def _remove_api_profile(self, key: str) -> None:
        try:
            removed = self.cli.api_profiles.remove(key)
            message = f"Removed API profile: {removed['provider']} · {removed['model']}"
        except (KeyError, OSError, ValueError) as error:
            message = f"API profile removal failed: {error}"
        self._mount_message(message, "event-card")

    @work(thread=True, exclusive=True)
    def _activate_api_profile(self, provider: str, model: str, api_key: str) -> None:
        self.call_from_thread(self._set_busy, True, f"Connecting {provider}…")
        output = io.StringIO()
        with redirect_stdout(output):
            success = self.cli._activate_api(
                provider, api_key, model, refresh_metadata=True
            )
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
        choices = []
        for path in self.cli.session_memory.list():
            try:
                record = self.cli.session_memory.load_record(path)
                label = record.title or path.stem
            except (OSError, ValueError):
                label = path.stem
            choices.append((str(path), label))
        if not choices:
            self._mount_message("No previous workspace sessions.", "event-card")
            return
        self.push_screen(ChoiceScreen("Import session memory", choices), self._select_session)

    def _select_session(self, value: str | None) -> None:
        if value:
            path = Path(value)
            self.push_screen(
                ChoiceScreen(
                    "Session action",
                    [
                        ("resume", "Resume full session"),
                        ("import", "Import compact memory"),
                    ],
                ),
                lambda action: self._open_session(path, action),
            )

    def _open_session(self, path: Path, action: str | None) -> None:
        if action == "resume":
            self._resume_session(path)
        elif action == "import":
            self._import_session(path)

    @work(thread=True, exclusive=True)
    def _import_session(self, path: Path) -> None:
        self.call_from_thread(self._set_busy, True, "Importing session memory…")
        try:
            title = self.cli.import_session_memory(path)
            message = f"Imported memory capsule: {title}"
        except Exception as error:
            message = f"Session import failed: {error}"
        self.call_from_thread(self._mount_message, message, "event-card")
        self.call_from_thread(self._refresh_state)
        self.call_from_thread(self._set_busy, False)

    @work(thread=True, exclusive=True)
    def _resume_session(self, path: Path) -> None:
        self.call_from_thread(self._set_busy, True, "Resuming session")
        try:
            title, restored = self.cli.resume_chat_session(path)
            state = "full history restored" if restored else "archive opened; runtime history unavailable"
            message = f"Resumed: {title} · {state}"
        except Exception as error:
            message = f"Session resume failed: {error}"
        self.call_from_thread(self._mount_message, message, "event-card")
        self.call_from_thread(self._sync_session_plan)
        self.call_from_thread(self._refresh_state)
        self.call_from_thread(self._set_busy, False)

    @work(thread=True, exclusive=True)
    def _run_message(self, message: str) -> None:
        self.call_from_thread(self._set_busy, True)
        query, think_mode = self._parse_think_command(message)
        if query.startswith(("/", "!")):
            output = io.StringIO()
            try:
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
            pending_tokens = ""
            last_render = time.monotonic()
            for event in self.cli.stream_turn(query, think_mode=think_mode):
                if event.type == "token":
                    pending_tokens += event.content
                    if len(pending_tokens) < 512 and time.monotonic() - last_render < 0.05:
                        continue
                    event = AgentEvent("token", pending_tokens)
                    pending_tokens = ""
                    last_render = time.monotonic()
                elif pending_tokens:
                    self.call_from_thread(
                        self._handle_event, AgentEvent("token", pending_tokens)
                    )
                    pending_tokens = ""
                self.call_from_thread(self._handle_event, event)
            if pending_tokens:
                self.call_from_thread(
                    self._handle_event, AgentEvent("token", pending_tokens)
                )
            self.call_from_thread(self._sync_session_plan)
        self.call_from_thread(self._refresh_state)
        self.call_from_thread(self._set_busy, False)

    @staticmethod
    def _parse_think_command(message: str) -> tuple[str, bool]:
        if message.lower().startswith("/think "):
            return message.split(" ", 1)[1].strip(), True
        return message, False

    def _begin_assistant(self) -> None:
        self._assistant_text = ""
        self._response_render_truncated = False
        self._assistant = Markdown("_Thinking…_", classes="assistant-message")
        self.query_one("#timeline", VerticalScroll).mount(self._assistant)

    def _handle_event(self, event: AgentEvent) -> None:
        self._events.append(event)
        if event.type == "token":
            remaining = self.MAX_RENDERED_RESPONSE_CHARS - len(self._assistant_text)
            if remaining <= 0:
                if not self._response_render_truncated:
                    self._response_render_truncated = True
                    self._mount_message(
                        "Response display capped for TUI stability; full bounded response remains in session.",
                        "event-card",
                    )
                return
            self._assistant_text += event.content[:remaining]
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
            added = int(details.get("added_lines", 0))
            removed = int(details.get("removed_lines", 0))
            title = (
                f"Change · {details.get('path', event.summary)} "
                f"(+{added}/-{removed}){suffix}"
            )
            self._mount_collapsible(title, self._diff_preview(diff))
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

    @staticmethod
    def _diff_preview(diff: str) -> Text:
        """Render bounded unified diffs as text, never Textual markup."""
        preview = Text()
        for line in diff.splitlines(keepends=True):
            if line.startswith("+") and not line.startswith("+++"):
                preview.append(line, style="green")
            elif line.startswith("-") and not line.startswith("---"):
                preview.append(line, style="red")
            elif line.startswith("@@"):
                preview.append(line, style="bold cyan")
            else:
                preview.append(line, style="dim")
        return preview

    def _mount_collapsible(self, title: str, body: str | Text) -> None:
        self.query_one("#timeline", VerticalScroll).mount(
            Collapsible(
                Static(body, markup=False),
                title=title,
                collapsed=True,
                classes="event-card",
            )
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
        markers = {"pending": "○", "in_progress": "◐", "completed": "●", "dismissed": "×"}
        for item in self.plan_items:
            view.append(ListItem(Label(f"{markers[item.status]} {item.text}")))
        if selected is not None and self.plan_items:
            view.index = min(selected, len(self.plan_items) - 1)

    def _sync_plan_context(self) -> None:
        self.cli.task_plan_context = "\n".join(
            f"- [{item.status}] {item.text} (id: {item.id})" for item in self.plan_items
        )

    def _sync_session_plan(self) -> None:
        if self.cli.chat_session is None:
            return
        expected_name = f"{self.cli.chat_session.session_id}.json"
        if self.plan_store is None or self.plan_store.path.name != expected_name:
            self.plan_store = TaskPlanStore(
                Path.cwd(), self.cli.chat_session.session_id, root=self.state_root
            )
            self.cli.task_plan_store = self.plan_store
        self.plan_items = self.plan_store.load()
        if self.cli.agent_runtime is not None:
            self.cli.agent_runtime.task_plan_store = self.plan_store
        self._sync_plan_context()
        self._refresh_plan()


__all__ = ["ChoiceScreen", "OpenCLITui", "PermissionScreen", "TextualPermissionBroker"]
