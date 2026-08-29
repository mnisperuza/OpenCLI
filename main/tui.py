"""Professional optional Textual workspace for OpenCLI."""

from __future__ import annotations

from concurrent.futures import Future
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import random
import re
import threading
import time
from typing import TYPE_CHECKING

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.events import Key
from textual.screen import ModalScreen
from textual.widgets import (
    Collapsible, Input, Label, Markdown, OptionList, Static, TextArea,
)
from textual.widgets.option_list import Option
from rich.text import Text

from main.command_registry import match_commands
from main.permissions import PermissionDecision, PermissionRequest
from main.task_plan import TaskPlanItem, TaskPlanStore
from main.ui_events import AgentEvent

if TYPE_CHECKING:
    from main.cli import OpenCLI


ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

ACTIVITY_LABELS = (
    "Thinking", "Computing", "Reviewing", "Spinning", "Pontificating",
    "Mapping", "Inspecting", "Tracing", "Comparing", "Weighing options",
    "Following clues", "Checking details", "Connecting dots", "Indexing context",
    "Reading signals", "Testing assumptions", "Untangling threads",
    "Sharpening plan", "Scanning workspace", "Gathering evidence",
    "Calibrating tools", "Parsing intent", "Stacking blocks", "Finding edges",
    "Cross-checking", "Working through it", "Pondering", "Drafting approach",
    "Resolving paths", "Aligning context", "Keeping watch", "Simmering ideas",
    "Exploring routes", "Chasing details", "Making sense", "Building context",
    "Sorting evidence", "Checking constraints", "Charting next move", "Deliberating",
)
ACTIVITY_GRADIENT = ("#4f8f8c", "#62a7a3", "#78bfba", "#91d1cb", "#78bfba", "#62a7a3")


class PermissionScreen(ModalScreen[PermissionDecision]):
    BINDINGS = [Binding("escape", "deny", "Deny", priority=True)]

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
            yield OptionList(
                Option("Deny", id=PermissionDecision.DENY.value),
                Option("Allow once", id=PermissionDecision.ALLOW_ONCE.value),
                Option("Allow for session", id=PermissionDecision.ALLOW_SESSION.value),
                Option("Always allow in workspace", id=PermissionDecision.ALWAYS_ALLOW.value),
                id="permission-actions",
            )

    def on_mount(self) -> None:
        actions = self.query_one("#permission-actions", OptionList)
        actions.highlighted = 0
        actions.focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(PermissionDecision(str(event.option.id)))

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
    BINDINGS = [Binding("escape", "cancel", "Cancel", priority=True)]
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
            yield Static("Enter save · Esc cancel", classes="dialog-help")

    def on_mount(self) -> None:
        self.query_one(Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip() or None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class ChoiceScreen(ModalScreen[str | None]):
    BINDINGS = [Binding("escape", "cancel", "Cancel", priority=True)]
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
            yield Static("Enter select · Esc cancel", classes="dialog-help")

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(str(event.option.id))

    def action_cancel(self) -> None:
        self.dismiss(None)


class ConfirmScreen(ModalScreen[bool]):
    BINDINGS = [
        Binding("escape,n", "cancel", "Cancel", priority=True),
        Binding("y", "confirm", "Confirm", priority=True),
    ]
    def __init__(self, title: str, message: str) -> None:
        super().__init__()
        self.dialog_title = title
        self.message = message

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-dialog"):
            yield Label(self.dialog_title, classes="dialog-title")
            yield Static(self.message)
            yield OptionList(
                Option("Cancel", id="cancel"),
                Option("Confirm", id="confirm"),
                id="confirm-actions",
            )

    def on_mount(self) -> None:
        actions = self.query_one("#confirm-actions", OptionList)
        actions.highlighted = 0
        actions.focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option.id == "confirm")

    def action_cancel(self) -> None:
        self.dismiss(False)

    def action_confirm(self) -> None:
        self.dismiss(True)


class FormScreen(ModalScreen[dict[str, str] | None]):
    """Small reusable Textual form; validation remains in domain registries."""

    BINDINGS = [
        Binding("ctrl+s", "save", "Save", priority=True),
        Binding("ctrl+enter", "save", "Save", priority=True),
        Binding("ctrl+j", "save", "Save", show=False, priority=True),
        Binding("escape", "cancel", "Cancel", priority=True),
    ]

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
                yield FormInput(value=default, password=password, id=f"form-{name}")
            yield Static("Ctrl+S save · Enter next/save · Esc cancel", classes="dialog-help")

    def on_mount(self) -> None:
        self.query_one(Input).focus()

    def action_save(self) -> None:
        self.dismiss(
            {
                name: self.query_one(f"#form-{name}", Input).value.strip()
                for name, *_ in self.fields
            }
        )

    def advance_or_save(self, source: Input) -> None:
        """Make model/API forms reliable across terminal key encodings."""
        fields = list(self.query(FormInput))
        try:
            index = fields.index(source)
        except ValueError:
            self.action_save()
            return
        if index + 1 < len(fields):
            fields[index + 1].focus()
        else:
            self.action_save()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        self.advance_or_save(event.input)

    def action_cancel(self) -> None:
        self.dismiss(None)


class FormInput(Input):
    """Single-line form input with terminal-safe save shortcuts."""

    def on_key(self, event: Key) -> None:
        if event.key in {"ctrl+enter", "ctrl+j", "ctrl+s"}:
            event.prevent_default()
            event.stop()
            screen = self.screen
            if isinstance(screen, FormScreen):
                screen.action_save()


class PromptTextArea(TextArea):
    """Multiline editor with reliable agent-style submit keys."""

    def on_key(self, event: Key) -> None:
        # Many terminals encode Ctrl+Enter as Enter or Ctrl+J. Enter therefore
        # submits consistently; Shift+Enter remains TextArea's newline action.
        if self.app.command_suggestions_visible:
            if event.key in {"up", "down"}:
                event.prevent_default()
                event.stop()
                self.app.move_command_selection(-1 if event.key == "up" else 1)
                return
            if event.key in {"tab", "enter", "ctrl+enter", "ctrl+j"}:
                event.prevent_default()
                event.stop()
                self.app.complete_selected_command()
                return
            if event.key == "escape":
                event.prevent_default()
                event.stop()
                self.app.hide_command_suggestions()
                return
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
    MAX_MOUNTED_WIDGETS = 240
    CSS = """
    Screen { background: #080b0d; color: #c8d0d0; }
    Screen.reduced-color { background: black; color: white; }
    Screen.reduced-color #identity, Screen.reduced-color #status-line { background: #080808; }
    #identity { height: 1; padding: 0 2; background: #0c1114; color: #8aa19d; text-style: bold; }
    #status-line {
        height: 1; padding: 0 2; background: #0c1114; color: #657673;
    }
    #timeline { width: 1fr; height: 1fr; padding: 1 3 2 3; background: #080b0d; }
    .pane-title, .dialog-title { color: #78bda8; text-style: bold; margin-bottom: 1; }
    .user-message {
        background: #111a1e; color: #e0e6e5; border-left: thick #58ad94;
        padding: 1 2; margin: 1 0;
    }
    .assistant-message { color: #d2d9d8; padding: 1 2; margin-bottom: 1; }
    .event-card {
        background: #0d1316; color: #82908f; border-left: tall #26363a;
        padding: 0 1; margin: 0 0 1 1;
    }
    .status-card { color: #8b9c9a; }
    .react-card { background: #10171a; color: #75bda8; border-left: tall #397d69; }
    .tool-card { background: #101519; color: #9aaba9; border-left: tall #41606a; }
    .result-card { color: #78aa90; border-left: tall #365f4a; }
    .change-card { background: #0d1514; color: #8ebfaf; border-left: tall #4a8b74; }
    .thinking-card { background: #15130f; color: #c8ae70; border-left: tall #856b36; }
    .error-card { background: #1c1013; color: #e18b96; border-left: tall #9d4654; }
    #activity { height: 1; background: #0d1215; color: #657371; padding: 0 2; }
    #activity.busy { background: #17150f; color: #d0b46f; text-style: bold; }
    #activity.ready { color: #66847b; }
    #command-suggestions {
        display: none; height: auto; max-height: 10; margin: 0 2;
        background: #10171a; border: round #31534d;
    }
    #command-suggestions.visible { display: block; }
    #command-suggestions > .option-list--option { color: #9aa9a6; }
    #command-suggestions > .option-list--option-highlighted {
        background: #183129; color: #dce7e3; text-style: bold;
    }
    #composer-shell { height: auto; background: #090d0f; border-top: solid #1b272a; }
    #prompt { height: 6; background: #0c1114; border: round #31534d; margin: 0 1; padding: 0 1; }
    TextArea:focus { border: round #58ad94; }
    #prompt-help { height: 1; padding: 0 2; color: #53615f; }
    ModalScreen { align: center middle; background: rgba(0, 0, 0, 0.65); }
    #permission-dialog, #text-dialog, #choice-dialog, #confirm-dialog, #form-dialog {
        width: 72; max-width: 92%; height: auto; max-height: 85%;
        border: round #477c6d; background: #101619; padding: 1 2;
    }
    #form-dialog { width: 80; height: 90%; }
    .form-label { color: #9ba9a7; margin-top: 1; }
    #permission-details { height: auto; }
    #choice-list { height: 18; }
    #permission-actions, #confirm-actions { height: auto; max-height: 10; margin-top: 1; }
    .dialog-help { height: 1; color: #667572; margin-top: 1; }
    """
    BINDINGS = [
        Binding("escape", "stop", "Stop", priority=True),
        ("ctrl+g", "follow_latest", "Latest"),
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
        self._assistant_segment_text = ""
        self._response_render_truncated = False
        self._follow_output = True
        self._events: list[AgentEvent] = []
        self._react_card: Static | None = None
        self._plan_card: Static | None = None
        self._thinking_card: Collapsible | None = None
        self._thinking_body: Static | None = None
        self._thinking_text = ""
        self._activity_message = "Ready"
        self._activity_dynamic = False
        self._activity_label_index = 0
        self._activity_gradient_phase = 0
        self._busy_started = 0.0
        self._layout_width = 120
        self._command_matches = ()
        self._suppress_suggestions_once = False
        self._history_windowed = False
        self.permission_broker = TextualPermissionBroker(self)
        self.plan_store: TaskPlanStore | None = None
        self.plan_items: list[TaskPlanItem] = []
        self._pending_api_client = None
        self._pending_api_key = ""

    def compose(self) -> ComposeResult:
        yield Static("OpenCLI", id="identity")
        yield VerticalScroll(id="timeline")
        with Vertical(id="composer-shell"):
            yield OptionList(id="command-suggestions")
            yield Static("Ready", id="activity")
            yield PromptTextArea(id="prompt")
            yield Static(
                "/ commands · Enter send · Shift+Enter newline · Esc stop",
                id="prompt-help",
            )
            yield Static(id="status-line")

    def on_mount(self) -> None:
        if self.cli.chat_session is None:
            self.cli.chat_session = self.cli.session_memory.create()
        self.plan_store = TaskPlanStore(
            self.cli.workspace_context.root,
            self.cli.chat_session.session_id,
            root=self.state_root,
        )
        self.cli.task_plan_store = self.plan_store
        self.plan_items = self.plan_store.load()
        if self.cli.agent_runtime is not None:
            self.cli.agent_runtime.task_plan_store = self.plan_store
        self._sync_plan_context()
        self.cli.permission_manager.approval_callback = self.permission_broker.request
        if getattr(self.console, "color_system", None) in {None, "standard", "windows"}:
            self.screen.add_class("reduced-color")
        self._mount_message(
            "OpenCLI agent workspace\nType / for commands. Tools and permissions remain workspace-scoped.",
            "assistant-message",
        )
        self._refresh_plan()
        self._refresh_state()
        self.set_interval(0.25, self._refresh_activity_clock)
        self.query_one("#prompt", PromptTextArea).focus()
        if self.api_start:
            self.call_after_refresh(self._start_default_api)

    def on_unmount(self) -> None:
        self.permission_broker.close()
        self.cli._save_chat_session()

    def on_resize(self, event) -> None:
        self._layout_width = event.size.width
        self._refresh_state()

    def action_submit(self) -> None:
        if self._busy:
            return
        prompt = self.query_one("#prompt", PromptTextArea)
        message = prompt.text.strip()
        if not message:
            return
        self.hide_command_suggestions()
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
        if message.lower() == "/memory":
            self.action_sessions()
            return
        if message.lower() in {"/clear", "/cls"}:
            self.action_clear_timeline()
            return
        self._mount_message(f"You\n{message}", "user-message")
        self._run_message(message)

    @property
    def command_suggestions_visible(self) -> bool:
        try:
            return self.query_one("#command-suggestions", OptionList).has_class("visible")
        except Exception:
            return False

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.text_area.id != "prompt":
            return
        if self._suppress_suggestions_once:
            self._suppress_suggestions_once = False
            self.hide_command_suggestions()
            return
        text = event.text_area.text.strip()
        if not text.startswith("/") or "\n" in text:
            self.hide_command_suggestions()
            return
        self._command_matches = match_commands(text)
        suggestions = self.query_one("#command-suggestions", OptionList)
        suggestions.clear_options()
        suggestions.add_options(
            [
                Option(f"{spec.usage}\n  {spec.description}", id=spec.command)
                for spec in self._command_matches
            ]
        )
        if self._command_matches:
            suggestions.highlighted = 0
            suggestions.add_class("visible")
        else:
            suggestions.remove_class("visible")

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id == "command-suggestions":
            self.complete_selected_command()

    def move_command_selection(self, offset: int) -> None:
        suggestions = self.query_one("#command-suggestions", OptionList)
        if offset < 0:
            suggestions.action_cursor_up()
        else:
            suggestions.action_cursor_down()

    def complete_selected_command(self) -> None:
        suggestions = self.query_one("#command-suggestions", OptionList)
        option = suggestions.highlighted_option
        if option is None:
            self.hide_command_suggestions()
            return
        spec = next(
            (item for item in self._command_matches if item.command == str(option.id)),
            None,
        )
        if spec is None:
            return
        prompt = self.query_one("#prompt", PromptTextArea)
        self._suppress_suggestions_once = True
        prompt.text = spec.completion
        prompt.move_cursor((0, len(spec.completion)))
        prompt.focus()
        self.hide_command_suggestions()

    def hide_command_suggestions(self) -> None:
        try:
            self.query_one("#command-suggestions", OptionList).remove_class("visible")
        except Exception:
            pass

    def action_compact(self) -> None:
        if not self._busy:
            self._run_message("/compact")

    def action_clear_timeline(self) -> None:
        self.query_one("#timeline", VerticalScroll).remove_children()
        self._react_card = None
        self._plan_card = None
        self._thinking_card = None
        self._thinking_body = None
        self._history_windowed = False

    def action_follow_latest(self) -> None:
        """Resume live-output following after inspecting older content."""
        self._follow_output = True
        self.query_one("#timeline", VerticalScroll).scroll_end(animate=False)

    def action_stop(self) -> None:
        active_screen = self.screen
        if isinstance(active_screen, PermissionScreen):
            active_screen.action_deny()
            return
        if isinstance(active_screen, (TextPromptScreen, ChoiceScreen, ConfirmScreen, FormScreen)):
            active_screen.action_cancel()
            return
        if self._busy:
            self.cli._request_generation_stop()
            self._activity_dynamic = False
            self._activity_message = "Cancelling model and closing stream"
            self.query_one("#activity", Static).update(
                "Cancelling model and closing stream…"
            )

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
            self._mount_message("No saved API profile. Use /model or /api.", "event-card")
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
                    ("reasoning_control", "Native reasoning control: none/chat_template_kwargs", "none", False),
                    ("reasoning_default", "Reasoning default: off/low/medium/high", "off", False),
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
                reasoning_control=values["reasoning_control"],
                reasoning_default=values["reasoning_default"],
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
        query, think_mode = self._parse_think_command(message)
        if query.startswith(("/", "!")):
            self.call_from_thread(self._set_busy, True, "Running command · Escape stops")
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
            self.call_from_thread(self._start_generation_activity)
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
        self._assistant_segment_text = ""
        self._response_render_truncated = False
        self._assistant = None
        self._react_card = None
        self._thinking_card = None
        self._thinking_body = None
        self._thinking_text = ""
        self._follow_output = True

    def _begin_assistant_segment(self) -> None:
        """Mount response text at its true chronological stream position."""
        self._assistant_segment_text = ""
        self._assistant = Markdown("", classes="assistant-message")
        self.query_one("#timeline", VerticalScroll).mount(self._assistant)

    @staticmethod
    def _near_timeline_end(timeline: VerticalScroll) -> bool:
        return timeline.scroll_y >= max(0, timeline.max_scroll_y - 1)

    def _handle_event(self, event: AgentEvent) -> None:
        timeline = self.query_one("#timeline", VerticalScroll)
        if self._follow_output and not self._near_timeline_end(timeline):
            self._follow_output = False
        self._events.append(event)
        if event.type == "token":
            if self._activity_dynamic:
                self._activity_dynamic = False
                self._activity_message = "Responding · Escape stops"
            remaining = self.MAX_RENDERED_RESPONSE_CHARS - len(self._assistant_text)
            if remaining <= 0:
                if not self._response_render_truncated:
                    self._response_render_truncated = True
                    self._mount_message(
                        "Response display capped for TUI stability; full bounded response remains in session.",
                        "event-card",
                    )
                return
            content = event.content[:remaining]
            if self._assistant is None:
                self._begin_assistant_segment()
            self._assistant_text += content
            self._assistant_segment_text += content
            if self._assistant:
                self._assistant.update(self._assistant_segment_text)
        elif event.type == "status":
            self._set_activity_override(event.content)
            self._assistant = None
            self._mount_message(f"Status · {event.content}", "event-card status-card")
        elif event.type == "thinking":
            self._assistant = None
            summary = event.content.strip()
            if summary:
                remaining = max(0, 8_000 - len(self._thinking_text))
                self._thinking_text += summary[:remaining]
                if self._thinking_card is None:
                    self._thinking_body = Static(self._thinking_text, markup=False)
                    self._thinking_card = Collapsible(
                        self._thinking_body,
                        title="Thinking · provider summary",
                        collapsed=True,
                        classes="event-card thinking-card",
                    )
                    timeline.mount(self._thinking_card)
                    self._prune_timeline()
                elif self._thinking_body is not None:
                    self._thinking_body.update(self._thinking_text)
        elif event.type == "react_state":
            self._assistant = None
            details = dict(event.details)
            phase = str(details.get("phase") or event.content or "ready")
            steps = int(details.get("steps", 0))
            maximum = int(details.get("max_steps", 0))
            label = f"ReAct · {phase} · {steps}/{maximum}"
            self._set_activity_override(label)
            if self._react_card is None:
                self._react_card = self._mount_message(
                    label, "event-card react-card"
                )
            else:
                self._react_card.update(label)
            self._refresh_state()
        elif event.type == "tool":
            self._set_activity_override(f"Using tool · {event.name}")
            self._assistant = None
            details = json.dumps(dict(event.arguments), indent=2, default=str)
            self._mount_collapsible(
                f"Tool · {event.name}", details, "event-card tool-card"
            )
        elif event.type == "tool_result":
            self._set_activity_override(f"Tool complete · {event.name}")
            self._assistant = None
            text = f"Result · {event.name}: {event.summary or 'complete'}"
            self._mount_message(text, "event-card result-card")
        elif event.type == "file_change":
            self._assistant = None
            details = dict(event.details)
            diff = str(details.get("diff", "")) or "Diff unavailable"
            suffix = " (truncated)" if details.get("truncated") else ""
            added = int(details.get("added_lines", 0))
            removed = int(details.get("removed_lines", 0))
            title = (
                f"Change · {details.get('path', event.summary)} "
                f"(+{added}/-{removed}){suffix}"
            )
            self._mount_collapsible(
                title, self._diff_preview(diff), "event-card change-card"
            )
        elif event.type == "task_plan":
            self._assistant = None
            self._sync_session_plan()
        elif event.type == "error":
            self._set_activity_override("Generation failed")
            self._assistant = None
            self._mount_message(f"Error · {event.content}", "event-card error-card")
        elif event.type == "done" and not self._assistant_text and event.content:
            self._assistant_text = event.content
            if self._assistant is None:
                self._begin_assistant_segment()
            self._assistant_segment_text = event.content
            if self._assistant:
                self._assistant.update(self._assistant_segment_text)
        if self._follow_output:
            timeline.scroll_end(animate=False)

    def _mount_message(self, text: str, classes: str) -> Static:
        message = Static(text, classes=classes)
        self.query_one("#timeline", VerticalScroll).mount(message)
        self._prune_timeline()
        return message

    @staticmethod
    def _diff_preview(diff: str) -> Text:
        """Render bounded unified diffs with full semantic foreground/background styles."""
        preview = Text()
        for line in diff.splitlines(keepends=True):
            if line.startswith("+") and not line.startswith("+++"):
                preview.append(line, style="#9bd8b4 on #10261d")
            elif line.startswith("-") and not line.startswith("---"):
                preview.append(line, style="#efa0aa on #2a1419")
            elif line.startswith("@@"):
                preview.append(line, style="bold #79b8d1 on #102029")
            elif line.startswith(("+++", "---")):
                preview.append(line, style="bold #8ca3a0 on #151d20")
            else:
                preview.append(line, style="#778482 on #0b1012")
        return preview

    def _mount_collapsible(
        self,
        title: str,
        body: str | Text,
        classes: str = "event-card",
    ) -> Collapsible:
        card = Collapsible(
            Static(body, markup=False),
            title=title,
            collapsed=True,
            classes=classes,
        )
        self.query_one("#timeline", VerticalScroll).mount(card)
        self._prune_timeline()
        return card

    def _prune_timeline(self) -> None:
        timeline = self.query_one("#timeline", VerticalScroll)
        excess = len(timeline.children) - self.MAX_MOUNTED_WIDGETS
        if excess <= 0:
            return
        protected = {self._assistant, self._react_card, self._plan_card, self._thinking_card}
        for child in list(timeline.children):
            if excess <= 0:
                break
            if child in protected:
                continue
            child.remove()
            excess -= 1
            self._history_windowed = True

    def _set_busy(self, busy: bool, message: str | None = None) -> None:
        self._busy = busy
        self._activity_dynamic = False
        self._activity_message = message or (
            "Working · Escape stops" if busy else "Ready"
        )
        self._busy_started = time.monotonic() if busy else 0.0
        activity = self.query_one("#activity", Static)
        activity.set_class(busy, "busy")
        activity.set_class(not busy, "ready")
        activity.update(self._activity_message)

    def _start_generation_activity(self) -> None:
        self._busy = True
        self._activity_dynamic = True
        # Keep one label for the entire prompt and avoid repeating it next turn.
        candidate = random.randrange(len(ACTIVITY_LABELS) - 1)
        if candidate >= self._activity_label_index:
            candidate += 1
        self._activity_label_index = candidate
        self._activity_gradient_phase = random.randrange(len(ACTIVITY_GRADIENT))
        self._busy_started = time.monotonic()
        activity = self.query_one("#activity", Static)
        activity.set_class(True, "busy")
        activity.set_class(False, "ready")
        self._render_activity(activity)

    def _set_activity_override(self, message: str) -> None:
        if not self._busy:
            return
        self._activity_dynamic = False
        self._activity_message = f"{message} · Escape stops"

    def _render_activity(self, activity: Static) -> None:
        elapsed = max(0.0, time.monotonic() - self._busy_started)
        if not self._activity_dynamic:
            activity.update(f"{self._activity_message} · {elapsed:0.1f}s")
            return
        label = ACTIVITY_LABELS[self._activity_label_index]
        text = Text()
        for index, character in enumerate(f"{label}…"):
            color = ACTIVITY_GRADIENT[
                (index + self._activity_gradient_phase) % len(ACTIVITY_GRADIENT)
            ]
            text.append(character, style=f"bold {color}")
        text.append(f" · {elapsed:0.1f}s · Escape stops", style="#657371")
        activity.update(text)

    def _refresh_activity_clock(self) -> None:
        if not self._busy:
            return
        if self._activity_dynamic:
            self._activity_gradient_phase = (
                self._activity_gradient_phase + 1
            ) % len(ACTIVITY_GRADIENT)
        self._render_activity(self.query_one("#activity", Static))

    def _refresh_state(self) -> None:
        snapshot = self.cli._context_snapshot()
        usage = self.cli.context_accounting.usage
        filled = min(16, max(0, round(snapshot.percent_used * 16 / 100)))
        meter = "█" * filled + "░" * (16 - filled)
        session = self.cli.chat_session.session_id if self.cli.chat_session else "not started"
        react = self.cli.react_status()
        react_label = (
            f"{react['phase']} {react['steps']}/{react['max_steps']}"
            if self.cli.react_enabled
            else "off"
        )
        context_marker = "~" if snapshot.estimated else ""
        usage_marker = "~" if usage.estimated_turns else ""
        cwd = self.cli.workspace_context.relative_path()
        self.query_one("#identity", Static).update(
            f"OpenCLI · {snapshot.profile.display_name} · {cwd}"
        )
        if self._layout_width < 60:
            status = f"ctx {snapshot.percent_used:.0f}% · react {react['phase']}"
        elif self._layout_width < 80:
            status = (
                f"ctx {context_marker}{snapshot.used_tokens:,}/{snapshot.profile.context_window:,} "
                f"· usage {usage_marker}{usage.total_tokens:,} · tools "
                f"{'on' if self.cli.tools_enabled else 'off'} · react {react_label}"
            )
        else:
            history = " · older UI history windowed" if self._history_windowed else ""
            status = (
                f"ctx [{meter}] {snapshot.percent_used:.0f}% "
                f"{context_marker}{snapshot.used_tokens:,}/{snapshot.profile.context_window:,} "
                f"· usage {usage_marker}{usage.total_tokens:,} "
                f"· tools {'on' if self.cli.tools_enabled else 'off'} "
                f"· web {'on' if self.cli.permission_manager.web_enabled else 'off'} "
                f"· react {react_label} · sandbox {self.cli.sandbox.backend} "
                f"· session {session[:8]}{history}"
            )
        self.query_one("#status-line", Static).update(status)

    def _refresh_plan(self, selected: int | None = None) -> None:
        if not self.plan_items:
            if self._plan_card is not None:
                self._plan_card.remove()
                self._plan_card = None
            return
        markers = {"pending": "○", "in_progress": "◐", "completed": "●", "dismissed": "×"}
        content = "Plan\n" + "\n".join(
            f"{markers.get(item.status, '○')} {item.text}" for item in self.plan_items
        )
        if self._plan_card is None:
            self._plan_card = self._mount_message(content, "event-card react-card")
        else:
            self._plan_card.update(content)

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
                self.cli.workspace_context.root,
                self.cli.chat_session.session_id,
                root=self.state_root,
            )
            self.cli.task_plan_store = self.plan_store
        self.plan_items = self.plan_store.load()
        if self.cli.agent_runtime is not None:
            self.cli.agent_runtime.task_plan_store = self.plan_store
        self._sync_plan_context()
        self._refresh_plan()


__all__ = ["ChoiceScreen", "OpenCLITui", "PermissionScreen", "TextualPermissionBroker"]
