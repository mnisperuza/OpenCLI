from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import patch
from textual.containers import VerticalScroll
from textual.widgets import Button, Collapsible, Footer, Header, Markdown, OptionList, Static

from main.cli import OpenCLI, main
from main.session_memory import SessionMemoryStore
from main.task_plan import TaskPlanItem, TaskPlanStore
from main.tui import (
    ACTIVITY_LABELS, ChoiceScreen, ConfirmScreen, FormScreen, OpenCLITui,
    PermissionScreen,
)
from main.permissions import PermissionDecision, PermissionRequest
from main.ui_events import AgentEvent


class OpenCLITuiTests(IsolatedAsyncioTestCase):
    async def test_escape_during_generation_requests_transport_cancel(self):
        with TemporaryDirectory() as directory:
            cli = OpenCLI(dry_run=True)
            cli.session_memory = SessionMemoryStore(Path.cwd(), Path(directory) / "sessions")
            app = OpenCLITui(cli, state_root=Path(directory) / "plans")
            with patch.object(cli, "_request_generation_stop") as stop:
                async with app.run_test() as pilot:
                    app._set_busy(True, "Generating")
                    await pilot.press("escape")
                    await pilot.pause()
                    stop.assert_not_called()

                    app._set_busy(True, "Generating", "generation")
                    await pilot.press("escape")
                    await pilot.pause()
                    stop.assert_called_once_with()
                    self.assertIn("Stopping generation", app._activity_message)

    async def test_escape_during_model_load_requests_hard_stop(self):
        with TemporaryDirectory() as directory:
            cli = OpenCLI(dry_run=True)
            cli.session_memory = SessionMemoryStore(Path.cwd(), Path(directory) / "sessions")
            app = OpenCLITui(cli, state_root=Path(directory) / "plans")
            with patch.object(cli, "_request_generation_stop") as stop:
                async with app.run_test() as pilot:
                    app._set_busy(True, "Loading model", "model_loading")
                    await pilot.press("escape")
                    await pilot.pause()
                    stop.assert_called_once_with()
                    self.assertIn("Stopping model load", app._activity_message)

    async def test_escape_on_permission_dialog_stops_generation(self):
        with TemporaryDirectory() as directory:
            cli = OpenCLI(dry_run=True)
            cli.session_memory = SessionMemoryStore(Path.cwd(), Path(directory) / "sessions")
            app = OpenCLITui(cli, state_root=Path(directory) / "plans")
            request = PermissionRequest(
                category="file_write",
                action="write file",
                target="example.txt",
                reason="test",
                workspace=Path(directory),
            )
            with patch.object(cli, "_request_generation_stop") as stop:
                async with app.run_test() as pilot:
                    app._start_generation_activity()
                    app.push_screen(PermissionScreen(request))
                    await pilot.pause()
                    await pilot.press("escape")
                    await pilot.pause()
                    stop.assert_called_once_with()

    async def test_dynamic_activity_stops_on_first_token(self):
        with TemporaryDirectory() as directory:
            cli = OpenCLI(dry_run=True)
            cli.session_memory = SessionMemoryStore(Path.cwd(), Path(directory) / "sessions")
            app = OpenCLITui(cli, state_root=Path(directory) / "plans")
            async with app.run_test() as pilot:
                self.assertGreaterEqual(len(ACTIVITY_LABELS), 30)
                app._start_generation_activity()
                self.assertTrue(app._activity_dynamic)
                first_label = app._activity_label_index
                for _ in range(20):
                    app._refresh_activity_clock()
                self.assertEqual(app._activity_label_index, first_label)
                app._begin_assistant()
                app._handle_event(AgentEvent("token", "Hello"))
                await pilot.pause()
                self.assertFalse(app._activity_dynamic)
                self.assertEqual(app._activity_message, "Responding · Escape stops")

                app._start_generation_activity()
                self.assertNotEqual(app._activity_label_index, first_label)

    async def test_real_activity_state_overrides_random_label(self):
        with TemporaryDirectory() as directory:
            cli = OpenCLI(dry_run=True)
            cli.session_memory = SessionMemoryStore(Path.cwd(), Path(directory) / "sessions")
            app = OpenCLITui(cli, state_root=Path(directory) / "plans")
            async with app.run_test():
                app._start_generation_activity()
                app._handle_event(AgentEvent("tool", name="read_text_file"))
                self.assertFalse(app._activity_dynamic)
                self.assertIn("Using tool", app._activity_message)

    async def test_mount_shows_agent_controls_and_context(self):
        with TemporaryDirectory() as directory:
            cli = OpenCLI(dry_run=True)
            cli.session_memory = SessionMemoryStore(Path.cwd(), Path(directory) / "sessions")
            app = OpenCLITui(cli, state_root=Path(directory) / "plans")
            async with app.run_test() as pilot:
                await pilot.pause()
                state = str(app.query_one("#status-line").render())
                self.assertIn("ctx", state)
                self.assertIn("tools", state)
                self.assertIn("session", state)

    async def test_live_events_update_response_and_inspector(self):
        with TemporaryDirectory() as directory:
            cli = OpenCLI(dry_run=True)
            cli.session_memory = SessionMemoryStore(Path.cwd(), Path(directory) / "sessions")
            app = OpenCLITui(cli, state_root=Path(directory) / "plans")
            async with app.run_test() as pilot:
                app._begin_assistant()
                app._handle_event(AgentEvent("token", "Hello"))
                app._handle_event(AgentEvent("tool_result", name="read_text_file", summary="12 characters"))
                await pilot.pause()
                self.assertEqual(app._assistant_text, "Hello")
                self.assertEqual(app._events[-1].type, "tool_result")

    async def test_react_state_updates_one_progress_card(self):
        with TemporaryDirectory() as directory:
            cli = OpenCLI(dry_run=True)
            cli.session_memory = SessionMemoryStore(Path.cwd(), Path(directory) / "sessions")
            app = OpenCLITui(cli, state_root=Path(directory) / "plans")
            async with app.run_test() as pilot:
                app._begin_assistant()
                app._handle_event(
                    AgentEvent("react_state", details={"phase": "plan", "steps": 0, "max_steps": 10})
                )
                app._handle_event(
                    AgentEvent("react_state", details={"phase": "act", "steps": 1, "max_steps": 10})
                )
                await pilot.pause()
                cards = list(app.query(".react-card"))
                self.assertEqual(len(cards), 1)
                self.assertIn("act", str(cards[0].render()))

    async def test_final_response_mounts_after_tool_trace(self):
        with TemporaryDirectory() as directory:
            cli = OpenCLI(dry_run=True)
            cli.session_memory = SessionMemoryStore(Path.cwd(), Path(directory) / "sessions")
            app = OpenCLITui(cli, state_root=Path(directory) / "plans")
            async with app.run_test() as pilot:
                app._begin_assistant()
                app._handle_event(AgentEvent("token", "Planning"))
                first_response = app._assistant
                app._handle_event(
                    AgentEvent("tool_result", name="list_files", summary="3 files")
                )
                app._handle_event(AgentEvent("token", "Final answer"))
                await pilot.pause()

                timeline = app.query_one("#timeline", VerticalScroll)
                children = list(timeline.children)
                self.assertIsInstance(first_response, Markdown)
                self.assertIsInstance(children[-2], Static)
                self.assertIsInstance(children[-1], Markdown)
                self.assertIsNot(children[-1], first_response)
                self.assertEqual(app._assistant_segment_text, "Final answer")
                self.assertEqual(app._assistant_text, "PlanningFinal answer")

    async def test_user_scroll_disables_follow_until_latest_action(self):
        with TemporaryDirectory() as directory:
            cli = OpenCLI(dry_run=True)
            cli.session_memory = SessionMemoryStore(Path.cwd(), Path(directory) / "sessions")
            app = OpenCLITui(cli, state_root=Path(directory) / "plans")
            async with app.run_test(size=(100, 24)) as pilot:
                for index in range(40):
                    app._mount_message(f"History {index}", "event-card")
                await pilot.pause()
                timeline = app.query_one("#timeline", VerticalScroll)
                timeline.scroll_home(animate=False)
                await pilot.pause()

                app._begin_assistant()
                app._handle_event(AgentEvent("token", "Streaming output"))
                await pilot.pause()

                self.assertFalse(app._follow_output)
                self.assertEqual(timeline.scroll_y, 0)
                app.action_follow_latest()
                await pilot.pause()
                self.assertTrue(app._follow_output)
                self.assertGreater(timeline.scroll_y, 0)

    async def test_file_preview_treats_code_as_text_not_markup(self):
        with TemporaryDirectory() as directory:
            cli = OpenCLI(dry_run=True)
            cli.session_memory = SessionMemoryStore(Path.cwd(), Path(directory) / "sessions")
            app = OpenCLITui(cli, state_root=Path(directory) / "plans")
            async with app.run_test() as pilot:
                app._handle_event(
                    AgentEvent(
                        "file_change",
                        summary="example.py",
                        details={
                            "path": "example.py",
                            "diff": "+value = [\"markup-safe\"]\n-old = 1\n",
                            "added_lines": 1,
                            "removed_lines": 1,
                        },
                    )
                )
                await pilot.pause()
                preview = app.query_one(Collapsible)
                preview.collapsed = False
                await pilot.pause()
                self.assertFalse(preview.collapsed)
                self.assertTrue(preview.has_class("change-card"))

    def test_diff_preview_uses_line_backgrounds(self):
        preview = OpenCLITui._diff_preview("+added\n-removed\n context\n")
        styles = [str(span.style) for span in preview.spans]
        self.assertTrue(any("on #10261d" in style for style in styles))
        self.assertTrue(any("on #2a1419" in style for style in styles))

    async def test_permission_escape_denies(self):
        with TemporaryDirectory() as directory:
            cli = OpenCLI(dry_run=True)
            cli.session_memory = SessionMemoryStore(Path.cwd(), Path(directory) / "sessions")
            app = OpenCLITui(cli, state_root=Path(directory) / "plans")
            decisions = []
            request = PermissionRequest("file_write", "edit", "a.py", "test", Path.cwd())
            async with app.run_test() as pilot:
                app.push_screen(PermissionScreen(request), decisions.append)
                await pilot.pause()
                await pilot.press("escape")
                await pilot.pause()
            self.assertEqual(decisions, [PermissionDecision.DENY])

    async def test_buttonless_permission_defaults_to_deny_on_enter(self):
        with TemporaryDirectory() as directory:
            cli = OpenCLI(dry_run=True)
            cli.session_memory = SessionMemoryStore(Path.cwd(), Path(directory) / "sessions")
            app = OpenCLITui(cli, state_root=Path(directory) / "plans")
            decisions = []
            request = PermissionRequest("file_write", "edit", "a.py", "test", Path.cwd())
            async with app.run_test() as pilot:
                app.push_screen(PermissionScreen(request), decisions.append)
                await pilot.pause()
                self.assertEqual(len(app.screen.query(Button)), 0)
                await pilot.press("enter")
                await pilot.pause()
            self.assertEqual(decisions, [PermissionDecision.DENY])

    async def test_buttonless_form_and_confirmation_use_keyboard(self):
        with TemporaryDirectory() as directory:
            cli = OpenCLI(dry_run=True)
            cli.session_memory = SessionMemoryStore(Path.cwd(), Path(directory) / "sessions")
            app = OpenCLITui(cli, state_root=Path(directory) / "plans")
            forms = []
            confirms = []
            async with app.run_test() as pilot:
                app.push_screen(FormScreen("Profile", [("name", "Name", "Open", False)]), forms.append)
                await pilot.pause()
                self.assertEqual(len(app.screen.query(Button)), 0)
                await pilot.press("ctrl+enter")
                await pilot.pause()
                app.push_screen(ConfirmScreen("Remove", "Remove profile?"), confirms.append)
                await pilot.pause()
                self.assertEqual(len(app.screen.query(Button)), 0)
                await pilot.press("enter")
                await pilot.pause()
            self.assertEqual(forms, [{"name": "Open"}])
            self.assertEqual(confirms, [False])

    async def test_form_save_has_terminal_safe_shortcuts_and_enter_fallback(self):
        with TemporaryDirectory() as directory:
            cli = OpenCLI(dry_run=True)
            cli.session_memory = SessionMemoryStore(Path.cwd(), Path(directory) / "sessions")
            app = OpenCLITui(cli, state_root=Path(directory) / "plans")
            saved = []
            async with app.run_test() as pilot:
                app.push_screen(
                    FormScreen(
                        "Profile",
                        [("name", "Name", "Open", False), ("path", "Path", "model.gguf", False)],
                    ),
                    saved.append,
                )
                await pilot.pause()
                await pilot.press("enter")
                self.assertEqual(app.screen.focused.id, "form-path")
                await pilot.press("ctrl+s")
                await pilot.pause()
            self.assertEqual(saved, [{"name": "Open", "path": "model.gguf"}])

    async def test_single_stream_has_no_side_panes_or_persistent_buttons(self):
        with TemporaryDirectory() as directory:
            cli = OpenCLI(dry_run=True)
            cli.session_memory = SessionMemoryStore(Path.cwd(), Path(directory) / "sessions")
            app = OpenCLITui(cli, state_root=Path(directory) / "plans")
            async with app.run_test(size=(70, 28)) as pilot:
                await pilot.pause()
                self.assertEqual(len(app.query("#plan-pane")), 0)
                self.assertEqual(len(app.query("#inspector-pane")), 0)
                self.assertEqual(len(app.query(Button)), 0)
                self.assertEqual(len(app.query(Header)), 0)
                self.assertEqual(len(app.query(Footer)), 0)
                self.assertIn("ctx", str(app.query_one("#status-line").render()))

    async def test_model_manager_exposes_profile_crud_inside_tui(self):
        with TemporaryDirectory() as directory:
            cli = OpenCLI(dry_run=True)
            cli.session_memory = SessionMemoryStore(Path.cwd(), Path(directory) / "sessions")
            app = OpenCLITui(cli, state_root=Path(directory) / "plans")
            async with app.run_test() as pilot:
                app.action_models()
                await pilot.pause()
                self.assertIsInstance(app.screen, ChoiceScreen)
                values = {value for value, _label in app.screen.choices}
                self.assertIn("manage::add-model", values)
                self.assertIn("manage::add-api", values)
                app.pop_screen()
                app.action_add_model()
                await pilot.pause()
                self.assertIsInstance(app.screen, FormScreen)

    async def test_enter_submits_while_shift_enter_adds_line(self):
        with TemporaryDirectory() as directory:
            cli = OpenCLI(dry_run=True)
            cli.session_memory = SessionMemoryStore(Path.cwd(), Path(directory) / "sessions")
            app = OpenCLITui(cli, state_root=Path(directory) / "plans")
            with patch.object(
                cli,
                "stream_turn",
                side_effect=lambda *_args, **_kwargs: iter(
                    [AgentEvent("token", "answer"), AgentEvent("done", "answer")]
                ),
            ):
                async with app.run_test() as pilot:
                    prompt = app.query_one("#prompt")
                    prompt.text = "first"
                    await pilot.press("enter")
                    await app.workers.wait_for_complete()
                    self.assertEqual(app._assistant_text, "answer")

                    prompt.text = "line one"
                    await pilot.press("shift+enter")
                    self.assertIn("\n", prompt.text)

                    prompt.text = "second"
                    await pilot.press("enter")
                    await app.workers.wait_for_complete()
                    self.assertEqual(cli.stream_turn.call_count, 2)

    async def test_slash_autocomplete_filters_and_completes_existing_command(self):
        with TemporaryDirectory() as directory:
            cli = OpenCLI(dry_run=True)
            cli.session_memory = SessionMemoryStore(Path.cwd(), Path(directory) / "sessions")
            app = OpenCLITui(cli, state_root=Path(directory) / "plans")
            async with app.run_test() as pilot:
                prompt = app.query_one("#prompt")
                prompt.text = "/cont"
                await pilot.pause()
                suggestions = app.query_one("#command-suggestions", OptionList)
                self.assertTrue(suggestions.has_class("visible"))
                self.assertEqual(str(suggestions.highlighted_option.id), "/context")
                await pilot.press("tab")
                await pilot.pause()
                self.assertEqual(prompt.text, "/context")
                self.assertFalse(suggestions.has_class("visible"))

    async def test_thinking_updates_one_collapsed_provider_summary(self):
        with TemporaryDirectory() as directory:
            cli = OpenCLI(dry_run=True)
            cli.session_memory = SessionMemoryStore(Path.cwd(), Path(directory) / "sessions")
            app = OpenCLITui(cli, state_root=Path(directory) / "plans")
            async with app.run_test() as pilot:
                app._begin_assistant()
                app._handle_event(AgentEvent("thinking", "Checking constraints. "))
                app._handle_event(AgentEvent("thinking", "Comparing evidence."))
                await pilot.pause()
                cards = list(app.query(".thinking-card"))
                self.assertEqual(len(cards), 1)
                self.assertTrue(cards[0].collapsed)
                self.assertIn("Checking constraints", app._thinking_text)
                self.assertIn("Comparing evidence", app._thinking_text)

    async def test_plan_is_one_inline_card(self):
        with TemporaryDirectory() as directory:
            cli = OpenCLI(dry_run=True)
            cli.session_memory = SessionMemoryStore(Path.cwd(), Path(directory) / "sessions")
            app = OpenCLITui(cli, state_root=Path(directory) / "plans")
            async with app.run_test() as pilot:
                app.plan_items = [TaskPlanItem("one", "Inspect workspace", "in_progress")]
                app._refresh_plan()
                app.plan_items.append(TaskPlanItem("two", "Run tests", "pending"))
                app._refresh_plan()
                await pilot.pause()
                cards = list(app.query(".react-card"))
                self.assertEqual(len(cards), 1)
                rendered = str(cards[0].render())
                self.assertIn("Inspect workspace", rendered)
                self.assertIn("Run tests", rendered)

    async def test_timeline_mounts_are_windowed_for_long_sessions(self):
        with TemporaryDirectory() as directory:
            cli = OpenCLI(dry_run=True)
            cli.session_memory = SessionMemoryStore(Path.cwd(), Path(directory) / "sessions")
            app = OpenCLITui(cli, state_root=Path(directory) / "plans")
            async with app.run_test() as pilot:
                for index in range(app.MAX_MOUNTED_WIDGETS + 20):
                    app._mount_message(f"event {index}", "event-card")
                await pilot.pause()
                timeline = app.query_one("#timeline", VerticalScroll)
                self.assertLessEqual(len(timeline.children), app.MAX_MOUNTED_WIDGETS)
                self.assertTrue(app._history_windowed)


class TuiCommandTests(TestCase):
    def test_think_command_keeps_prompt_and_enables_thinking(self):
        prompt, think_mode = OpenCLITui._parse_think_command("/think inspect this")
        self.assertEqual(prompt, "inspect this")
        self.assertTrue(think_mode)

    def test_normal_prompt_does_not_enable_thinking(self):
        prompt, think_mode = OpenCLITui._parse_think_command("inspect this")
        self.assertEqual(prompt, "inspect this")
        self.assertFalse(think_mode)

    def test_cli_launches_textual_app_only_when_requested(self):
        cli = OpenCLI(dry_run=True)
        with patch("main.tui.OpenCLITui") as tui:
            cli.run_tui()
        tui.assert_called_once_with(cli, api_start=False)
        tui.return_value.run.assert_called_once_with()

    def test_textual_is_default_and_cli_flag_keeps_classic_interface(self):
        with (
            patch("main.cli.WorkspaceTrust.confirm", return_value=True),
            patch("main.cli.OpenCLI") as opencli,
            patch("sys.argv", ["opencli"]),
        ):
            main()
        opencli.return_value.run_tui.assert_called_once_with(api_start=False)
        opencli.return_value.run.assert_not_called()

        with (
            patch("main.cli.WorkspaceTrust.confirm", return_value=True),
            patch("main.cli.OpenCLI") as opencli,
            patch("sys.argv", ["opencli", "--cli"]),
        ):
            main()
        opencli.return_value.run.assert_called_once_with(api_start=False)
        opencli.return_value.run_tui.assert_not_called()

    def test_agent_event_normalizes_chunks(self):
        event = AgentEvent.from_chunk(
            {"type": "tool", "name": "read_text_file", "arguments": {"path": "a.py"}}
        )
        self.assertEqual(event.name, "read_text_file")
        self.assertEqual(event.arguments["path"], "a.py")

    def test_manual_plan_round_trip(self):
        with TemporaryDirectory() as directory:
            store = TaskPlanStore(Path.cwd(), "session", Path(directory))
            items = []
            TaskPlanStore.add(items, "Run focused tests")
            items[0].status = "in_progress"
            store.save(items)
            loaded = store.load()
        self.assertEqual(loaded[0].text, "Run focused tests")
        self.assertEqual(loaded[0].status, "in_progress")

    def test_plan_item_can_be_completed_or_dismissed_by_stable_id(self):
        with TemporaryDirectory() as directory:
            store = TaskPlanStore(Path.cwd(), "session", Path(directory))
            items = []
            item = TaskPlanStore.add(items, "Remove obsolete step")
            store.save(items)
            updated = store.update_status(item.id, "dismissed")
            loaded = store.load()
        self.assertEqual(updated.status, "dismissed")
        self.assertEqual(loaded[0].id, item.id)
        self.assertEqual(loaded[0].status, "dismissed")

    def test_stream_turn_exposes_typed_live_events_and_records_usage(self):
        class Engine:
            model = object()
            MODELS = {"auto": {"family": "auto", "has_thinking": False}}

            @staticmethod
            def prepare_input_payload(prompt):
                return SimpleNamespace(
                    enhanced_prompt=prompt,
                    file_paths=[],
                    clipboard_image_used=False,
                    image_attachments=[],
                )

        class Runtime:
            @staticmethod
            def generate_stream(_prompt):
                yield {"type": "token", "content": "hello"}
                yield {"type": "done", "content": "hello"}

            @staticmethod
            def export_transcript():
                return ""

        cli = OpenCLI(dry_run=True)
        cli.engine = Engine()
        cli.agent_runtime = Runtime()
        cli.interrupt_handler = SimpleNamespace(reset=lambda: None)
        with patch.object(
            cli, "_context_snapshot", return_value=SimpleNamespace(used_tokens=4, estimated=True)
        ):
            events = list(cli.stream_turn("hi"))

        self.assertTrue(all(isinstance(event, AgentEvent) for event in events))
        self.assertEqual(events[0].content, "hello")
        self.assertEqual(cli.context_accounting.usage.turns, 1)
