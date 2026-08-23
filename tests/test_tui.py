from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import patch
from textual.widgets import Collapsible

from main.cli import OpenCLI, main
from main.session_memory import SessionMemoryStore
from main.task_plan import TaskPlanStore
from main.tui import ChoiceScreen, FormScreen, OpenCLITui, PermissionScreen
from main.permissions import PermissionDecision, PermissionRequest
from main.ui_events import AgentEvent


class OpenCLITuiTests(IsolatedAsyncioTestCase):
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

    async def test_narrow_terminal_hides_side_panes(self):
        with TemporaryDirectory() as directory:
            cli = OpenCLI(dry_run=True)
            cli.session_memory = SessionMemoryStore(Path.cwd(), Path(directory) / "sessions")
            app = OpenCLITui(cli, state_root=Path(directory) / "plans")
            async with app.run_test(size=(70, 28)) as pilot:
                await pilot.pause()
                self.assertEqual(app.query_one("#plan-pane").styles.display, "none")
                self.assertEqual(app.query_one("#inspector-pane").styles.display, "none")

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

    async def test_enter_and_send_button_submit_while_shift_enter_adds_line(self):
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
                    await pilot.click("#send-button")
                    await app.workers.wait_for_complete()
                    self.assertEqual(cli.stream_turn.call_count, 2)


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
