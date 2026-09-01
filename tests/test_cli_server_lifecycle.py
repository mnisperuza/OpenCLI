from unittest import TestCase
from threading import Event
from unittest.mock import Mock, patch

from main.cli import EscapeInterruptWatcher, OpenCLI
from main.permissions import PermissionDecision


class FakeEngine:
    def __init__(self):
        self.unload_calls = 0

    def unload_model(self):
        self.unload_calls += 1


class ServerLifecycleTests(TestCase):
    def test_escape_watcher_calls_shared_cancellation_callback(self):
        called = Event()
        watcher = EscapeInterruptWatcher(called.set, poll_seconds=0.001)

        with patch("main.cli.check_for_esc", return_value=True):
            watcher.start()
            self.assertTrue(called.wait(0.2))
        watcher.stop()

    @patch("main.cli._thread.interrupt_main")
    def test_escape_uses_engine_stop_and_main_interrupt(self, mocked_interrupt):
        cli = OpenCLI(dry_run=True)
        cli.engine = Mock()

        cli._interrupt_from_escape()

        cli.engine.stop_generation.assert_called_once_with()
        mocked_interrupt.assert_called_once_with()

    def test_runtime_cancellation_does_not_stop_engine_twice(self):
        cli = OpenCLI(dry_run=True)
        cli.engine = Mock()
        cli.agent_runtime = Mock()

        cli._request_generation_stop()

        cli.agent_runtime.request_cancel.assert_called_once_with()
        cli.engine.stop_generation.assert_not_called()

    def make_cli(self):
        cli = object.__new__(OpenCLI)
        cli.engine = FakeEngine()
        cli.agent_runtime = object()
        cli.server_stopped_by_user = False
        return cli

    def test_endserver_unloads_model_but_keeps_cli_running(self):
        cli = self.make_cli()

        with patch("builtins.print"):
            self.assertTrue(cli.handle_command("/endserver"))

        self.assertEqual(cli.engine.unload_calls, 1)
        self.assertIsNone(cli.agent_runtime)
        self.assertTrue(cli.server_stopped_by_user)

    def test_exit_unloads_model_and_exits(self):
        cli = self.make_cli()

        with patch("builtins.print"):
            self.assertFalse(cli.handle_command("/exit"))

        self.assertEqual(cli.engine.unload_calls, 1)
        self.assertFalse(cli.server_stopped_by_user)

    @patch("builtins.input", return_value="yes")
    def test_restart_confirmation_accepts_yes(self, _mocked_input):
        self.assertTrue(self.make_cli().confirm_server_restart())

    @patch("builtins.input", return_value="n")
    def test_restart_confirmation_rejects_no(self, _mocked_input):
        self.assertFalse(self.make_cli().confirm_server_restart())

    def test_dry_run_shell_command_does_not_execute(self):
        cli = OpenCLI(dry_run=True)
        cli.sandbox.run = Mock()

        with patch("builtins.print"):
            self.assertTrue(cli.handle_command("!pytest"))

        cli.sandbox.run.assert_not_called()

    def test_web_command_disables_network_tools(self):
        cli = OpenCLI()

        with patch("builtins.print"):
            self.assertTrue(cli.handle_command("/web off"))

        self.assertFalse(cli.permission_manager.web_enabled)

    def test_denied_shell_command_does_not_execute(self):
        cli = OpenCLI()
        cli.sandbox_enabled = True
        cli.sandbox.run = Mock()
        cli.permission_manager.approval_callback = (
            lambda _request: PermissionDecision.DENY
        )

        with patch("builtins.print"):
            self.assertTrue(cli.handle_command("!pytest"))

        cli.sandbox.run.assert_not_called()
