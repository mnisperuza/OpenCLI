from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from main.engine import FileHandler
from main.permissions import PermissionDecision, PermissionManager


class PermissionManagerTests(TestCase):
    def test_allow_session_prompts_once_per_category(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            decisions = []

            def approve(request):
                decisions.append(request)
                return PermissionDecision.ALLOW_SESSION

            manager = PermissionManager(
                root,
                state_file=root / "permissions.json",
                approval_callback=approve,
            )

            self.assertTrue(manager.request("web", "web_search", "one", "test"))
            self.assertTrue(manager.request("web", "web_fetch", "two", "test"))
            self.assertEqual(len(decisions), 1)

    def test_always_allow_persists_for_workspace(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            state_file = root / "permissions.json"
            first = PermissionManager(
                root,
                state_file=state_file,
                approval_callback=lambda _request: PermissionDecision.ALWAYS_ALLOW,
            )
            self.assertTrue(
                first.request("file_read", "read_text_file", "a.txt", "test")
            )

            second = PermissionManager(root, state_file=state_file)
            self.assertTrue(
                second.request("file_read", "read_text_file", "b.txt", "test")
            )

    def test_web_always_can_be_set_and_restored_without_reset(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            state_file = root / "permissions.json"
            manager = PermissionManager(root, state_file=state_file)

            manager.set_persistent_allow("web", True)
            self.assertTrue(
                PermissionManager(root, state_file=state_file).request(
                    "web", "web_search", "query", "test"
                )
            )

            manager.set_persistent_allow("web", False)
            self.assertNotIn("web", manager.persistent_allowed())

    def test_web_off_denies_without_prompt(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            prompts = []
            manager = PermissionManager(
                root,
                state_file=root / "permissions.json",
                approval_callback=lambda request: prompts.append(request),
            )
            manager.web_enabled = False

            self.assertFalse(
                manager.request("web", "web_search", "query", "test")
            )
            self.assertEqual(prompts, [])

    def test_reset_clears_session_and_persistent_permissions(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manager = PermissionManager(
                root,
                state_file=root / "permissions.json",
                approval_callback=lambda _request: PermissionDecision.ALWAYS_ALLOW,
            )
            manager.request("command", "run_command", "test", "test")
            manager.session_allowed.add("web")

            manager.reset()

            self.assertEqual(manager.session_allowed, set())
            self.assertEqual(manager.persistent_allowed(), set())

    def test_file_handler_denial_prevents_file_read(self):
        with TemporaryDirectory() as temporary_directory:
            target = Path(temporary_directory) / "secret.txt"
            target.write_text("secret", encoding="utf-8")
            handler = FileHandler(permission_callback=lambda *_args: False)

            success, message = handler.read_file(target)

            self.assertFalse(success)
            self.assertIn("Permission denied", message)
