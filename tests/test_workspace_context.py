from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from main.agent_runtime import LocalWorkspaceTools, RuntimeConfig
from main.engine import FileHandler
from main.workspace_context import WorkspaceContext


class WorkspaceContextTests(TestCase):
    def test_relative_and_absolute_paths_stay_in_workspace(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "src" / "auth"
            nested.mkdir(parents=True)
            context = WorkspaceContext(root)

            self.assertEqual(context.set_current_directory("src/auth"), nested.resolve())
            self.assertEqual(context.resolve("handler.py"), nested / "handler.py")
            self.assertEqual(context.resolve(str(nested)), nested.resolve())
            with self.assertRaisesRegex(ValueError, "trusted workspace"):
                context.resolve("../../../outside.txt")

    def test_workspace_tools_use_logical_directory_and_return_root_relative_paths(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "src"
            nested.mkdir()
            (nested / "app.py").write_text("print('ok')", encoding="utf-8")
            context = WorkspaceContext(root)
            tools = LocalWorkspaceTools(root, RuntimeConfig(), workspace_context=context)

            changed = tools.set_working_directory("src")
            loaded = tools.read_text_file("app.py")

            self.assertEqual(changed["current_directory"], "src")
            self.assertEqual(loaded["path"], "src/app.py")
            self.assertEqual(tools.get_working_directory()["current_directory"], "src")
            self.assertEqual(tools.list_allowed_roots()["allowed_roots"], [str(root.resolve())])

    def test_prompt_attachment_paths_cannot_escape_configured_workspace(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "workspace"
            root.mkdir(parents=True)
            nested = root / "src"
            nested.mkdir()
            inside = nested / "app.py"
            inside.write_text("pass", encoding="utf-8")
            outside = root.parent / "outside.py"
            outside.write_text("secret", encoding="utf-8")
            handler = FileHandler()
            handler.current_path = nested
            handler.workspace_root = root

            self.assertEqual(handler.resolve_path("app.py"), inside.resolve())
            self.assertIsNone(handler.resolve_path("../../outside.py"))
