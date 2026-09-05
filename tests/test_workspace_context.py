from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from fenrir_agent.agent_runtime import LocalWorkspaceTools, PydanticAgentRuntime, RuntimeConfig
from fenrir_agent.engine import FileHandler
from fenrir_agent.workspace_context import WorkspaceContext


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

    def test_tool_returned_root_relative_path_round_trips_from_nested_directory(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "Documents" / "tests"
            nested.mkdir(parents=True)
            target = nested / "IMPROVEMENTS.md"
            target.write_text("roadmap", encoding="utf-8")
            context = WorkspaceContext(root)
            tools = LocalWorkspaceTools(root, RuntimeConfig(), workspace_context=context)
            tools.set_working_directory("Documents/tests")

            listed = tools.list_files(".")
            returned_path = listed["files"][0]
            loaded = tools.read_text_file(returned_path)

            self.assertEqual(returned_path, "Documents/tests/IMPROVEMENTS.md")
            self.assertEqual(loaded["content"], "roadmap")
            self.assertEqual(loaded["path"], returned_path)

    def test_root_relative_write_path_does_not_duplicate_current_directory(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "src"
            nested.mkdir()
            context = WorkspaceContext(root)
            tools = LocalWorkspaceTools(root, RuntimeConfig(), workspace_context=context)
            tools.set_working_directory("src")

            result = tools.write_text_file("src/new.py", "pass\n")

            self.assertEqual(result["path"], "src/new.py")
            self.assertTrue((nested / "new.py").is_file())
            self.assertFalse((nested / "src" / "new.py").exists())

    def test_missing_read_returns_recoverable_observation(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "IMPROVEMENTS.md").write_text("roadmap", encoding="utf-8")
            events = []
            tools = LocalWorkspaceTools(root, RuntimeConfig(), event_sink=events.append)

            result = tools.read_text_file("IMPROVEMENT.md")

            self.assertTrue(result["not_found"])
            self.assertIn("IMPROVEMENTS.md", result["suggestions"])
            outcome = next(event["outcome"] for event in events if event["type"] == "tool_result")
            self.assertEqual(outcome["error_code"], "not_found")

    def test_agent_recovers_from_bad_read_path_in_same_react_turn(self):
        class Engine:
            current_mode = "test"
            MODELS = {"test": {"path": "test-model"}}

            def __init__(self):
                self.calls = 0

            def generate_runtime_stream(self, _prompt):
                self.calls += 1
                if self.calls == 1:
                    content = (
                        '<tool_call>{"name":"read_text_file","arguments":'
                        '{"path":"Documents/tests/MISSING.md"}}</tool_call>'
                    )
                elif self.calls == 2:
                    content = (
                        '<tool_call>{"name":"read_text_file","arguments":'
                        '{"path":"Documents/tests/IMPROVEMENTS.md"}}</tool_call>'
                    )
                else:
                    content = "Recovered and read roadmap."
                yield {"type": "token", "content": content}

        with TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "Documents" / "tests"
            nested.mkdir(parents=True)
            (nested / "IMPROVEMENTS.md").write_text("roadmap", encoding="utf-8")
            context = WorkspaceContext(root)
            context.set_current_directory("Documents/tests")
            runtime = PydanticAgentRuntime(
                Engine(),
                workspace=root,
                workspace_context=context,
                config=RuntimeConfig(persist_state=False),
            )

            events = list(runtime.generate_stream("Read improvements"))

        self.assertEqual(runtime.react.status()["steps"], 2)
        self.assertEqual(runtime.react.status()["phase"], "finish")
        self.assertTrue(any(
            event.get("type") == "tool_result"
            and event.get("outcome", {}).get("error_code") == "not_found"
            for event in events
        ))
        self.assertIn("Recovered", events[-1]["content"])

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
