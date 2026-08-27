import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from main.agent_runtime import LocalModelAdapter, PydanticAgentRuntime, RuntimeConfig
from main.cli import OpenCLI
from main.react_loop import (
    ReactCritique, ReactLoopController, ReactLoopLimitError, ReactLoopPolicy,
)
from main.sandbox import E2BSandbox, SandboxConfig, SandboxManager
from main.task_plan import TaskPlanStore


class FakeRemoteFiles:
    def __init__(self):
        self.data = {}

    def write(self, path, content):
        self.data[path] = bytes(content)

    def read(self, path, format="text"):
        content = self.data[path]
        return content if format == "bytes" else content.decode("utf-8")

    def list(self, root, depth=1):
        return [
            SimpleNamespace(path=path, type="file")
            for path in sorted(self.data)
            if path.startswith(root.rstrip("/") + "/")
        ]


class FakeRemoteCommands:
    def __init__(self):
        self.calls = []

    def run(self, command, **kwargs):
        self.calls.append((command, kwargs))
        return SimpleNamespace(exit_code=0, stdout="ok\n", stderr="")


class FakeRemoteSandbox:
    sandbox_id = "sandbox-user-owned"

    def __init__(self):
        self.files = FakeRemoteFiles()
        self.commands = FakeRemoteCommands()
        self.running = True

    def is_running(self):
        return self.running

    def kill(self):
        self.running = False
        return True


class E2BSandboxTests(TestCase):
    def test_explicit_push_excludes_secrets_and_quotes_argv(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "app.py").write_text("print('ok')", encoding="utf-8")
            (root / ".env").write_text("TOKEN=secret", encoding="utf-8")
            (root / ".git").mkdir()
            (root / ".git" / "config").write_text("secret", encoding="utf-8")
            remote = FakeRemoteSandbox()
            sandbox = E2BSandbox(root, sandbox=remote)

            pushed = sandbox.push_workspace()
            result = sandbox.run(["python", "a file.py"])

        self.assertEqual(pushed["uploaded"], 1)
        self.assertIn("/workspace/app.py", remote.files.data)
        self.assertNotIn("/workspace/.env", remote.files.data)
        self.assertEqual(remote.commands.calls[0][0], "python 'a file.py'")
        self.assertEqual(result["backend"], "e2b")

    def test_pull_applies_only_remote_changes_without_local_conflicts(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "app.py"
            target.write_text("old", encoding="utf-8")
            remote = FakeRemoteSandbox()
            sandbox = E2BSandbox(root, sandbox=remote)
            sandbox.push_workspace()
            remote.files.data["/workspace/app.py"] = b"remote"
            remote.files.data["/workspace/new.py"] = b"new"

            preview = sandbox.pull_workspace(apply=False)
            applied = sandbox.pull_workspace(apply=True)

            self.assertEqual(preview["changed"], ["app.py", "new.py"])
            self.assertEqual(target.read_text(encoding="utf-8"), "remote")
            self.assertEqual((root / "new.py").read_text(encoding="utf-8"), "new")
            self.assertTrue(applied["deletions_ignored"])

    def test_pull_rejects_local_conflict(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "app.py"
            target.write_text("old", encoding="utf-8")
            remote = FakeRemoteSandbox()
            sandbox = E2BSandbox(root, sandbox=remote)
            sandbox.push_workspace()
            target.write_text("local", encoding="utf-8")
            remote.files.data["/workspace/app.py"] = b"remote"

            result = sandbox.pull_workspace(apply=True)

            self.assertEqual(result["conflicts"], ["app.py"])
            self.assertEqual(target.read_text(encoding="utf-8"), "local")

    @patch("main.sandbox.DockerSandbox.is_available", return_value=True)
    def test_manager_accepts_user_selected_docker_image(self, _available):
        with TemporaryDirectory() as directory:
            manager = SandboxManager(Path(directory))
            status = manager.use_docker("node:22-slim")
        self.assertEqual(status["image"], "node:22-slim")
        self.assertEqual(manager.backend, "docker")

    @patch("main.sandbox.shutil.which", return_value="docker")
    @patch("main.sandbox.subprocess.run")
    def test_docker_write_mount_and_logical_cwd_are_explicit(self, mocked_run, _which):
        mocked_run.side_effect = [
            SimpleNamespace(returncode=0),
            SimpleNamespace(returncode=0),
            SimpleNamespace(returncode=0),
            SimpleNamespace(returncode=0, stdout="ok", stderr=""),
        ]
        with TemporaryDirectory() as directory:
            manager = SandboxManager(Path(directory))
            manager.use_docker()
            manager.run(["python", "-V"], write_access=True, cwd="src")
        invocation = mocked_run.call_args_list[-1].args[0]
        mount = invocation[invocation.index("--mount") + 1]
        self.assertNotIn("readonly", mount)
        self.assertEqual(invocation[invocation.index("--workdir") + 1], "/workspace/src")


class ReactLoopTests(TestCase):
    def test_local_runtime_recovers_dispatch_and_critique_formatting(self):
        class Engine:
            current_mode = "test"
            MODELS = {"test": {"path": "test-model"}}

            def __init__(self):
                self.prompts = []

            def generate_runtime_stream(self, prompt):
                self.prompts.append(prompt)
                responses = [
                    "I'll create it now.",
                    '<tool_call>{"name":"react_dispatch","arguments":'
                    '{"decision":"act","summary":"Write required",'
                    '"goal":"Create recovered.txt","paths":["recovered.txt"]}}'
                    "</tool_call>",
                    '<tool_call>{"name":"write_text_file","arguments":'
                    '{"path":"recovered.txt","content":"recovered"}}'
                    "</tool_call>",
                    "The write worked, so I am done.",
                    '<tool_call>{"name":"critique_and_plan","arguments":'
                    '{"progress":"File created","evidence":["write succeeded"],'
                    '"complete":true}}</tool_call>',
                    "Created recovered.txt.",
                ]
                yield {"type": "token", "content": responses[len(self.prompts) - 1]}

        with TemporaryDirectory() as directory:
            workspace = Path(directory)
            engine = Engine()
            runtime = PydanticAgentRuntime(
                engine,
                workspace=workspace,
                config=RuntimeConfig(
                    persist_state=False,
                    react_decision_retries=2,
                ),
            )

            events = list(runtime.generate_stream("Create recovered.txt"))

            self.assertEqual(
                (workspace / "recovered.txt").read_text(encoding="utf-8"),
                "recovered",
            )
        self.assertEqual(len(engine.prompts), 6)
        self.assertIn("STRUCTURED OUTPUT ERROR", engine.prompts[1])
        self.assertIn("STRUCTURED OUTPUT ERROR", engine.prompts[4])
        self.assertEqual(runtime.react.status()["phase"], "finish")
        self.assertTrue(any(
            event.get("type") == "token"
            and "Created recovered.txt." in str(event.get("content", ""))
            for event in events
        ))

    def test_local_forced_dispatch_retries_prose_then_accepts_tool_call(self):
        class Engine:
            current_mode = "test"
            MODELS = {"test": {"path": "test-model"}}

            def __init__(self):
                self.prompts = []

            def generate_runtime_stream(self, prompt):
                self.prompts.append(prompt)
                content = (
                    "I'll inspect it now."
                    if len(self.prompts) == 1
                    else '<tool_call>{"name":"react_dispatch","arguments":'
                    '{"decision":"act","summary":"Needs inspection",'
                    '"goal":"Inspect workspace"}}</tool_call>'
                )
                yield {"type": "token", "content": content}

        controller = ReactLoopController()
        controller.begin_turn("inspect workspace")
        engine = Engine()
        adapter = LocalModelAdapter(
            engine, react_controller=controller, react_decision_retries=2
        )
        info = SimpleNamespace(function_tools=[SimpleNamespace(
            name="react_dispatch", description="", parameters_json_schema={}
        )])

        async def collect():
            return [event async for event in adapter.stream([], info)]

        events = asyncio.run(collect())
        call = next(iter(events[0].values()))
        self.assertEqual(call.name, "react_dispatch")
        self.assertEqual(len(engine.prompts), 2)
        self.assertIn("STRUCTURED OUTPUT ERROR", engine.prompts[1])

    def test_local_forced_dispatch_falls_back_after_bounded_failures(self):
        class Engine:
            current_mode = "test"
            MODELS = {"test": {"path": "test-model"}}

            def __init__(self):
                self.calls = 0

            def generate_runtime_stream(self, _prompt):
                self.calls += 1
                yield {"type": "token", "content": "I will do that."}

        controller = ReactLoopController()
        controller.begin_turn("inspect workspace")
        engine = Engine()
        adapter = LocalModelAdapter(
            engine, react_controller=controller, react_decision_retries=2
        )
        info = SimpleNamespace(function_tools=[SimpleNamespace(
            name="react_dispatch", description="", parameters_json_schema={}
        )])

        async def collect():
            return [event async for event in adapter.stream([], info)]

        events = asyncio.run(collect())
        self.assertEqual(engine.calls, 2)
        self.assertIn("failed after 2 attempts", events[0])
        self.assertEqual(controller.status()["phase"], "ask_user")

    def test_every_turn_starts_at_dispatch_and_routes_without_tools(self):
        controller = ReactLoopController()
        controller.begin_turn("hello")

        self.assertEqual(controller.status()["phase"], "dispatch")
        status = controller.dispatch("answer", summary="Simple conversation")

        self.assertEqual(status["phase"], "finish")
        self.assertFalse(status["requested"])
        self.assertEqual(status["steps"], 0)

    def test_act_dispatch_remains_in_timeline_after_task_start(self):
        controller = ReactLoopController()
        controller.begin_turn("inspect")
        controller.dispatch("act", summary="Workspace evidence needed")
        status = controller.start_task("Inspect workspace")

        self.assertEqual(
            [event["phase"] for event in status["timeline"]],
            ["dispatch", "plan", "plan"],
        )

    def test_dispatch_rejects_invalid_or_repeated_transition(self):
        controller = ReactLoopController()
        controller.begin_turn("inspect")
        with self.assertRaisesRegex(ValueError, "answer, act, or ask_user"):
            controller.dispatch("maybe")
        controller.dispatch("act")
        with self.assertRaisesRegex(ValueError, "start of a turn"):
            controller.dispatch("act")

    def test_observation_requires_critique_before_next_action(self):
        controller = ReactLoopController()
        controller.begin_turn("inspect project")
        controller.start_task("inspect project")
        controller.before_tool("list_files", {"path": "."})
        controller.after_tool({"summary": "found main and tests"})

        self.assertEqual(controller.status()["phase"], "critique")
        with self.assertRaisesRegex(ReactLoopLimitError, "phase: critique"):
            controller.before_tool("read_text_file", {"path": "main/app.py"})

    def test_critique_controls_finish_ask_user_and_continue(self):
        controller = ReactLoopController()
        controller.begin_turn("inspect project")
        controller.start_task("inspect project")
        controller.before_tool("list_files", {"path": "."})
        controller.after_tool({"summary": "found main"})
        status = controller.submit_critique(ReactCritique(
            progress="Layout found",
            evidence=("main exists",),
            next_action="Read entrypoint",
        ))
        self.assertEqual(status["phase"], "act")

        controller.before_tool("read_text_file", {"path": "main/app.py"})
        controller.after_tool({"summary": "entrypoint read"})
        status = controller.submit_critique({
            "progress": "Review complete", "complete": True,
        })
        self.assertEqual(status["phase"], "finish")

    def test_loop_context_tracks_timeline_evidence_and_remaining_budget(self):
        controller = ReactLoopController(ReactLoopPolicy(max_steps=3))
        controller.begin_turn("inspect")
        controller.start_task("inspect")
        controller.before_tool("list_files", {"path": "."})
        controller.after_tool({"summary": "found two files"})
        context = controller.loop_context()

        self.assertEqual(context["phase"], "critique")
        self.assertEqual(context["remaining_steps"], 2)
        self.assertEqual(context["recent_evidence"], ["found two files"])
        self.assertEqual(
            [event["phase"] for event in controller.status()["timeline"]][-3:],
            ["act", "observe", "critique"],
        )

    def test_repeated_identical_action_stops_before_third_execution(self):
        controller = ReactLoopController(
            ReactLoopPolicy(max_steps=10, max_repeated_action=2)
        )
        controller.begin_turn("inspect project")
        controller.start_task("inspect project")
        controller.before_tool("read_text_file", {"path": "a.py"})
        controller.before_tool("read_text_file", {"path": "a.py"})
        with self.assertRaisesRegex(ReactLoopLimitError, "repeated action"):
            controller.before_tool("read_text_file", {"path": "a.py"})

    def test_consecutive_failures_block_next_action(self):
        controller = ReactLoopController(ReactLoopPolicy(max_consecutive_failures=2))
        controller.begin_turn("run tests")
        controller.start_task("run tests")
        controller.before_tool("run", {"command": ["pytest"]})
        controller.after_tool({"summary": "error: missing pytest"})
        controller.submit_critique({
            "progress": "pytest command unavailable",
            "next_action": "Try Python module invocation",
        })
        controller.before_tool("run", {"command": ["python", "-m", "pytest"]})
        controller.after_tool({"summary": "failed: missing pytest"})
        with self.assertRaisesRegex(ReactLoopLimitError, "consecutive tool failures"):
            controller.before_tool("run", {"command": ["pytest", "-q"]})

    def test_normal_tool_use_does_not_start_react_until_model_requests_it(self):
        controller = ReactLoopController(ReactLoopPolicy(max_steps=3))
        controller.begin_turn("what is in this project?")

        self.assertEqual(controller.before_tool("list_files", {"path": "."}), 0)
        status = controller.start_task("Review source layout", paths=("main",), max_steps=99)

        self.assertTrue(status["requested"])
        self.assertEqual(status["paths"], ["main"])
        self.assertEqual(status["max_steps"], 3)
        self.assertEqual(controller.before_tool("list_files", {"path": "main"}), 1)
        with self.assertRaisesRegex(ValueError, "already active"):
            controller.start_task("Reset budget")

    def test_plan_completion_requires_tool_evidence(self):
        class Engine:
            current_mode = "test"
            MODELS = {"test": {"path": "test-model"}}

        with TemporaryDirectory() as directory:
            root = Path(directory)
            store = TaskPlanStore(root, "session", root / "plans")
            item = store.add_item("Verify tests")
            runtime = PydanticAgentRuntime(
                Engine(), workspace=root,
                config=RuntimeConfig(persist_state=False),
                task_plan_store=store,
            )

            denied = runtime.update_task_plan_item(item.id, "completed")
            runtime._record_event(
                {"type": "tool_result", "name": "read_text_file", "summary": "read app.py"}
            )
            completed = runtime.update_task_plan_item(item.id, "completed")

        self.assertFalse(denied["updated"])
        self.assertTrue(completed["updated"])
        self.assertIn("create_task_plan", runtime.available_tools)

    def test_agent_e2b_command_requires_command_and_write_approval(self):
        class Engine:
            current_mode = "test"
            MODELS = {"test": {"path": "test-model"}}

        class Sandbox:
            backend = "e2b"

            def is_available(self):
                return True

            def status(self):
                return {"backend": "e2b", "available": True}

            def run(self, command, **kwargs):
                return {"exit_code": 0, "output": "ok", "backend": "e2b"}

        approvals = []
        runtime = PydanticAgentRuntime(
            Engine(),
            config=RuntimeConfig(persist_state=False),
            sandbox=Sandbox(),
            permission_callback=lambda category, *_args: approvals.append(category) or True,
        )
        result = runtime.run_sandboxed_command(["python", "-V"])
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(approvals, ["command", "file_write"])

    def test_planning_only_prompt_is_not_forced_into_mutation_mode(self):
        class Engine:
            current_mode = "test"
            MODELS = {"test": {"path": "test-model"}}

        runtime = PydanticAgentRuntime(
            Engine(), config=RuntimeConfig(persist_state=False)
        )
        self.assertFalse(runtime._is_workspace_mutation_request("Plan improvements for this repo"))
        self.assertTrue(runtime._is_workspace_mutation_request("Plan and implement improvements"))

    def test_language_wrapper_cannot_turn_status_request_into_file_mutation(self):
        class Engine:
            current_mode = "test"
            MODELS = {"test": {"path": "test-model"}}

        runtime = PydanticAgentRuntime(
            Engine(), config=RuntimeConfig(persist_state=False)
        )
        prompt = (
            "RESPONSE LANGUAGE: English. Always write final assistant prose in English."
            "\n\nUSER REQUEST:\nEverything ok?"
        )

        self.assertFalse(runtime._is_workspace_mutation_request(prompt))
        self.assertNotIn('"name": "write_text_file"', runtime._tool_prompt_text)
        self.assertIn('"name": "write_text_file"', runtime._mutation_tool_prompt_text)

    def test_runtime_control_tools_are_visible_and_manual_start_remains_capped(self):
        class Engine:
            current_mode = "test"
            MODELS = {"test": {"path": "test-model"}}

        with TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = PydanticAgentRuntime(
                Engine(), workspace=root,
                config=RuntimeConfig(persist_state=False, react_max_steps=4),
            )
            result = runtime.start_react_task(
                "Inspect module layout", paths=["main"], max_steps=99
            )

        self.assertTrue(result["started"])
        self.assertEqual(result["max_steps"], 4)
        self.assertEqual(result["paths"], ["main"])
        self.assertIn("react_dispatch", runtime.available_tools)
        self.assertIn("critique_and_plan", runtime.available_tools)
        self.assertNotIn("start_react_task", runtime.available_tools)
        self.assertIn('"name": "react_dispatch"', runtime._tool_prompt_text)

    def test_remote_adapter_keeps_one_tool_action_per_model_step(self):
        class Client:
            def stream_chat(self, _messages, _tools):
                yield {
                    "type": "tool_calls",
                    "calls": [
                        {"id": "1", "name": "first", "arguments": "{}"},
                        {"id": "2", "name": "second", "arguments": "{}"},
                    ],
                }

        class Engine:
            backend = "remote_api"
            api_client = Client()
            current_mode = "api"
            MODELS = {"api": {"path": "provider/model"}}

        info = SimpleNamespace(
            function_tools=[
                SimpleNamespace(name="first", description="", parameters_json_schema={}),
                SimpleNamespace(name="second", description="", parameters_json_schema={}),
            ]
        )
        adapter = LocalModelAdapter(Engine(), single_tool_per_step=True)

        async def collect():
            return [event async for event in adapter._stream_remote([], info)]

        events = asyncio.run(collect())
        calls = next(event for event in events if isinstance(event, dict))
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "first")


class SandboxCliTests(TestCase):
    @patch("main.sandbox.DockerSandbox.is_available", return_value=True)
    def test_cli_selects_docker_and_toggles_react(self, _available):
        cli = OpenCLI(dry_run=True)
        with patch("builtins.print"):
            cli.handle_command("/sandbox docker python:3.12-slim")
            cli.handle_command("/react off")
        self.assertTrue(cli.sandbox_enabled)
        self.assertEqual(cli.sandbox.backend, "docker")
        self.assertFalse(cli.react_enabled)
        self.assertEqual(cli.react_status()["mode"], "ordinary_agent")
        self.assertEqual(cli.react_status()["phase"], "off")

    def test_cli_react_on_reports_strict_every_turn_mode(self):
        cli = OpenCLI(dry_run=True)
        cli.react_enabled = False
        with patch("builtins.print"):
            cli.handle_command("/react on")

        status = cli.react_status()
        self.assertTrue(status["enabled"])
        self.assertEqual(status["mode"], "strict_every_turn")
        self.assertEqual(status["phase"], "ready")
