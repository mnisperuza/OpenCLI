import asyncio
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from fenrir_agent.agent_runtime import LocalModelAdapter, PydanticAgentRuntime, RuntimeConfig
from fenrir_agent.cli import FenrirAgent
from fenrir_agent.react_loop import (
    ReactCritique, ReactLoopController, ReactLoopLimitError, ReactLoopPolicy,
)
from fenrir_agent.sandbox import (
    CodexSandbox, DockerSandbox, E2BSandbox, SandboxManager,
)
from fenrir_agent.task_plan import TaskPlanStore


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
    def test_docker_snapshot_excludes_secret_and_control_files(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "workspace"
            root.mkdir()
            (root / "app.py").write_text("print('ok')", encoding="utf-8")
            (root / ".env").write_text("TOKEN=secret", encoding="utf-8")
            (root / ".git").mkdir()
            (root / ".git" / "config").write_text("private", encoding="utf-8")
            snapshot = Path(directory) / "snapshot"
            snapshot.mkdir()

            DockerSandbox(root)._snapshot_workspace(snapshot)

            self.assertTrue((snapshot / "app.py").is_file())
            self.assertFalse((snapshot / ".env").exists())
            self.assertFalse((snapshot / ".git").exists())
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

    @patch("fenrir_agent.sandbox.DockerSandbox.is_available", return_value=True)
    def test_manager_accepts_user_selected_docker_image(self, _available):
        with TemporaryDirectory() as directory:
            manager = SandboxManager(Path(directory))
            status = manager.use_docker("node:22-slim")
        self.assertEqual(status["image"], "node:22-slim")
        self.assertEqual(manager.backend, "docker")

    @patch("fenrir_agent.sandbox.shutil.which", return_value="docker")
    @patch("fenrir_agent.sandbox.subprocess.run")
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


class CodexSandboxTests(TestCase):
    def test_command_environment_excludes_credentials(self):
        with patch.dict(
            os.environ,
            {"PATH": "tools", "OPENAI_API_KEY": "secret", "CUSTOM_TOKEN": "secret"},
            clear=True,
        ):
            environment = CodexSandbox._safe_environment()

        self.assertEqual(environment["PATH"], "tools")
        self.assertNotIn("OPENAI_API_KEY", environment)
        self.assertNotIn("CUSTOM_TOKEN", environment)

    @patch("fenrir_agent.sandbox.CodexSandbox.is_available", return_value=True)
    @patch("fenrir_agent.sandbox.subprocess.run")
    def test_uses_builtin_profiles_and_never_a_shell(self, mocked_run, _available):
        mocked_run.return_value = SimpleNamespace(
            returncode=0, stdout="ok\n", stderr=""
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "codex.exe"
            executable.touch()
            sandbox = CodexSandbox(root, executable=executable)

            read_result = sandbox.run(["python", "-V"])
            read_call = mocked_run.call_args
            write_result = sandbox.run(
                ["python", "-m", "pytest"], write_access=True
            )
            write_call = mocked_run.call_args

        self.assertIn("fenrir-read-only", read_call.args[0])
        self.assertIn("fenrir-workspace", write_call.args[0])
        self.assertIn(
            "permissions.fenrir-read-only.network.enabled=false",
            read_call.args[0],
        )
        self.assertIn(
            "permissions.fenrir-workspace.network.enabled=false",
            write_call.args[0],
        )
        self.assertFalse(read_call.kwargs["shell"])
        self.assertFalse(write_call.kwargs["shell"])
        self.assertEqual(read_result["network"], "disabled")
        self.assertFalse(read_result["changes_persisted"])
        self.assertTrue(write_result["changes_persisted"])

    @patch("fenrir_agent.sandbox.CodexSandbox.is_available", return_value=True)
    def test_rejects_cwd_escape_before_execution(self, _available):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "codex.exe"
            executable.touch()
            sandbox = CodexSandbox(root, executable=executable)
            with self.assertRaises(ValueError):
                sandbox.run(["python", "-V"], cwd="../outside")

    @patch("fenrir_agent.sandbox.CodexSandbox.is_available", return_value=True)
    @patch("fenrir_agent.sandbox.subprocess.run")
    def test_windows_setup_falls_back_to_unelevated(self, mocked_run, _available):
        mocked_run.side_effect = [
            SimpleNamespace(returncode=1, stdout="", stderr="elevation denied"),
            SimpleNamespace(returncode=0, stdout="", stderr=""),
        ]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "codex.exe"
            executable.touch()
            sandbox = CodexSandbox(root, executable=executable)
            with patch("fenrir_agent.sandbox.os.name", "nt"):
                status = sandbox.prepare()

        self.assertTrue(status["available"])
        self.assertEqual(status["windows_mode"], "unelevated")
        self.assertEqual(mocked_run.call_count, 2)

    @patch("fenrir_agent.sandbox.CodexSandbox.prepare")
    def test_manager_selects_codex_as_default(self, prepare):
        prepare.return_value = {
            "backend": "codex", "available": True, "network": "disabled"
        }
        with TemporaryDirectory() as directory:
            manager = SandboxManager(Path(directory))
            result = manager.use_default()

        self.assertEqual(result["backend"], "codex")
        self.assertEqual(manager.backend, "codex")


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
                    react_strict_control=True,
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
        controller.after_tool({
            "summary": "pytest unavailable",
            "outcome": {
                "schema_version": 1,
                "status": "retryable_error",
                "summary": "pytest unavailable",
                "error_code": "execution_failed",
            },
        })
        controller.submit_critique({
            "progress": "pytest command unavailable",
            "next_action": "Try Python module invocation",
        })
        controller.before_tool("run", {"command": ["python", "-m", "pytest"]})
        controller.after_tool({
            "summary": "python module unavailable",
            "outcome": {
                "schema_version": 1,
                "status": "retryable_error",
                "summary": "python module unavailable",
                "error_code": "execution_failed",
            },
        })
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

    def test_normal_react_warns_before_repeated_action_hard_stop(self):
        controller = ReactLoopController(ReactLoopPolicy(
            strict_control=False,
            max_steps=10,
            max_repeated_action=5,
            warn_repeated_action=2,
            hard_stagnation_limit=10,
        ))
        controller.begin_turn("inspect")
        for _index in range(2):
            controller.before_tool("read_text_file", {"path": "same.py"})
            controller.after_tool({"summary": "same content"})

        status = controller.status()
        self.assertEqual(status["phase"], "act")
        self.assertIn("warning", status["guardrail_warning"].casefold())

        for _index in range(3):
            controller.before_tool("read_text_file", {"path": "same.py"})
            controller.after_tool({"summary": "same content"})
        with self.assertRaisesRegex(ReactLoopLimitError, "repeated action"):
            controller.before_tool("read_text_file", {"path": "same.py"})

    def test_normal_react_closes_tools_at_budget_for_graceful_answer(self):
        controller = ReactLoopController(ReactLoopPolicy(
            strict_control=False, max_steps=1, max_repeated_action=5
        ))
        controller.begin_turn("inspect")
        controller.before_tool("list_files", {"path": "."})
        controller.after_tool({"summary": "found files"})

        self.assertTrue(controller.status()["wrap_up_required"])
        self.assertEqual(controller.status()["phase"], "act")

    def test_interactive_guardrails_warn_without_blocking_repeated_actions(self):
        controller = ReactLoopController(ReactLoopPolicy(
            strict_control=False,
            hard_stops=False,
            max_steps=10,
            max_repeated_action=2,
            warn_repeated_action=2,
            hard_stagnation_limit=2,
        ))
        controller.begin_turn("explore")
        for _index in range(4):
            controller.before_tool("read_text_file", {"path": "same.py"})
            controller.after_tool({"summary": "same content"})

        self.assertEqual(controller.status()["steps"], 4)
        self.assertFalse(controller.status()["halted_reason"])
        self.assertIn("warning", controller.status()["guardrail_warning"].casefold())

    def test_react_off_keeps_quiet_tool_budget_guard(self):
        controller = ReactLoopController(ReactLoopPolicy(
            strict_control=False, hard_stops=False, max_steps=1
        ))
        controller.enabled = False
        controller.begin_turn("ordinary tool use")

        self.assertEqual(controller.before_tool("list_files", {"path": "."}), 1)
        controller.after_tool({"summary": "found files"})
        with self.assertRaisesRegex(ReactLoopLimitError, "tool steps"):
            controller.before_tool("read_text_file", {"path": "a.py"})

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

    def test_agent_codex_command_relies_on_native_boundary_without_prompt(self):
        class Engine:
            current_mode = "test"
            MODELS = {"test": {"path": "test-model"}}

        class Sandbox:
            backend = "codex"

            def is_available(self):
                return True

            def status(self):
                return {"backend": "codex", "available": True}

            def run(self, command, **kwargs):
                return {
                    "exit_code": 0,
                    "output": "ok",
                    "backend": "codex",
                    "changes_persisted": kwargs.get("write_access", False),
                }

        approvals = []
        runtime = PydanticAgentRuntime(
            Engine(),
            config=RuntimeConfig(persist_state=False),
            sandbox=Sandbox(),
            permission_callback=lambda category, *_args: (
                approvals.append(category) or False
            ),
        )
        result = runtime.run_sandboxed_command(
            ["python", "-V"], write_access=True
        )

        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(approvals, [])

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

    def test_task_plan_context_cannot_turn_chat_into_file_mutation(self):
        class Engine:
            current_mode = "test"
            MODELS = {"test": {"path": "test-model"}}

        runtime = PydanticAgentRuntime(
            Engine(), config=RuntimeConfig(persist_state=False)
        )
        prompt = (
            "USER REQUEST:\nAre you okay?"
            "\n\nUSER-MAINTAINED TASK PLAN:\n"
            "- Create a Python script and run tests"
        )

        self.assertFalse(runtime._is_workspace_mutation_request(prompt))

    def test_runtime_control_tools_are_visible_and_manual_start_remains_capped(self):
        class Engine:
            current_mode = "test"
            MODELS = {"test": {"path": "test-model"}}

        with TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = PydanticAgentRuntime(
                Engine(), workspace=root,
                config=RuntimeConfig(
                    persist_state=False, react_max_steps=4,
                    react_strict_control=True,
                ),
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

    def test_default_runtime_uses_normal_host_managed_react(self):
        class Engine:
            current_mode = "test"
            MODELS = {"test": {"path": "test-model"}}

            def __init__(self):
                self.prompts = []

            def generate_runtime_stream(self, prompt):
                self.prompts.append(prompt)
                if len(self.prompts) == 1:
                    content = (
                        '<tool_call>{"name":"write_text_file","arguments":'
                        '{"path":"normal.txt","content":"done"}}</tool_call>'
                    )
                else:
                    content = "Created normal.txt."
                yield {"type": "token", "content": content}

        with TemporaryDirectory() as directory:
            root = Path(directory)
            engine = Engine()
            runtime = PydanticAgentRuntime(
                engine, workspace=root, config=RuntimeConfig(persist_state=False)
            )
            events = list(runtime.generate_stream("Create normal.txt"))

            self.assertEqual((root / "normal.txt").read_text(encoding="utf-8"), "done")
        self.assertEqual(len(engine.prompts), 2)
        self.assertNotIn("react_dispatch", runtime.available_tools)
        self.assertEqual(runtime.react.status()["phase"], "finish")
        self.assertTrue(any(event.get("type") == "tool" for event in events))

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

    def test_remote_adapter_allows_three_tool_actions_by_default(self):
        class Client:
            def stream_chat(self, _messages, _tools):
                yield {
                    "type": "tool_calls",
                    "calls": [
                        {
                            "id": str(index), "name": "read",
                            "arguments": json.dumps({"path": str(index)}),
                        }
                        for index in range(4)
                    ],
                }

        class Engine:
            backend = "remote_api"
            api_client = Client()
            current_mode = "api"
            MODELS = {"api": {"path": "provider/model"}}

        info = SimpleNamespace(function_tools=[
            SimpleNamespace(name="read", description="", parameters_json_schema={})
        ])
        adapter = LocalModelAdapter(Engine())

        async def collect():
            return [event async for event in adapter._stream_remote([], info)]

        events = asyncio.run(collect())
        calls = next(event for event in events if isinstance(event, dict))
        self.assertEqual(len(calls), 3)

    def test_remote_adapter_does_not_finalize_text_before_later_tool_calls(self):
        captured = []

        class Client:
            def stream_chat(self, _messages, _tools, _tool_choice="auto"):
                yield {"type": "token", "content": "I will inspect it first."}
                yield {
                    "type": "tool_calls",
                    "calls": [{
                        "id": "read-1",
                        "name": "read_text_file",
                        "arguments": '{"path":"IMPROVEMENTS.md"}',
                    }],
                }

        class Engine:
            backend = "remote_api"
            api_client = Client()
            current_mode = "api"
            MODELS = {"api": {"path": "provider/model"}}

        info = SimpleNamespace(function_tools=[SimpleNamespace(
            name="read_text_file", description="", parameters_json_schema={}
        )])
        adapter = LocalModelAdapter(Engine(), event_sink=captured.append)

        async def collect():
            return [event async for event in adapter.stream([], info)]

        events = asyncio.run(collect())
        self.assertFalse(any(isinstance(event, str) for event in events))
        calls = next(event for event in events if isinstance(event, dict))
        self.assertEqual(calls[0].tool_call_id, "read-1")
        turn = next(event for event in captured if event["type"] == "model_turn")
        self.assertEqual(turn["disposition"], "tool_calls")
        self.assertEqual(turn["content"], "I will inspect it first.")

    def test_adapter_reserves_last_model_request_for_final_answer(self):
        class Client:
            def __init__(self):
                self.choices = []

            def stream_chat(self, _messages, _tools, tool_choice="auto"):
                self.choices.append(tool_choice)
                yield {"type": "token", "content": "done"}

        class Engine:
            backend = "remote_api"
            api_client = Client()
            current_mode = "api"
            MODELS = {"api": {"path": "provider/model"}}

        info = SimpleNamespace(function_tools=[SimpleNamespace(
            name="read_text_file", description="", parameters_json_schema={}
        )])
        adapter = LocalModelAdapter(
            Engine(), max_model_requests=2, final_response_request_reserve=1
        )

        async def collect():
            for _index in range(2):
                _ = [event async for event in adapter.stream([], info)]

        asyncio.run(collect())
        self.assertEqual(adapter.engine.api_client.choices, ["auto", "none"])

    def test_local_adapter_buffers_long_prose_until_tool_decision(self):
        captured = []

        class Engine:
            current_mode = "test"
            MODELS = {"test": {"path": "test-model"}}

            def generate_runtime_stream(self, _prompt):
                yield {"type": "token", "content": "thinking " * 100}
                yield {
                    "type": "token",
                    "content": (
                        '<tool_call>{"name":"read_text_file","arguments":'
                        '{"path":"IMPROVEMENTS.md"}}</tool_call>'
                    ),
                }

        info = SimpleNamespace(function_tools=[SimpleNamespace(
            name="read_text_file", description="", parameters_json_schema={}
        )])
        adapter = LocalModelAdapter(Engine(), event_sink=captured.append)

        async def collect():
            return [event async for event in adapter.stream([], info)]

        events = asyncio.run(collect())
        self.assertEqual(len(events), 1)
        self.assertIsInstance(events[0], dict)
        self.assertEqual(events[0][0].name, "read_text_file")
        self.assertFalse(any(isinstance(event, str) for event in events))
        self.assertEqual(captured[0]["type"], "model_turn")
        self.assertEqual(captured[0]["disposition"], "tool_calls")

    def test_local_adapter_returns_structured_error_for_malformed_tool_call(self):
        captured = []

        class Engine:
            current_mode = "test"
            MODELS = {"test": {"path": "test-model"}}

            def generate_runtime_stream(self, _prompt):
                yield {
                    "type": "token",
                    "content": '<tool_call>{"name":"read_text_file","arguments":',
                }

        info = SimpleNamespace(function_tools=[SimpleNamespace(
            name="read_text_file", description="", parameters_json_schema={}
        )])
        adapter = LocalModelAdapter(Engine(), event_sink=captured.append)

        async def collect():
            return [event async for event in adapter.stream([], info)]

        events = asyncio.run(collect())
        self.assertEqual(events, ["Tool call rejected: invalid JSON or unsupported tool."])
        self.assertEqual(captured[0]["disposition"], "invalid")

    def test_local_adapter_repairs_malformed_tool_call_before_finalizing(self):
        captured = []

        class Engine:
            current_mode = "test"
            MODELS = {"test": {"path": "test-model"}}

            def __init__(self):
                self.prompts = []

            def generate_runtime_stream(self, prompt):
                self.prompts.append(prompt)
                content = (
                    '<tool_call>{"name":"read_text_file","arguments":'
                    if len(self.prompts) == 1
                    else '<tool_call>{"name":"read_text_file",'
                    '"arguments":{"path":"index.html"}}</tool_call>'
                )
                yield {"type": "token", "content": content}

        engine = Engine()
        info = SimpleNamespace(function_tools=[SimpleNamespace(
            name="read_text_file", description="Read a workspace file.",
            parameters_json_schema={"type": "object"},
        )])
        adapter = LocalModelAdapter(engine, event_sink=captured.append)

        async def collect():
            return [event async for event in adapter.stream([], info)]

        events = asyncio.run(collect())
        self.assertEqual(len(events), 1)
        self.assertIsInstance(events[0], dict)
        self.assertEqual(events[0][0].name, "read_text_file")
        self.assertEqual(len(engine.prompts), 2)
        self.assertIn("TOOL CALL REJECTED", engine.prompts[1])
        self.assertIn('<tool_call>{"name":"tool_name"', engine.prompts[1])
        self.assertEqual(captured[0]["disposition"], "tool_calls")

    def test_local_adapter_preserves_final_request_budget_over_repair(self):
        class Engine:
            current_mode = "test"
            MODELS = {"test": {"path": "test-model"}}

            def __init__(self):
                self.prompts = []

            def generate_runtime_stream(self, prompt):
                self.prompts.append(prompt)
                yield {
                    "type": "token",
                    "content": '<tool_call>{"name":"read_text_file","arguments":',
                }

        engine = Engine()
        info = SimpleNamespace(function_tools=[SimpleNamespace(
            name="read_text_file", description="Read a workspace file.",
            parameters_json_schema={"type": "object"},
        )])
        adapter = LocalModelAdapter(
            engine, max_model_requests=2, final_response_request_reserve=1,
        )

        async def collect():
            return [event async for event in adapter.stream([], info)]

        events = asyncio.run(collect())
        self.assertEqual(events, ["Tool call rejected: invalid JSON or unsupported tool."])
        self.assertEqual(len(engine.prompts), 1)

    def test_remote_adapter_repairs_invalid_tool_event_before_finalizing(self):
        class Client:
            def __init__(self):
                self.requests = []

            def stream_chat(self, messages, _tools, _tool_choice="auto"):
                self.requests.append(messages)
                if len(self.requests) == 1:
                    yield {
                        "type": "tool_calls",
                        "calls": [{"id": "bad", "name": "not_a_tool", "arguments": "{}"}],
                    }
                else:
                    yield {
                        "type": "tool_calls",
                        "calls": [{
                            "id": "good", "name": "read_text_file",
                            "arguments": '{"path":"index.html"}',
                        }],
                    }

        class Engine:
            backend = "remote_api"
            current_mode = "api"
            MODELS = {"api": {"path": "provider/model"}}

            def __init__(self):
                self.api_client = Client()

        engine = Engine()
        info = SimpleNamespace(function_tools=[SimpleNamespace(
            name="read_text_file", description="Read a workspace file.",
            parameters_json_schema={"type": "object"},
        )])
        adapter = LocalModelAdapter(engine)

        async def collect():
            return [event async for event in adapter.stream([], info)]

        events = asyncio.run(collect())
        self.assertEqual(events[0][0].name, "read_text_file")
        self.assertEqual(len(engine.api_client.requests), 2)
        repair = engine.api_client.requests[1][-1]["content"]
        self.assertIn("TOOL CALL REJECTED", repair)
        self.assertIn("read_text_file", str(engine.api_client.requests[1]))


class SandboxCliTests(TestCase):
    @patch("fenrir_agent.sandbox.CodexSandbox.prepare")
    def test_cli_sandbox_is_off_until_default_is_enabled(self, prepare):
        prepare.return_value = {
            "backend": "codex", "available": True, "network": "disabled"
        }
        cli = FenrirAgent(dry_run=True)
        self.assertFalse(cli.sandbox_enabled)
        self.assertEqual(cli.sandbox.backend, "none")

        with patch("builtins.print"):
            cli.handle_command("/sandbox on")

        self.assertTrue(cli.sandbox_enabled)
        self.assertEqual(cli.sandbox.backend, "codex")

    @patch("fenrir_agent.sandbox.DockerSandbox.is_available", return_value=True)
    def test_cli_selects_docker_and_toggles_react(self, _available):
        cli = FenrirAgent(dry_run=True)
        with patch("builtins.print"):
            cli.handle_command("/sandbox docker python:3.12-slim")
            cli.handle_command("/react off")
        self.assertTrue(cli.sandbox_enabled)
        self.assertEqual(cli.sandbox.backend, "docker")
        self.assertFalse(cli.react_enabled)
        self.assertEqual(cli.react_status()["mode"], "ordinary_agent")
        self.assertEqual(cli.react_status()["phase"], "off")

    def test_cli_react_on_reports_host_managed_mode(self):
        cli = FenrirAgent(dry_run=True)
        cli.react_enabled = False
        with patch("builtins.print"):
            cli.handle_command("/react on")

        status = cli.react_status()
        self.assertTrue(status["enabled"])
        self.assertEqual(status["mode"], "host_managed")
        self.assertFalse(status["single_action_per_model_step"])
        self.assertEqual(status["phase"], "ready")

    def test_cli_react_trace_is_hidden_by_default_and_can_be_enabled(self):
        cli = FenrirAgent(dry_run=True)
        self.assertFalse(cli.react_trace_enabled)
        with patch("builtins.print"):
            cli.handle_command("/react-trace on")
        self.assertTrue(cli.react_trace_enabled)

    def test_cli_can_select_staged_harness_mode(self):
        cli = FenrirAgent(dry_run=True)
        cli.agent_runtime = object()
        with patch("builtins.print"):
            self.assertTrue(cli.handle_command("/harness mode legacy"))
        self.assertEqual(cli.harness_mode, "legacy")
        self.assertIsNone(cli.agent_runtime)
