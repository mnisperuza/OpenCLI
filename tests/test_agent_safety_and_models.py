from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from main.agent_runtime import LocalWorkspaceTools, PydanticAgentRuntime, RuntimeConfig
from main.api_profiles import ApiProfileRegistry
from main.cli import OpenCLI
from main.model_registry import ModelRegistry, ModelRegistryError
from main.permissions import PermissionManager
from main.sandbox import DockerSandbox


class WorkspaceToolSafetyTests(TestCase):
    def test_write_requires_file_write_approval_and_uses_expected_hash(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            target = root / "note.txt"
            target.write_text("old", encoding="utf-8")
            calls = []
            tools = LocalWorkspaceTools(
                root,
                RuntimeConfig(),
                permission_callback=lambda *request: calls.append(request) or True,
            )

            info = tools.file_info("note.txt")
            result = tools.write_text_file("note.txt", "new", info["sha256"])

            self.assertEqual(target.read_text(encoding="utf-8"), "new")
            self.assertEqual(result["chars"], 3)
            self.assertEqual(calls[-1][0], "file_write")

    def test_agent_cannot_read_or_write_protected_path(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            protected = root / ".env"
            protected.write_text("SECRET=value", encoding="utf-8")
            tools = LocalWorkspaceTools(root, RuntimeConfig())

            self.assertTrue(tools.read_text_file(".env")["protected"])
            self.assertTrue(tools.write_text_file(".env", "other")["protected"])
            self.assertEqual(protected.read_text(encoding="utf-8"), "SECRET=value")

    def test_agent_cannot_modify_workspace_configuration(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = root / ".opencli" / "config.toml"
            config.parent.mkdir()
            config.write_text("[models]", encoding="utf-8")
            tools = LocalWorkspaceTools(root, RuntimeConfig())

            result = tools.write_text_file(".opencli/config.toml", "changed")

            self.assertTrue(result["protected"])
            self.assertEqual(config.read_text(encoding="utf-8"), "[models]")

    def test_edit_requires_exactly_one_match(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "note.txt").write_text("same same", encoding="utf-8")
            tools = LocalWorkspaceTools(root, RuntimeConfig())

            with self.assertRaisesRegex(ValueError, "exactly one"):
                tools.edit_text_file("note.txt", "same", "new")

    def test_dry_run_write_never_changes_file(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            target = root / "note.txt"
            target.write_text("old", encoding="utf-8")
            tools = LocalWorkspaceTools(root, RuntimeConfig(dry_run=True))

            result = tools.write_text_file("note.txt", "new")

            self.assertTrue(result["dry_run"])
            self.assertEqual(target.read_text(encoding="utf-8"), "old")


class ModelRegistryTests(TestCase):
    def test_local_profile_persists_and_cannot_replace_builtin_key(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            model_file = root / "example.gguf"
            model_file.write_bytes(b"GGUF")
            state_file = root / "models.json"
            registry = ModelRegistry(state_file)

            key = registry.add(
                name="My GGUF",
                source_type="local",
                path=str(model_file),
                reserved_keys={"auto"},
            )

            self.assertEqual(key, "my-gguf")
            self.assertEqual(ModelRegistry(state_file).models[key]["path"], str(model_file.resolve()))
            with self.assertRaises(ModelRegistryError):
                registry.add(
                    name="Auto",
                    source_type="local",
                    path=str(model_file),
                    reserved_keys={"auto"},
                )

    def test_context_bar_shows_no_model_until_inference_is_loaded(self):
        cli = OpenCLI(dry_run=True)
        self.assertIn("No model loaded", cli.context_bar())

    def test_run_opens_without_starting_the_default_model(self):
        cli = OpenCLI(dry_run=True)
        with (
            patch.object(cli, "clear"),
            patch.object(cli, "banner"),
            patch.object(cli, "get_session", return_value=object()),
            patch("main.cli.get_styled_input", return_value="/exit"),
            patch.object(cli, "init_engine") as initialize,
            patch("builtins.print"),
        ):
            cli.run()
        initialize.assert_not_called()

    def test_api_quantization_marker_never_reaches_local_loader(self):
        class FakeEngine:
            def __init__(self):
                self.loaded = None

            def load_model(self, mode, quant):
                self.loaded = (mode, quant)
                return True, "loaded"

        cli = OpenCLI(dry_run=True)
        cli.engine = FakeEngine()
        cli.quant = "api"
        with patch("main.cli.loading_spinner"):
            self.assertTrue(cli.load_model("auto", show_picker=False))
        self.assertEqual(cli.engine.loaded, ("auto", "int4"))

    def test_quant_picker_value_reaches_local_loader(self):
        class FakeEngine:
            def __init__(self):
                self.loaded = None

            def load_model(self, mode, quant):
                self.loaded = (mode, quant)
                return True, "loaded"

        cli = OpenCLI(dry_run=True)
        cli.engine = FakeEngine()
        with (
            patch("main.cli.loading_spinner"),
            patch.object(cli, "pick_quant", return_value="fp16"),
        ):
            self.assertTrue(cli.load_model("auto", show_picker=True))
        self.assertEqual(cli.engine.loaded, ("auto", "fp16"))

    def test_tools_off_command_rebuilds_runtime_in_chat_only_mode(self):
        cli = OpenCLI(dry_run=True)
        with patch("builtins.print"):
            self.assertTrue(cli.handle_command("/tools-off"))
        self.assertFalse(cli.tools_enabled)
        with patch("builtins.print"):
            self.assertTrue(cli.handle_command("/tools-on"))
        self.assertTrue(cli.tools_enabled)

    def test_auto_tool_routing_is_off_by_default_and_toggleable(self):
        cli = OpenCLI(dry_run=True)
        self.assertFalse(cli.auto_tool_routing)
        with patch("builtins.print"):
            self.assertTrue(cli.handle_command("/tool-auto on"))
        self.assertTrue(cli.auto_tool_routing)
        with patch("builtins.print"):
            self.assertTrue(cli.handle_command("/tool-auto off"))
        self.assertFalse(cli.auto_tool_routing)

    def test_prompt_session_erases_submitted_input_before_panel_render(self):
        cli = OpenCLI(dry_run=True)
        with patch("main.cli.PromptSession") as prompt_session:
            cli.get_session()
        self.assertTrue(prompt_session.call_args.kwargs["erase_when_done"])

    def test_api_activation_unloads_local_backend_and_saves_keyless_profile(self):
        class Engine:
            def __init__(self):
                self.client = None

            def configure_api(self, client):
                self.client = client
                return True, "API ready"

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            cli = OpenCLI(dry_run=True)
            cli.engine = Engine()
            cli.api_profiles = ApiProfileRegistry(root / "api-profiles.json")
            cli.permission_manager = PermissionManager(
                root,
                state_file=root / "permissions.json",
                approval_callback=lambda _request: "always_allow",
            )

            self.assertTrue(cli._activate_api("groq", "secret", "test/model"))

            self.assertEqual(cli.mode, "api")
            self.assertEqual(cli.quant, "api")
            self.assertEqual(cli.engine.client.model, "test/model")
            self.assertEqual(
                cli.api_profiles.default(),
                {"provider": "groq", "model": "test/model"},
            )


class ToolsOffRuntimeTests(TestCase):
    def test_disabled_runtime_exposes_no_tools(self):
        class Engine:
            current_mode = "test"
            MODELS = {"test": {"path": "test-model"}}

        runtime = PydanticAgentRuntime(
            Engine(),
            config=RuntimeConfig(persist_state=False, tools_enabled=False),
        )
        self.assertEqual(runtime.available_tools, [])

    def test_default_runtime_does_not_preemptively_route_tools(self):
        class Engine:
            current_mode = "test"
            MODELS = {"test": {"path": "test-model"}}

            def generate_runtime_stream(self, _prompt):
                yield {"type": "token", "content": "answer"}

        runtime = PydanticAgentRuntime(
            Engine(), config=RuntimeConfig(persist_state=False)
        )
        with (
            patch.object(runtime, "_ground_local_workspace_request") as local,
            patch.object(runtime, "_ground_explicit_web_request") as web,
        ):
            list(runtime.generate_stream("Read 3.5"))
        local.assert_not_called()
        web.assert_not_called()


class DockerSandboxTests(TestCase):
    @patch("main.sandbox.shutil.which", return_value="docker")
    @patch("main.sandbox.subprocess.run")
    def test_docker_invocation_has_no_network_read_only_mount_and_no_shell(
        self, mocked_run, _which
    ):
        mocked_run.side_effect = [
            SimpleNamespace(returncode=0),
            SimpleNamespace(returncode=0, stdout="ok", stderr=""),
        ]
        with TemporaryDirectory() as temporary_directory:
            sandbox = DockerSandbox(Path(temporary_directory))
            result = sandbox.run(["python", "-V"])

        invocation = mocked_run.call_args_list[-1]
        args = invocation.args[0]
        self.assertEqual(result["exit_code"], 0)
        self.assertIn("--network", args)
        self.assertIn("none", args)
        self.assertIn("readonly", args[args.index("--mount") + 1])
        self.assertFalse(invocation.kwargs["shell"])
