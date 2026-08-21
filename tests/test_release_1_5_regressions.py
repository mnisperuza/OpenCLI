"""Small release-gate contracts across every supported execution boundary."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from main.agent_runtime import LocalWorkspaceTools, RuntimeConfig
from main.api_providers import OpenAICompatibleClient
from main.interfaces import ModelDescriptor, PermissionRequestData, ToolDescriptor
from main.model_registry import ModelRegistry
from main.sandbox import DockerSandbox
from main.web_retrieval import WebRetriever


class ReleaseOneFiveRegressionTests(TestCase):
    def test_local_model_profile_retains_context_and_output_limits(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "local.gguf"
            model.write_bytes(b"GGUF")
            registry = ModelRegistry(root / "models.json")
            key = registry.add(
                name="Release local model", source_type="local", path=str(model),
                context=8192, max_tokens=2048,
            )
            saved = ModelRegistry(root / "models.json").models[key]
            self.assertEqual(saved["context"], 8192)
            self.assertEqual(saved["max_tokens"], 2048)

    def test_api_model_id_keeps_provider_path(self):
        client = OpenAICompatibleClient("groq", "secret", "openai/gpt-oss-20b")
        self.assertEqual(client.model, "openai/gpt-oss-20b")

    def test_web_denial_never_constructs_client(self):
        factory_calls = []
        retriever = WebRetriever(
            client_factory=lambda: factory_calls.append(True),
            permission_callback=lambda *_args: False,
        )
        result = retriever.web_search("release test")
        self.assertTrue(result["permission_denied"])
        self.assertEqual(factory_calls, [])

    def test_workspace_tool_rejects_escape_path(self):
        with TemporaryDirectory() as directory:
            tools = LocalWorkspaceTools(Path(directory), RuntimeConfig())
            with self.assertRaisesRegex(ValueError, "trusted workspace"):
                tools.read_text_file("../outside.txt")

    @patch("main.sandbox.shutil.which", return_value="docker")
    @patch("main.sandbox.subprocess.run")
    def test_docker_command_remains_network_isolated(self, mocked_run, _which):
        mocked_run.side_effect = [
            SimpleNamespace(returncode=0),
            SimpleNamespace(returncode=0, stdout="ok", stderr=""),
        ]
        with TemporaryDirectory() as directory:
            result = DockerSandbox(Path(directory)).run(["python", "-V"])
        invocation = mocked_run.call_args_list[-1].args[0]
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(invocation[invocation.index("--network") + 1], "none")

    def test_interface_descriptors_preserve_security_metadata(self):
        model = ModelDescriptor("local", "Local", "llama_cpp", 32768, True)
        tool = ToolDescriptor("write_text_file", "workspace", True, "file_write")
        request = PermissionRequestData(
            "file_write", "write_text_file", "a.txt", "test", Path(".")
        )
        self.assertEqual(model.context_window, 32768)
        self.assertTrue(tool.mutates_workspace)
        self.assertEqual(request.category, "file_write")
