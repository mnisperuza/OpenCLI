import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.error import HTTPError

from main.agent_runtime import PydanticAgentRuntime, RuntimeConfig
from main.api_profiles import ApiProfileRegistry
from main.api_providers import OpenAICompatibleClient, PROVIDERS


class FakeResponse:
    def __init__(self, payload=b"", lines=None):
        self.payload = payload
        self.lines = lines or []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.payload

    def __iter__(self):
        return iter(self.lines)


class ApiProviderClientTests(unittest.TestCase):
    def test_direct_providers_precede_optional_gateways(self):
        self.assertEqual(next(iter(PROVIDERS)), "groq")
        self.assertGreaterEqual(len(PROVIDERS), 15)

    def test_additional_direct_providers_use_openai_compatible_defaults(self):
        expected = {
            "cerebras": (
                "Cerebras",
                "https://api.cerebras.ai/v1",
                "CEREBRAS_API_KEY",
            ),
            "openai": (
                "OpenAI",
                "https://api.openai.com/v1",
                "OPENAI_API_KEY",
            ),
            "deepseek": (
                "DeepSeek",
                "https://api.deepseek.com",
                "DEEPSEEK_API_KEY",
            ),
            "xai": (
                "xAI",
                "https://api.x.ai/v1",
                "XAI_API_KEY",
            ),
            "nvidia": (
                "NVIDIA NIM",
                "https://integrate.api.nvidia.com/v1",
                "NVIDIA_API_KEY",
            ),
            "mistral": (
                "Mistral AI",
                "https://api.mistral.ai/v1",
                "MISTRAL_API_KEY",
            ),
            "fireworks": (
                "Fireworks AI",
                "https://api.fireworks.ai/inference/v1",
                "FIREWORKS_API_KEY",
            ),
            "together": (
                "Together AI",
                "https://api.together.xyz/v1",
                "TOGETHER_API_KEY",
            ),
        }

        for provider, (name, base_url, environment_variable) in expected.items():
            definition = PROVIDERS[provider]
            self.assertEqual(definition.name, name)
            self.assertEqual(definition.base_url, base_url)
            self.assertEqual(definition.environment_variable, environment_variable)
            self.assertEqual(
                OpenAICompatibleClient(provider, "secret").base_url,
                base_url,
            )

    def test_additional_direct_provider_model_routes_are_correct(self):
        expected = {
            "cerebras": "https://api.cerebras.ai/v1/models",
            "openai": "https://api.openai.com/v1/models",
            "deepseek": "https://api.deepseek.com/models",
            "xai": "https://api.x.ai/v1/models",
            "nvidia": "https://integrate.api.nvidia.com/v1/models",
            "mistral": "https://api.mistral.ai/v1/models",
            "fireworks": "https://api.fireworks.ai/inference/v1/models",
            "together": "https://api.together.xyz/v1/models",
        }
        payload = json.dumps({"data": [{"id": "provider-model"}]}).encode()

        for provider, url in expected.items():
            with self.subTest(provider=provider):
                client = OpenAICompatibleClient(provider, "secret")
                with patch(
                    "main.api_providers.urlopen", return_value=FakeResponse(payload)
                ) as call:
                    self.assertEqual(client.list_models(), ["provider-model"])
                request = call.call_args.args[0]
                self.assertEqual(request.full_url, url)
                self.assertEqual(request.get_header("Authorization"), "Bearer secret")

    def test_new_direct_providers_share_streaming_tool_contract(self):
        expected = {
            "openai": "https://api.openai.com/v1/chat/completions",
            "deepseek": "https://api.deepseek.com/chat/completions",
            "xai": "https://api.x.ai/v1/chat/completions",
            "nvidia": "https://integrate.api.nvidia.com/v1/chat/completions",
        }
        tools = [{"type": "function", "function": {"name": "inspect_workspace"}}]

        for provider, url in expected.items():
            with self.subTest(provider=provider):
                client = OpenAICompatibleClient(provider, "secret", "agent-model")
                with patch(
                    "main.api_providers.urlopen",
                    return_value=FakeResponse(lines=[b"data: [DONE]\\n"]),
                ) as call:
                    list(client.stream_chat([{"role": "user", "content": "Inspect"}], tools))

                request = call.call_args.args[0]
                body = json.loads(request.data.decode("utf-8"))
                self.assertEqual(request.full_url, url)
                self.assertEqual(request.get_header("Authorization"), "Bearer secret")
                self.assertTrue(body["stream"])
                self.assertEqual(body["tools"], tools)

    def test_litellm_remains_an_optional_gateway_provider(self):
        definition = PROVIDERS["litellm"]
        self.assertEqual(definition.name, "LiteLLM Gateway")
        self.assertEqual(definition.base_url, "http://127.0.0.1:4000")
        self.assertEqual(definition.environment_variable, "LITELLM_API_KEY")
        self.assertEqual(
            definition.base_url_environment_variable, "LITELLM_BASE_URL"
        )

    @patch.dict("main.api_providers.os.environ", {}, clear=True)
    def test_local_gateway_http_urls_are_allowed(self):
        self.assertEqual(
            OpenAICompatibleClient("freellmapi", "secret").base_url,
            "http://127.0.0.1:3001/v1",
        )
        self.assertEqual(
            OpenAICompatibleClient("litellm", "secret").base_url,
            "http://127.0.0.1:4000",
        )
        self.assertEqual(
            OpenAICompatibleClient("ds2api", "secret").base_url,
            "http://127.0.0.1:5001/v1",
        )

    @patch.dict("main.api_providers.os.environ", {}, clear=True)
    def test_freellmapi_lists_only_models_ready_on_the_local_router(self):
        payload = json.dumps(
            {
                "data": [
                    {
                        "id": "auto",
                        "context_length": 131072,
                        "supported_parameters": ["tools"],
                    },
                    {
                        "id": "deepseek-r1",
                        "context_window": 65536,
                        "supported_parameters": ["tools"],
                    },
                ]
            }
        ).encode()
        client = OpenAICompatibleClient("freellmapi", "unified-key")

        with patch("main.api_providers.urlopen", return_value=FakeResponse(payload)) as call:
            self.assertEqual(client.list_models(), ["auto", "deepseek-r1"])

        request = call.call_args.args[0]
        self.assertEqual(
            request.full_url,
            "http://127.0.0.1:3001/v1/models?ready=true",
        )
        self.assertEqual(request.get_header("Authorization"), "Bearer unified-key")
        self.assertEqual(client.model_metadata("auto")["context"], 131072)
        self.assertTrue(client.model_metadata("auto")["supports_tools"])

    @patch.dict(
        "main.api_providers.os.environ",
        {
            "FREELLMAPI_API_KEY": "dashboard-key",
            "FREELLMAPI_BASE_URL": "https://free.example/v1/",
        },
        clear=True,
    )
    def test_freellmapi_supports_environment_configuration(self):
        definition = PROVIDERS["freellmapi"]
        self.assertEqual(definition.api_key_from_environment(), "dashboard-key")
        self.assertEqual(
            OpenAICompatibleClient("freellmapi", "dashboard-key").base_url,
            "https://free.example/v1",
        )

    @patch.dict("main.api_providers.os.environ", {}, clear=True)
    def test_freellmapi_stream_uses_standard_chat_and_tool_contract(self):
        client = OpenAICompatibleClient("freellmapi", "unified-key", "auto:coding")
        tools = [{"type": "function", "function": {"name": "inspect_workspace"}}]

        with patch(
            "main.api_providers.urlopen",
            return_value=FakeResponse(lines=[b"data: [DONE]\n"]),
        ) as call:
            list(client.stream_chat([{"role": "user", "content": "Inspect"}], tools))

        request = call.call_args.args[0]
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(
            request.full_url,
            "http://127.0.0.1:3001/v1/chat/completions",
        )
        self.assertEqual(body["model"], "auto:coding")
        self.assertTrue(body["stream"])
        self.assertEqual(body["tools"], tools)

    @patch.dict(
        "main.api_providers.os.environ",
        {"LITELLM_BASE_URL": "http://gateway.example/v1"},
        clear=True,
    )
    def test_remote_gateway_http_url_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Invalid LiteLLM Gateway"):
            OpenAICompatibleClient("litellm", "secret")

    @patch.dict(
        "main.api_providers.os.environ",
        {"DS2API_BASE_URL": "https://deepseek.example/v1/"},
        clear=True,
    )
    def test_remote_gateway_https_url_is_allowed(self):
        self.assertEqual(
            OpenAICompatibleClient("ds2api", "secret").base_url,
            "https://deepseek.example/v1",
        )

    @patch.dict("main.api_providers.os.environ", {}, clear=True)
    def test_gateway_discovery_uses_shared_openai_path_and_bearer_auth(self):
        payload = json.dumps({"data": [{"id": "agent-model"}]}).encode()
        client = OpenAICompatibleClient("litellm", "gateway-secret")
        with patch("main.api_providers.urlopen", return_value=FakeResponse(payload)) as call:
            self.assertEqual(client.list_models(), ["agent-model"])
        request = call.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:4000/models")
        self.assertEqual(request.get_header("Authorization"), "Bearer gateway-secret")

    @patch.dict("main.api_providers.os.environ", {}, clear=True)
    def test_litellm_stream_uses_standard_chat_completions_contract(self):
        client = OpenAICompatibleClient("litellm", "gateway-secret", "claude-agent")
        tools = [{"type": "function", "function": {"name": "inspect_workspace"}}]
        with patch(
            "main.api_providers.urlopen",
            return_value=FakeResponse(lines=[b"data: [DONE]\n"]),
        ) as call:
            list(client.stream_chat([{"role": "user", "content": "Inspect"}], tools))

        request = call.call_args.args[0]
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(request.full_url, "http://127.0.0.1:4000/chat/completions")
        self.assertEqual(body["model"], "claude-agent")
        self.assertTrue(body["stream"])
        self.assertEqual(body["tools"], tools)

    def test_qwen_cloud_uses_dashscope_openai_compatible_api(self):
        payload = json.dumps({"data": [{"id": "qwen-plus"}]}).encode()
        client = OpenAICompatibleClient("qwen", "secret")

        with patch("main.api_providers.urlopen", return_value=FakeResponse(payload)) as call:
            self.assertEqual(client.list_models(), ["qwen-plus"])

        self.assertEqual(
            call.call_args.args[0].full_url,
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/models",
        )
        self.assertEqual(PROVIDERS["qwen"].environment_variable, "DASHSCOPE_API_KEY")

    @patch.dict(
        "main.api_providers.os.environ",
        {"QWEN_BASE_URL": "https://dashscope-us.aliyuncs.com/compatible-mode/v1/"},
    )
    def test_qwen_cloud_supports_regional_base_url(self):
        client = OpenAICompatibleClient("qwen", "secret", "qwen-plus")
        with patch(
            "main.api_providers.urlopen",
            return_value=FakeResponse(lines=[b"data: [DONE]\n"]),
        ) as call:
            list(client.stream_chat([], []))

        self.assertEqual(
            call.call_args.args[0].full_url,
            "https://dashscope-us.aliyuncs.com/compatible-mode/v1/chat/completions",
        )

    @patch.dict(
        "main.api_providers.os.environ",
        {"DASHSCOPE_API_KEY": "", "QWEN_API_KEY": "alias-secret"},
        clear=False,
    )
    def test_qwen_cloud_accepts_qwen_api_key_alias(self):
        self.assertEqual(
            PROVIDERS["qwen"].api_key_from_environment(), "alias-secret"
        )

    @patch.dict(
        "main.api_providers.os.environ",
        {"QWEN_BASE_URL": "http://unsafe.example/v1"},
    )
    def test_qwen_cloud_rejects_insecure_base_url(self):
        with self.assertRaisesRegex(ValueError, "Invalid Qwen Cloud"):
            OpenAICompatibleClient("qwen", "secret")

    def test_cancel_closes_active_api_stream(self):
        client = OpenAICompatibleClient("groq", "secret", "model")
        response = Mock()
        client._active_response = response

        client.cancel()

        self.assertTrue(client._cancel_requested.is_set())
        response.close.assert_called_once_with()
        self.assertTrue(client.capability_report()["cancellation"])

    def test_model_id_validation_preserves_provider_identifier(self):
        self.assertEqual(
            OpenAICompatibleClient("groq", "secret", "Meta-Llama/model").model,
            "Meta-Llama/model",
        )
        with self.assertRaisesRegex(ValueError, "Invalid API model ID"):
            OpenAICompatibleClient("groq", "secret", "bad model")

    def test_discovers_and_filters_openrouter_tool_models(self):
        payload = json.dumps(
            {
                "data": [
                    {"id": "tool/model", "supported_parameters": ["tools"]},
                    {"id": "plain/model", "supported_parameters": ["temperature"]},
                ]
            }
        ).encode()
        client = OpenAICompatibleClient("openrouter", "secret")
        with patch("main.api_providers.urlopen", return_value=FakeResponse(payload)) as call:
            self.assertEqual(client.list_models(), ["tool/model"])
        request = call.call_args.args[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer secret")
        self.assertEqual(request.full_url, "https://openrouter.ai/api/v1/models")

    def test_discovery_keeps_context_and_output_metadata(self):
        payload = json.dumps(
            {
                "data": [
                    {
                        "id": "tool/model",
                        "supported_parameters": ["tools"],
                        "context_length": 131072,
                        "top_provider": {"max_completion_tokens": 16384},
                    }
                ]
            }
        ).encode()
        client = OpenAICompatibleClient("openrouter", "secret")
        with patch("main.api_providers.urlopen", return_value=FakeResponse(payload)):
            client.list_models()

        self.assertEqual(
            client.model_metadata("tool/model"),
            {"supports_tools": True, "context": 131072, "max_tokens": 16384},
        )

    def test_discovery_reads_nested_and_camel_case_limits(self):
        payload = json.dumps(
            {
                "data": [
                    {
                        "id": "nested/model",
                        "inputTokenLimit": 200000,
                        "metadata": {"max_output_tokens": 12000},
                    }
                ]
            }
        ).encode()
        client = OpenAICompatibleClient("groq", "secret")
        with patch("main.api_providers.urlopen", return_value=FakeResponse(payload)):
            client.list_models()

        self.assertEqual(client.model_metadata("nested/model")["context"], 200000)
        self.assertEqual(client.model_metadata("nested/model")["max_tokens"], 12000)

    def test_streams_text_and_reassembles_native_tool_call(self):
        lines = [
            b'data: {"choices":[{"delta":{"content":"Hi "}}]}\n',
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"write_text_file","arguments":"{\\\"path\\\":"}}]}}]}\n',
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\\"a.txt\\\",\\\"content\\\":\\\"ok\\\"}"}}]}}]}\n',
            b"data: [DONE]\n",
        ]
        client = OpenAICompatibleClient("groq", "secret", "model")
        with patch("main.api_providers.urlopen", return_value=FakeResponse(lines=lines)):
            events = list(client.stream_chat([], []))
        self.assertEqual(events[0], {"type": "token", "content": "Hi "})
        self.assertEqual(events[1]["calls"][0]["id"], "call_1")
        self.assertEqual(events[1]["calls"][0]["name"], "write_text_file")
        self.assertEqual(
            json.loads(events[1]["calls"][0]["arguments"]),
            {"path": "a.txt", "content": "ok"},
        )

    def test_stream_exposes_provider_reported_usage(self):
        lines = [
            b'data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":4}}\n',
            b"data: [DONE]\n",
        ]
        client = OpenAICompatibleClient("groq", "secret", "model")
        with patch("main.api_providers.urlopen", return_value=FakeResponse(lines=lines)):
            events = list(client.stream_chat([], []))
        self.assertEqual(
            events,
            [{"type": "usage", "input_tokens": 12, "output_tokens": 4}],
        )

    def test_profile_output_limit_reaches_api_request(self):
        client = OpenAICompatibleClient("groq", "secret", "model")
        client.max_output_tokens = 2048
        with patch(
            "main.api_providers.urlopen", return_value=FakeResponse(lines=[b"data: [DONE]\n"])
        ) as call:
            list(client.stream_chat([], []))
        body = json.loads(call.call_args.args[0].data.decode("utf-8"))
        self.assertEqual(body["max_tokens"], 2048)

    def test_native_reasoning_effort_reaches_supported_api_request(self):
        client = OpenAICompatibleClient("groq", "secret", "model")
        client.reasoning_control = "api_parameter"
        client.reasoning_effort = "high"
        with patch(
            "main.api_providers.urlopen", return_value=FakeResponse(lines=[b"data: [DONE]\n"])
        ) as call:
            list(client.stream_chat([], []))
        body = json.loads(call.call_args.args[0].data.decode("utf-8"))
        self.assertEqual(body["reasoning_effort"], "high")

    def test_reasoning_effort_omitted_without_native_adapter(self):
        client = OpenAICompatibleClient("groq", "secret", "model")
        client.reasoning_effort = "high"
        with patch(
            "main.api_providers.urlopen", return_value=FakeResponse(lines=[b"data: [DONE]\n"])
        ) as call:
            list(client.stream_chat([], []))
        body = json.loads(call.call_args.args[0].data.decode("utf-8"))
        self.assertNotIn("reasoning_effort", body)

    def test_named_tool_choice_reaches_api_request(self):
        client = OpenAICompatibleClient("groq", "secret", "model")
        choice = {"type": "function", "function": {"name": "react_dispatch"}}
        tools = [{"type": "function", "function": {"name": "react_dispatch"}}]
        with patch(
            "main.api_providers.urlopen", return_value=FakeResponse(lines=[b"data: [DONE]\n"])
        ) as call:
            list(client.stream_chat([], tools, choice))
        body = json.loads(call.call_args.args[0].data.decode("utf-8"))
        self.assertEqual(body["tool_choice"], choice)

    def test_named_tool_choice_rejection_retries_with_auto(self):
        client = OpenAICompatibleClient("groq", "secret", "model")
        choice = {"type": "function", "function": {"name": "react_dispatch"}}
        tools = [{"type": "function", "function": {"name": "react_dispatch"}}]
        rejection = HTTPError(
            "https://api.groq.com/openai/v1/chat/completions",
            400,
            "unsupported tool_choice",
            {},
            BytesIO(b'{"error":"unsupported tool_choice"}'),
        )
        accepted = FakeResponse(lines=[
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"d1","function":{"name":"react_dispatch","arguments":"{\\"decision\\":\\"answer\\",\\"summary\\":\\"done\\"}"}}]}}]}\n',
            b"data: [DONE]\n",
        ])

        with patch(
            "main.api_providers.urlopen", side_effect=[rejection, accepted]
        ) as call:
            events = list(client.stream_chat([], tools, choice))

        choices = [
            json.loads(item.args[0].data.decode("utf-8"))["tool_choice"]
            for item in call.call_args_list
        ]
        self.assertEqual(choices, [choice, "auto"])
        self.assertEqual(events[0]["calls"][0]["name"], "react_dispatch")

    def test_stream_stops_at_hard_character_limit(self):
        lines = [
            b'data: {"choices":[{"delta":{"content":"abcdefgh"}}]}\n',
            b"data: [DONE]\n",
        ]
        client = OpenAICompatibleClient("groq", "secret", "model")
        client.max_stream_chars = 5
        with patch("main.api_providers.urlopen", return_value=FakeResponse(lines=lines)):
            events = list(client.stream_chat([], []))

        self.assertEqual(events[0], {"type": "token", "content": "abcde"})
        self.assertEqual(events[1]["type"], "output_limit")

    def test_profiles_persist_provider_and_model_but_not_key(self):
        with tempfile.TemporaryDirectory() as directory:
            state_file = Path(directory) / "api-profiles.json"
            registry = ApiProfileRegistry(state_file)
            key = registry.save("groq", "openai/gpt-oss-20b")

            self.assertEqual(key, "groq:openai/gpt-oss-20b")
            self.assertEqual(
                ApiProfileRegistry(state_file).default(),
                {"provider": "groq", "model": "openai/gpt-oss-20b"},
            )
            self.assertNotIn("key", state_file.read_text(encoding="utf-8").lower())
            registry.remove(key)
            self.assertIsNone(ApiProfileRegistry(state_file).default())

    def test_profiles_can_persist_discovered_context_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = ApiProfileRegistry(Path(directory) / "api-profiles.json")
            registry.save(
                "openrouter",
                "nvidia/model",
                context_window=131072,
                max_output_tokens=8192,
            )

            self.assertEqual(
                ApiProfileRegistry(registry.state_file).default()["context_window"],
                131072,
            )

    def test_profile_rejects_unsafe_context_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = ApiProfileRegistry(Path(directory) / "api-profiles.json")

            with self.assertRaisesRegex(ValueError, "half the context"):
                registry.save(
                    "groq",
                    "model",
                    context_window=4096,
                    max_output_tokens=4096,
                )


class FakeRemoteClient:
    provider_name = "Test API"
    model = "test/model"

    def __init__(self):
        self.requests = []

    def stream_chat(self, messages, tools, tool_choice="auto"):
        self.requests.append((messages, tools, tool_choice))
        if len(self.requests) == 1:
            yield {
                "type": "tool_calls",
                "calls": [
                    {
                        "id": "call_dispatch",
                        "name": "react_dispatch",
                        "arguments": json.dumps(
                            {
                                "decision": "act",
                                "summary": "File creation needs a tool",
                                "goal": "Create api-created.txt",
                                "paths": ["api-created.txt"],
                            }
                        ),
                    }
                ],
            }
        elif len(self.requests) == 2:
            yield {
                "type": "tool_calls",
                "calls": [{
                    "id": "call_write",
                    "name": "write_text_file",
                    "arguments": json.dumps(
                        {"path": "api-created.txt", "content": "made by tool"}
                    ),
                }],
            }
        elif len(self.requests) == 3:
            yield {
                "type": "tool_calls",
                "calls": [{
                    "id": "call_critique",
                    "name": "critique_and_plan",
                    "arguments": json.dumps({
                        "progress": "File created",
                        "evidence": ["write succeeded"],
                        "complete": True,
                    }),
                }],
            }
        else:
            yield {"type": "token", "content": "Created api-created.txt."}


class FakeRemoteEngine:
    backend = "remote_api"
    current_mode = "api"

    def __init__(self):
        self.api_client = FakeRemoteClient()
        self.MODELS = {"api": {"path": self.api_client.model}}


class ProseOnlyRemoteClient:
    provider_name = "Broken API"
    model = "broken/model"

    def __init__(self):
        self.requests = []

    def stream_chat(self, messages, tools, tool_choice="auto"):
        self.requests.append((messages, tools, tool_choice))
        yield {"type": "token", "content": "I'll inspect it now."}


class ProseOnlyRemoteEngine:
    backend = "remote_api"
    current_mode = "api"

    def __init__(self):
        self.api_client = ProseOnlyRemoteClient()
        self.MODELS = {"api": {"path": self.api_client.model}}


class RemoteAgentIntegrationTests(unittest.TestCase):
    def test_native_api_tool_call_executes_workspace_write(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            engine = FakeRemoteEngine()
            runtime = PydanticAgentRuntime(
                engine,
                workspace=workspace,
                config=RuntimeConfig(persist_state=False),
                permission_callback=lambda *_args: True,
            )
            events = list(runtime.generate_stream("Create api-created.txt"))

            self.assertEqual(
                (workspace / "api-created.txt").read_text(encoding="utf-8"),
                "made by tool",
            )
            self.assertGreaterEqual(len(engine.api_client.requests), 2)
            names = {
                tool["function"]["name"]
                for tool in engine.api_client.requests[0][1]
            }
            self.assertIn("write_text_file", names)
            self.assertTrue(any(event.get("type") == "token" for event in events))
            phases = [
                event.get("content") for event in events
                if event.get("type") == "react_state"
            ]
            self.assertIn("plan", phases)
            self.assertIn("act", phases)
            self.assertEqual(runtime.react.status()["phase"], "finish")

    def test_prose_only_response_remains_a_normal_agent_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            engine = ProseOnlyRemoteEngine()
            runtime = PydanticAgentRuntime(
                engine,
                workspace=workspace,
                config=RuntimeConfig(
                    persist_state=False,
                    react_decision_retries=2,
                ),
            )

            events = list(runtime.generate_stream("Review current workspace"))

        named_choices = [request[2] for request in engine.api_client.requests]
        self.assertEqual(named_choices, ["auto"])
        self.assertEqual(runtime.react.status()["phase"], "finish")
        self.assertEqual(runtime.react.status()["steps"], 0)
        self.assertFalse(any(event.get("type") == "tool" for event in events))
        self.assertTrue(any(
            "I'll inspect it now." in str(event.get("content", ""))
            for event in events
        ))

    def test_prose_before_native_tool_call_does_not_skip_execution(self):
        class MixedTurnClient:
            provider_name = "Mixed API"
            model = "mixed/model"

            def __init__(self):
                self.requests = []

            def stream_chat(self, messages, tools, tool_choice="auto"):
                self.requests.append((messages, tools, tool_choice))
                if len(self.requests) == 1:
                    yield {"type": "token", "content": "Let me read that first."}
                    yield {
                        "type": "tool_calls",
                        "calls": [{
                            "id": "read-improvements",
                            "name": "read_text_file",
                            "arguments": json.dumps({"path": "IMPROVEMENTS.md"}),
                        }],
                    }
                else:
                    yield {"type": "token", "content": "The file says: reliable loop."}

        class MixedTurnEngine:
            backend = "remote_api"
            current_mode = "api"

            def __init__(self):
                self.api_client = MixedTurnClient()
                self.MODELS = {"api": {"path": self.api_client.model}}

        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "IMPROVEMENTS.md").write_text(
                "reliable loop", encoding="utf-8"
            )
            engine = MixedTurnEngine()
            runtime = PydanticAgentRuntime(
                engine,
                workspace=workspace,
                config=RuntimeConfig(persist_state=False),
                permission_callback=lambda *_args: True,
            )
            events = list(runtime.generate_stream("Read IMPROVEMENTS.md"))

        self.assertEqual(len(engine.api_client.requests), 2)
        self.assertTrue(any(
            event.get("type") == "tool_result"
            and event.get("name") == "read_text_file"
            for event in events
        ))
        transcript = "".join(
            str(event.get("content", ""))
            for event in events
            if event.get("type") == "token"
        )
        self.assertIn("The file says: reliable loop.", transcript)
        self.assertNotIn("Tool not executed", transcript)


if __name__ == "__main__":
    unittest.main()
