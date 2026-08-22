import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from main.agent_runtime import PydanticAgentRuntime, RuntimeConfig
from main.api_profiles import ApiProfileRegistry
from main.api_providers import OpenAICompatibleClient


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


class FakeRemoteClient:
    provider_name = "Test API"
    model = "test/model"

    def __init__(self):
        self.requests = []

    def stream_chat(self, messages, tools):
        self.requests.append((messages, tools))
        if len(self.requests) == 1:
            yield {
                "type": "tool_calls",
                "calls": [
                    {
                        "id": "call_write",
                        "name": "write_text_file",
                        "arguments": json.dumps(
                            {"path": "api-created.txt", "content": "made by tool"}
                        ),
                    }
                ],
            }
        else:
            yield {"type": "token", "content": "Created api-created.txt."}


class FakeRemoteEngine:
    backend = "remote_api"
    current_mode = "api"

    def __init__(self):
        self.api_client = FakeRemoteClient()
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


if __name__ == "__main__":
    unittest.main()
