import json
import threading
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, mock_open, patch

from fenrir_agent.engine import InputPayload, FenrirAgentEngine


class FakeResponse:
    def __init__(self, events):
        self.events = events

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def __iter__(self):
        for event in self.events:
            yield f"data: {json.dumps(event)}\n".encode()
        yield b"data: [DONE]\n"


class NeverInterrupted:
    @staticmethod
    def is_interrupted():
        return False


class LlamaCppStreamTests(TestCase):
    def test_stop_generation_closes_local_stream_and_cancels_api(self):
        engine = object.__new__(FenrirAgentEngine)
        engine.interrupt_handler = Mock()
        engine.api_client = Mock()
        engine._response_lock = threading.Lock()
        engine._active_generation_response = Mock()

        engine.stop_generation()

        engine.interrupt_handler.interrupt.assert_called_once_with()
        engine.api_client.cancel.assert_called_once_with()
        engine._active_generation_response.close.assert_called_once_with()

    @patch("fenrir_agent.engine.urlopen")
    def test_native_reasoning_uses_llama_chat_template_kwargs(self, mocked_urlopen):
        mocked_urlopen.return_value = FakeResponse([])
        engine = object.__new__(FenrirAgentEngine)
        engine.interrupt_handler = NeverInterrupted()
        engine._current_response = ""
        engine.reasoning_control = "chat_template_kwargs"
        engine.reasoning_effort = "medium"

        list(engine._generate_llama_cpp_stream(
            InputPayload(prompt="test", enhanced_prompt="test"),
            {"path": "test/model"},
            100,
        ))

        body = json.loads(mocked_urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertEqual(
            body["chat_template_kwargs"], {"reasoning_effort": "medium"}
        )

    @patch.dict(
        "fenrir_agent.engine.os.environ",
        {"FENRIR_LLAMA_CPP_STARTUP_TIMEOUT": "1"},
    )
    @patch("fenrir_agent.engine.open", new_callable=mock_open)
    @patch("fenrir_agent.engine.subprocess.Popen")
    @patch("fenrir_agent.engine.time.monotonic", side_effect=[10.0, 11.0])
    def test_llama_cpp_startup_times_out_and_stops_process(
        self, _monotonic, mocked_popen, _open
    ):
        engine = object.__new__(FenrirAgentEngine)
        engine._llama_process = None
        engine._llama_log = None
        engine._llama_log_path = None
        engine._find_llama_cpp_executable = lambda: "llama-server"
        engine._llama_cpp_is_ready = lambda _url: False
        engine._llama_log_tail = lambda: "still loading"
        engine._llama_log_download_failures = lambda: 0
        engine.shutdown = Mock()
        mocked_popen.return_value.poll.return_value = None

        success, message = engine._start_llama_cpp_server(
            {"path": "test/model", "context": 2048}, "int4"
        )

        self.assertFalse(success)
        self.assertIn("within 1 seconds", message)
        self.assertIn("still loading", message)
        engine.shutdown.assert_called_once_with()

    @patch("fenrir_agent.engine.open", new_callable=mock_open)
    @patch("fenrir_agent.engine.subprocess.Popen")
    def test_llama_cpp_startup_can_be_cancelled(self, mocked_popen, _open):
        engine = object.__new__(FenrirAgentEngine)
        engine._llama_process = None
        engine._llama_log = None
        engine._llama_log_path = None
        engine._llama_starting = False
        engine.interrupt_handler = Mock()
        engine.interrupt_handler.is_interrupted.return_value = True
        engine._find_llama_cpp_executable = lambda: "llama-server"
        engine._llama_cpp_is_ready = lambda _url: False
        engine.shutdown = Mock()
        mocked_popen.return_value.poll.return_value = None

        success, message = engine._start_llama_cpp_server(
            {"path": "test/model", "context": 2048}, "int4"
        )

        self.assertFalse(success)
        self.assertEqual(message, "llama.cpp startup cancelled.")
        engine.shutdown.assert_called_once_with()

    @patch("fenrir_agent.engine.open", new_callable=mock_open)
    @patch("fenrir_agent.engine.subprocess.Popen")
    def test_llama_cpp_download_failure_stops_startup(self, mocked_popen, _open):
        engine = object.__new__(FenrirAgentEngine)
        engine._llama_process = None
        engine._llama_log = None
        engine._llama_log_path = None
        engine._llama_starting = False
        engine.interrupt_handler = NeverInterrupted()
        engine._find_llama_cpp_executable = lambda: "llama-server"
        engine._llama_cpp_is_ready = lambda _url: False
        engine._llama_log_download_failures = lambda: 1
        engine._llama_log_tail = lambda: "download failed: connection reset"
        engine.shutdown = Mock()
        mocked_popen.return_value.poll.return_value = None

        success, message = engine._start_llama_cpp_server(
            {"path": "test/model", "context": 2048}, "int4"
        )

        self.assertFalse(success)
        self.assertIn("could not download", message)
        engine.shutdown.assert_called_once_with()

    def test_stop_generation_terminates_only_in_progress_server_startup(self):
        engine = object.__new__(FenrirAgentEngine)
        engine.interrupt_handler = Mock()
        engine.api_client = None
        engine._response_lock = threading.Lock()
        engine._active_generation_response = None
        engine._llama_starting = True
        engine.shutdown = Mock()

        engine.stop_generation()

        engine.shutdown.assert_called_once_with()

    def test_load_model_resets_and_clears_cancellation_state(self):
        engine = object.__new__(FenrirAgentEngine)
        engine.interrupt_handler = Mock()
        engine._model_loading = False
        engine._load_model_impl = Mock(return_value=(True, "ready"))

        result = engine.load_model("auto", "int4")

        self.assertEqual(result, (True, "ready"))
        engine.interrupt_handler.reset.assert_called_once_with()
        self.assertFalse(engine._model_loading)

    @patch("fenrir_agent.engine.os.name", "nt")
    @patch("fenrir_agent.engine.subprocess.run")
    def test_endserver_stops_unmanaged_local_llama_server(self, mocked_run):
        mocked_run.side_effect = [
            SimpleNamespace(
                stdout="  TCP    127.0.0.1:8080    0.0.0.0:0    LISTENING    4321\n"
            ),
            SimpleNamespace(stdout='"llama-server.exe","4321","Console","1","200 K"\n'),
            SimpleNamespace(stdout=""),
        ]
        engine = object.__new__(FenrirAgentEngine)

        engine._stop_unmanaged_windows_llama_cpp_server()

        self.assertEqual(
            mocked_run.call_args_list[-1].args[0],
            ["taskkill", "/PID", "4321", "/T", "/F"],
        )

    def test_stale_server_is_stopped_before_default_model_starts(self):
        engine = object.__new__(FenrirAgentEngine)
        engine.shutdown = lambda: None
        engine._llama_cpp_is_ready = lambda _url: False
        engine._llama_cpp_model_ids = lambda _url: ["LiquidAI/LFM2.5-8B-A1B-GGUF"]
        engine._start_llama_cpp_server = lambda _model, _quant: (True, "")
        engine.model = None
        engine.backend = None

        success, _message = engine._load_llama_cpp_model(
            "auto",
            {"path": "mistralai/Ministral-3-14B-Instruct-2512-GGUF", "name": "Ministral 3 14B Instruct"},
            "int4",
        )

        self.assertTrue(success)
        self.assertEqual(engine.current_mode, "auto")

    def test_model_identity_match_accepts_quantized_hugging_face_id(self):
        self.assertTrue(
            FenrirAgentEngine._llama_cpp_model_matches(
                "LiquidAI/LFM2.5-8B-A1B-GGUF",
                "LiquidAI/LFM2.5-8B-A1B-GGUF:Q4_K_M",
            )
        )
        self.assertFalse(
            FenrirAgentEngine._llama_cpp_model_matches(
                "mistralai/Ministral-3-14B-Instruct-2512-GGUF",
                "LiquidAI/LFM2.5-8B-A1B-GGUF:Q4_K_M",
            )
        )

    @patch("fenrir_agent.engine.platform.system", return_value="Darwin")
    @patch("fenrir_agent.engine.shutil.which", return_value=None)
    def test_macos_uses_homebrew_llama_cpp_guidance(self, _which, _system):
        self.assertEqual(
            FenrirAgentEngine._llama_cpp_install_hint(), "brew install llama.cpp"
        )

    @patch("fenrir_agent.engine.urlopen")
    def test_structured_tool_call_delta_becomes_adapter_protocol(self, mocked_urlopen):
        mocked_urlopen.return_value = FakeResponse(
            [
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {
                                            "name": "web_search",
                                            "arguments": '{"query":"2026 World',
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {
                                            "arguments": ' Cup","max_results":5}',
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                },
            ]
        )
        engine = object.__new__(FenrirAgentEngine)
        engine.interrupt_handler = NeverInterrupted()
        engine._current_response = ""

        events = list(
            engine._generate_llama_cpp_stream(
                InputPayload(prompt="test", enhanced_prompt="test"),
                {"path": "test/model"},
                100,
            )
        )

        tokens = [event["content"] for event in events if event["type"] == "token"]
        self.assertEqual(len(tokens), 1)
        self.assertIn('"name": "web_search"', tokens[0])
        self.assertIn('"max_results": 5', tokens[0])
        self.assertEqual(events[-1]["type"], "done")
