import sqlite3
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from main.web_retrieval import WebRetrievalError, WebRetriever
from main.agent_runtime import LocalModelAdapter, PydanticAgentRuntime, RuntimeConfig


def make_runtime(engine, workspace=None):
    return PydanticAgentRuntime(
        engine,
        workspace=workspace,
        config=RuntimeConfig(
            persist_state=False, auto_tool_routing=True, react_enabled=False
        ),
    )


class FakeDDGS:
    def __init__(self, results=None, extracted=None, error=None):
        self.results = results or []
        self.extracted = extracted or {"content": ""}
        self.error = error
        self.search_kwargs = None

    def text(self, query, **kwargs):
        if self.error:
            raise self.error
        self.search_kwargs = {"query": query, **kwargs}
        return self.results

    def extract(self, url, fmt="text_markdown"):
        if self.error:
            raise self.error
        return self.extracted


class WebRetrieverTests(TestCase):
    @patch("main.web_retrieval._is_public_web_url", return_value=True)
    def test_per_turn_fetch_limit_bounds_tool_context(self, _public_check):
        client = FakeDDGS(extracted={"content": "evidence"})
        retriever = WebRetriever(
            client_factory=lambda: client,
            max_fetches_per_turn=1,
        )

        first = retriever.web_fetch("https://example.com/one")
        second = retriever.web_fetch("https://example.com/two")
        retriever.begin_turn()
        third = retriever.web_fetch("https://example.com/three")

        self.assertEqual(first["content"], "evidence")
        self.assertIn("fetch limit", second["error"])
        self.assertEqual(third["content"], "evidence")

    def test_permission_denial_prevents_search_request(self):
        client = FakeDDGS(error=AssertionError("search must not run"))
        retriever = WebRetriever(
            client_factory=lambda: client,
            permission_callback=lambda *_args: False,
        )

        output = retriever.web_search("private query")

        self.assertTrue(output["permission_denied"])
        self.assertIsNone(client.search_kwargs)

    @patch("main.web_retrieval._is_public_web_url")
    def test_permission_denial_prevents_fetch_dns_lookup(self, public_check):
        retriever = WebRetriever(permission_callback=lambda *_args: False)

        output = retriever.web_fetch("https://example.com/article")

        self.assertTrue(output["permission_denied"])
        public_check.assert_not_called()

    def test_search_normalizes_deduplicates_and_caps_results(self):
        client = FakeDDGS(
            results=[
                {
                    "title": "  First   result ",
                    "href": "HTTPS://Example.COM/page?utm_source=x&id=1#part",
                    "body": " Fresh\n evidence ",
                },
                {
                    "title": "Duplicate",
                    "href": "https://example.com/page?id=1&utm_medium=y",
                    "body": "duplicate",
                },
                {
                    "title": "Second",
                    "href": "https://example.org/other",
                    "body": "More evidence",
                },
            ]
        )
        retriever = WebRetriever(max_results=10, client_factory=lambda: client)

        output = retriever.web_search("  latest   test ", max_results=2)

        self.assertEqual(output["query"], "latest test")
        self.assertEqual(output["result_count"], 2)
        self.assertEqual(output["results"][0]["rank"], 1)
        self.assertEqual(
            output["results"][0]["url"], "https://example.com/page?id=1"
        )
        self.assertEqual(output["results"][0]["snippet"], "Fresh evidence")
        self.assertTrue(output["retrieved_at"].endswith("Z"))
        self.assertEqual(client.search_kwargs["backend"], "auto")
        self.assertEqual(client.search_kwargs["max_results"], 4)

    def test_search_wraps_provider_failures(self):
        client = FakeDDGS(error=RuntimeError("offline"))
        retriever = WebRetriever(client_factory=lambda: client)

        with self.assertRaisesRegex(WebRetrievalError, "Web search failed"):
            retriever.web_search("test")

    @patch("main.web_retrieval._is_public_web_url", return_value=True)
    def test_fetch_bounds_extracted_content(self, _public_url):
        client = FakeDDGS(extracted={"content": "x" * 1_500})
        retriever = WebRetriever(
            max_content_chars=1_000,
            client_factory=lambda: client,
        )

        output = retriever.web_fetch("https://example.com/article")

        self.assertEqual(len(output["content"]), 1_000)
        self.assertTrue(output["truncated"])

    @patch("main.web_retrieval._is_public_web_url", return_value=True)
    def test_fetch_failure_is_recoverable_after_one_retry(self, _public_url):
        class FailingExtractor:
            def __init__(self):
                self.calls = 0

            def extract(self, _url, fmt="text_markdown"):
                self.calls += 1
                raise RuntimeError("HTTP 404")

        client = FailingExtractor()
        retriever = WebRetriever(client_factory=lambda: client)

        output = retriever.web_fetch("https://example.com/missing")

        self.assertEqual(client.calls, 2)
        self.assertTrue(output["recoverable"])
        self.assertEqual(output["content"], "")
        self.assertIn("HTTP 404", output["error"])

    def test_empty_query_is_rejected(self):
        retriever = WebRetriever(client_factory=FakeDDGS)
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            retriever.web_search("   ")


class FakeEngine:
    MODELS = {"test": {"path": "local/test-model"}}
    current_mode = "test"

    def __init__(self):
        self.prompts = []

    def generate_runtime_stream(self, prompt):
        self.prompts.append(prompt)
        if len(self.prompts) == 1:
            content = (
                '<tool_call>{"name":"web_search","arguments":'
                '{"query":"current OpenCLI test","max_results":1}}'
                "</tool_call>"
            )
        else:
            content = "Current answer with https://example.com/current"
        yield {"type": "token", "content": content}


class ToolProtocolTests(TestCase):
    def test_local_parser_deduplicates_identical_tool_calls(self):
        adapter = LocalModelAdapter(FakeEngine())
        calls = adapter._parse_tool_calls(
            '<tool_call>{"name":"web_search","arguments":{"query":"test"}}</tool_call>'
            '<tool_call>{"name":"web_search","arguments":{"query":"test"}}</tool_call>',
            {"web_search"},
        )
        self.assertEqual(calls, [{"name": "web_search", "arguments": {"query": "test"}}])


class LFMFakeEngine(FakeEngine):
    MODELS = {"lfm2.5-8b-a1b": {"path": "LiquidAI/LFM2.5-8B-A1B-GGUF"}}
    current_mode = "lfm2.5-8b-a1b"

    def generate_runtime_stream(self, prompt):
        self.prompts.append(prompt)
        if len(self.prompts) == 1:
            content = (
                "<|tool_call_start|>[web_search("
                "query='current OpenCLI test', max_results=1)]"
                "<|tool_call_end|>"
            )
        else:
            content = "Current answer with https://example.com/current"
        yield {"type": "token", "content": content}


class PrefixedToolEngine(FakeEngine):
    def generate_runtime_stream(self, prompt):
        self.prompts.append(prompt)
        if len(self.prompts) == 1:
            yield {"type": "token", "content": "I will inspect it first.\n"}
            yield {
                "type": "token",
                "content": (
                    "<tool_call>{\"name\": \"web_search\", "
                    "\"arguments\": {\"query\": \"current OpenCLI test\"}}"
                    "</tool_call>"
                ),
            }
        else:
            yield {"type": "token", "content": "Current answer"}


class InvalidToolEngine(FakeEngine):
    def generate_runtime_stream(self, prompt):
        self.prompts.append(prompt)
        yield {
            "type": "token",
            "content": (
                '<tool_call>{"name":"web_search","arguments":{"query":"bad "quote"}}'
                "</tool_call>"
            ),
        }


class DirectAnswerEngine(FakeEngine):
    def generate_runtime_stream(self, prompt):
        self.prompts.append(prompt)
        yield {
            "type": "token",
            "content": "Grounded answer with https://example.com/current",
        }


class DelayedWriteEngine(FakeEngine):
    def generate_runtime_stream(self, prompt):
        self.prompts.append(prompt)
        if len(self.prompts) == 1:
            content = "I created new.txt."
        elif len(self.prompts) == 2:
            content = (
                '<tool_call>{"name":"write_text_file","arguments":'
                '{"path":"new.txt","content":"hello"}}</tool_call>'
            )
        else:
            content = "Created new.txt using workspace tool."
        yield {"type": "token", "content": content}


class AgentWebToolTests(TestCase):
    def test_denied_explicit_search_does_not_prompt_again_in_same_run(self):
        engine = FakeEngine()
        decisions = []
        runtime = PydanticAgentRuntime(
            engine,
            config=RuntimeConfig(
                persist_state=False, auto_tool_routing=True, react_enabled=False
            ),
            permission_callback=lambda *request: decisions.append(request) or False,
        )
        client = FakeDDGS(error=AssertionError("search must not run"))
        runtime.web._client_factory = lambda: client

        events = list(runtime.generate_stream("Search current information"))

        self.assertEqual(len(decisions), 1)
        self.assertIsNone(client.search_kwargs)
        self.assertEqual(events[-1]["type"], "done")

    def test_search_runs_as_tool_observation_in_agent_loop(self):
        engine = FakeEngine()
        runtime = make_runtime(engine)
        client = FakeDDGS(
            results=[
                {
                    "title": "Current source",
                    "href": "https://example.com/current",
                    "body": "FAKE_CURRENT_EVIDENCE",
                }
            ]
        )
        runtime.web._client_factory = lambda: client

        events = list(runtime.generate_stream("What is current?"))

        self.assertEqual(len(engine.prompts), 2)
        self.assertNotIn("FAKE_CURRENT_EVIDENCE", engine.prompts[0])
        self.assertIn("TOOL RESULT [web_search]", engine.prompts[1])
        self.assertIn("FAKE_CURRENT_EVIDENCE", engine.prompts[1])
        self.assertTrue(
            any(
                event.get("type") == "tool"
                and event.get("name") == "web_search"
                for event in events
            )
        )
        self.assertEqual(events[-1]["type"], "done")

    def test_lfm_native_tool_call_runs_as_tool_observation(self):
        engine = LFMFakeEngine()
        runtime = make_runtime(engine)
        client = FakeDDGS(
            results=[
                {
                    "title": "Current source",
                    "href": "https://example.com/current",
                    "body": "FAKE_CURRENT_EVIDENCE",
                }
            ]
        )
        runtime.web._client_factory = lambda: client

        events = list(runtime.generate_stream("What is current?"))

        self.assertIn("<|tool_call_start|>", engine.prompts[0])
        self.assertEqual(len(engine.prompts), 2)
        self.assertIn("TOOL RESULT [web_search]", engine.prompts[1])
        self.assertTrue(
            any(
                event.get("type") == "tool"
                and event.get("name") == "web_search"
                for event in events
            )
        )
        self.assertEqual(events[-1]["type"], "done")

    def test_tool_tag_after_role_prefix_is_not_rendered_as_text(self):
        engine = PrefixedToolEngine()
        runtime = make_runtime(engine)
        runtime.web._client_factory = lambda: FakeDDGS(
            results=[{"title": "Current", "href": "https://example.com", "body": "evidence"}]
        )

        events = list(runtime.generate_stream("What is current?"))

        self.assertTrue(any(event.get("type") == "tool" for event in events))
        self.assertFalse(
            any("<tool_call>" in event.get("content", "") for event in events)
        )

    def test_invalid_tool_json_is_rejected_without_exposing_tag(self):
        events = list(make_runtime(InvalidToolEngine()).generate_stream("Search now"))
        text = "".join(event.get("content", "") for event in events)
        self.assertIn("Tool call rejected", text)
        self.assertNotIn("<tool_call>", text)

    def test_explicit_search_is_grounded_before_model_routing(self):
        engine = DirectAnswerEngine()
        runtime = make_runtime(engine)
        client = FakeDDGS(
            results=[
                {
                    "title": "Current source",
                    "href": "https://example.com/current",
                    "body": "FAKE_CURRENT_EVIDENCE",
                }
            ]
        )
        runtime.web._client_factory = lambda: client

        events = list(runtime.generate_stream("Search who won the 2026 World Cup"))

        self.assertEqual(len(engine.prompts), 1)
        self.assertIn("LIVE WEB SEARCH EVIDENCE", engine.prompts[0])
        self.assertIn("FAKE_CURRENT_EVIDENCE", engine.prompts[0])
        self.assertEqual(client.search_kwargs["max_results"], 10)
        self.assertTrue(
            any(
                event.get("type") == "tool"
                and event.get("name") == "web_search"
                for event in events
            )
        )

    def test_local_file_search_uses_pathlib_not_web_retrieval(self):
        with TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            (workspace / "hello.txt").write_text(
                "Hello from local evidence", encoding="utf-8"
            )
            engine = DirectAnswerEngine()
            runtime = make_runtime(engine, workspace=workspace)
            client = FakeDDGS(error=AssertionError("web search must not run"))
            runtime.web._client_factory = lambda: client

            events = list(
                runtime.generate_stream(
                    "Can you search the hello.txt file from the current directory?"
                )
            )

            self.assertIn("LOCAL WORKSPACE EVIDENCE", engine.prompts[0])
            self.assertIn("Hello from local evidence", engine.prompts[0])
            self.assertNotIn("LIVE WEB SEARCH EVIDENCE", engine.prompts[0])
            self.assertIsNone(client.search_kwargs)
            self.assertTrue(
                any(
                    event.get("type") == "tool"
                    and event.get("name") == "read_text_file"
                    for event in events
                )
            )

    def test_file_creation_reads_source_but_not_missing_output(self):
        with TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            (workspace / "index.html").write_text(
                "<h1>Source</h1>", encoding="utf-8"
            )
            decisions = []
            engine = DirectAnswerEngine()
            runtime = PydanticAgentRuntime(
                engine,
                workspace=workspace,
                config=RuntimeConfig(
                    persist_state=False, auto_tool_routing=True, react_enabled=False
                ),
                permission_callback=lambda *request: decisions.append(request) or True,
            )

            list(
                runtime.generate_stream(
                    "Create betterindex.html from index.html and improve it"
                )
            )

            self.assertIn("<h1>Source</h1>", engine.prompts[0])
            self.assertIn('"path": "betterindex.html", "status": "does not exist yet"', engine.prompts[0])
            self.assertIn("write_text_file", engine.prompts[0])
            self.assertEqual(len(decisions), 1)
            self.assertEqual(decisions[0][1], "read_text_file")

    def test_mutation_retries_false_claim_then_requires_write_tool(self):
        with TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            engine = DelayedWriteEngine()
            runtime = PydanticAgentRuntime(
                engine,
                workspace=workspace,
                config=RuntimeConfig(
                    persist_state=False, auto_tool_routing=True, react_enabled=False
                ),
                permission_callback=lambda *_request: True,
            )

            events = list(runtime.generate_stream("Create new.txt containing hello"))

            self.assertEqual((workspace / "new.txt").read_text(encoding="utf-8"), "hello")
            self.assertEqual(len(engine.prompts), 3)
            self.assertTrue(any(event.get("type") == "status" and "retrying" in event.get("content", "") for event in events))
            self.assertFalse(any(event.get("type") == "token" and event.get("content") == "I created new.txt." for event in events))

    def test_mutation_failure_reports_file_unchanged(self):
        with TemporaryDirectory() as temporary_directory:
            runtime = PydanticAgentRuntime(
                DirectAnswerEngine(),
                workspace=Path(temporary_directory),
                config=RuntimeConfig(
                    persist_state=False, auto_tool_routing=True, react_enabled=False
                ),
            )

            events = list(runtime.generate_stream("Create missing.txt"))
            text = "".join(event.get("content", "") for event in events if event.get("type") == "token")

            self.assertIn("File unchanged", text)
            self.assertFalse((Path(temporary_directory) / "missing.txt").exists())

    def test_new_session_id_does_not_restore_workspace_history(self):
        with TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            database = workspace / "state.sqlite3"
            first = PydanticAgentRuntime(
                DirectAnswerEngine(),
                workspace=workspace,
                config=RuntimeConfig(persist_state=True, state_db_path=database, session_id="one"),
            )
            list(first.generate_stream("hello"))

            second = PydanticAgentRuntime(
                DirectAnswerEngine(),
                workspace=workspace,
                config=RuntimeConfig(persist_state=True, state_db_path=database, session_id="two"),
            )

            self.assertEqual(second.message_count, 0)

    def test_sqlite_restores_conversation_and_tool_state(self):
        with TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            (workspace / "hello.txt").write_text("Persistent hello", encoding="utf-8")
            database = workspace / "state.sqlite3"
            config = RuntimeConfig(
                persist_state=True,
                state_db_path=database,
                session_id="test-session",
                auto_tool_routing=True,
            )
            first = PydanticAgentRuntime(
                DirectAnswerEngine(), workspace=workspace, config=config
            )

            list(first.generate_stream("Read the hello.txt file"))
            message_count = first.message_count

            second = PydanticAgentRuntime(
                DirectAnswerEngine(), workspace=workspace, config=config
            )
            self.assertEqual(second.message_count, message_count)
            with closing(sqlite3.connect(database)) as connection:
                tool_events = connection.execute(
                    "SELECT COUNT(*) FROM tool_events WHERE session_id = ?",
                    ("test-session",),
                ).fetchone()[0]
            self.assertGreater(tool_events, 0)

            second.clear()
            third = PydanticAgentRuntime(
                DirectAnswerEngine(), workspace=workspace, config=config
            )
            self.assertEqual(third.message_count, 0)
