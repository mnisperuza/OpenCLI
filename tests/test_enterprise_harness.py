"""Release gates for durable, evidence-backed enterprise harness behavior."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from urllib.error import URLError

from fenrir_agent.agent_runtime import PydanticAgentRuntime, RuntimeConfig
from fenrir_agent.harness_contracts import (
    ErrorCode,
    MemoryRecord,
    RunLifecycle,
    RunState,
    ToolOutcome,
    ToolStatus,
    TrustClass,
    new_id,
)
from fenrir_agent.provider_reliability import (
    ProviderReliabilityController,
    TransportRetryPolicy,
)
from fenrir_agent.react_loop import ReactLoopController, ReactLoopPolicy, ReactPhase
from fenrir_agent.run_ledger import RunLedger
from fenrir_agent.tool_runtime import (
    CompletionValidator,
    DeterministicReadBatchExecutor,
    default_tool_registry,
    default_toolset_registry,
    mutation_receipt,
)
from fenrir_agent.workspace_context import WorkspaceContext


class HarnessContractTests(TestCase):
    def test_toolsets_filter_capabilities_without_changing_tool_policy(self):
        registry = default_toolset_registry()
        self.assertIn("workspace", registry.names)
        self.assertIn("web_search", registry.enabled_tools(("web",)))
        self.assertNotIn("write_text_file", registry.enabled_tools(("web",)))

        class Engine:
            current_mode = "test"
            MODELS = {"test": {"path": "test-model"}}

        runtime = PydanticAgentRuntime(
            Engine(),
            config=RuntimeConfig(
                persist_state=False,
                enabled_toolsets=("workspace", "planning"),
                react_strict_control=True,
            ),
        )
        self.assertIn("list_files", runtime.available_tools)
        self.assertIn("react_dispatch", runtime.available_tools)
        self.assertNotIn("web_search", runtime.available_tools)
        self.assertNotIn('"name": "web_search"', runtime._tool_prompt_text)

    def test_summary_prose_never_controls_tool_status(self):
        outcome = ToolOutcome.from_event(
            {"summary": "failed error denied unchanged but this is only prose"}
        )
        self.assertEqual(outcome.status, ToolStatus.SUCCESS)

        controller = ReactLoopController()
        controller.begin_turn("inspect")
        controller.start_task("inspect")
        controller.before_tool("read_text_file", {"path": "a.py"})
        controller.after_tool({"summary": "failed is a word in the file"})
        self.assertEqual(controller.status()["failures"], 0)

    def test_explicit_typed_failures_control_react_budget(self):
        controller = ReactLoopController(ReactLoopPolicy(max_consecutive_failures=1))
        controller.begin_turn("run checks")
        controller.start_task("run checks")
        controller.before_tool("run", {"command": ["test"]})
        controller.after_tool(
            {
                "summary": "check unavailable",
                "outcome": ToolOutcome(
                    status=ToolStatus.RETRYABLE_ERROR,
                    summary="check unavailable",
                    error_code=ErrorCode.EXECUTION_FAILED,
                ).model_dump(mode="json"),
            }
        )
        self.assertEqual(controller.state.phase, ReactPhase.HALTED)
        self.assertEqual(controller.status()["failures"], 1)

    def test_completion_requires_verified_mutation_receipt(self):
        validator = CompletionValidator()
        unverified = ToolOutcome.success(
            "wrote file",
            evidence_ids=("e1",),
            changed=True,
            receipt=mutation_receipt(
                "a.py", pre_hash="old", post_hash="new", verified=False
            ),
        )
        rejected = validator.validate([unverified])
        self.assertFalse(rejected.accepted)
        verified = unverified.model_copy(
            update={
                "receipt": mutation_receipt(
                    "a.py", pre_hash="old", post_hash="new", verified=True
                )
            }
        )
        self.assertTrue(validator.validate([verified]).accepted)


class RunLedgerTests(TestCase):
    @staticmethod
    def _state(session_id: str = "session", **updates) -> RunState:
        state = RunState(
            run_id=new_id("run"),
            session_id=session_id,
            turn_id=new_id("turn"),
            goal="test durable execution",
        )
        return state.model_copy(update=updates)

    def test_existing_sqlite_tables_survive_harness_migration(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "state.sqlite3"
            connection = sqlite3.connect(path)
            connection.executescript(
                """
                CREATE TABLE conversations (
                    session_id TEXT PRIMARY KEY,
                    messages_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE tool_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    event_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            connection.execute(
                "INSERT INTO conversations VALUES ('session', '[]', 'before')"
            )
            connection.commit()
            connection.close()

            RunLedger(path, "session")

            connection = sqlite3.connect(path)
            row = connection.execute(
                "SELECT messages_json, updated_at FROM conversations WHERE session_id = 'session'"
            ).fetchone()
            version = connection.execute(
                "SELECT version FROM harness_schema WHERE component = 'enterprise_harness'"
            ).fetchone()[0]
            connection.close()
            self.assertEqual(row, ("[]", "before"))
            self.assertEqual(version, 1)

    def test_event_sequences_are_monotonic_and_abandoned_run_recovers(self):
        with TemporaryDirectory() as directory:
            ledger = RunLedger(Path(directory) / "state.sqlite3", "session")
            state = ledger.begin_run(self._state())
            ledger.append_event("model.requested", state, {"chars": 3})
            ledger.append_event("model.responded", state, {"chars": 4})
            self.assertEqual(
                [event.sequence for event in ledger.events(state.run_id)],
                [1, 2, 3],
            )
            recovered = ledger.mark_abandoned_recovering()
            self.assertEqual(recovered[0].lifecycle, RunLifecycle.RECOVERING)

    def test_execution_idempotency_and_receipt_completion(self):
        with TemporaryDirectory() as directory:
            ledger = RunLedger(Path(directory) / "state.sqlite3", "session")
            state = ledger.begin_run(
                self._state(step_id="step_1", lifecycle=RunLifecycle.PENDING)
            )
            first = ledger.begin_execution(state, "read_text_file", {"path": "a.py"})
            repeated = ledger.begin_execution(state, "read_text_file", {"path": "a.py"})
            self.assertEqual(first.receipt_id, repeated.receipt_id)
            ledger.complete_execution(
                state,
                first,
                ToolOutcome.success("read", evidence_ids=("e1",)),
            )
            self.assertEqual(ledger.uncertain_receipts(state.run_id), [])
            event_types = [event.event_type for event in ledger.events(state.run_id)]
            self.assertEqual(event_types.count("tool.started"), 1)
            self.assertEqual(event_types.count("tool.completed"), 1)

    def test_artifacts_are_content_addressed_and_reads_are_bounded(self):
        with TemporaryDirectory() as directory:
            ledger = RunLedger(Path(directory) / "state.sqlite3", "session")
            first = ledger.store_artifact("abcdefgh", run_id=None, origin="test")
            second = ledger.store_artifact("abcdefgh", run_id=None, origin="duplicate")
            self.assertEqual(first, second)
            excerpt = ledger.read_artifact(first, offset=2, limit=3)
            self.assertEqual(excerpt["content"], "cde")
            self.assertTrue(excerpt["truncated"])

    def test_sensitive_artifacts_encrypt_when_key_is_configured(self):
        try:
            from cryptography.fernet import Fernet
        except ImportError:
            self.skipTest("cryptography optional dependency unavailable")
        with TemporaryDirectory() as directory:
            path = Path(directory) / "state.sqlite3"
            ledger = RunLedger(
                path, "session", artifact_encryption_key=Fernet.generate_key()
            )
            artifact_id = ledger.store_artifact(
                "private artifact payload",
                run_id=None,
                origin="test",
                sensitivity="sensitive",
            )
            recovered = ledger.read_artifact(artifact_id)
            self.assertEqual(recovered["content"], "private artifact payload")
            self.assertTrue(recovered["encrypted"])
            raw_database = path.read_bytes()
            wal = path.with_name(path.name + "-wal")
            if wal.exists():
                raw_database += wal.read_bytes()
            self.assertNotIn(b"private artifact payload", raw_database)

    def test_ledger_redacts_secrets_before_persistence(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "state.sqlite3"
            ledger = RunLedger(path, "session")
            state = ledger.begin_run(self._state())
            event = ledger.append_event(
                "tool.proposed",
                state,
                {"api_key": "sk_very_secret_value", "max_tokens": 10},
            )
            self.assertEqual(event.payload["api_key"], "[redacted]")
            self.assertEqual(event.payload["max_tokens"], 10)
            self.assertNotIn(b"sk_very_secret_value", path.read_bytes())

    def test_memory_supersession_and_deletion_preserve_lineage(self):
        with TemporaryDirectory() as directory:
            ledger = RunLedger(Path(directory) / "state.sqlite3", "session")
            first = ledger.put_memory(
                MemoryRecord(
                    namespace="project",
                    scope="workspace",
                    content="Use Python 3.12",
                    provenance="user",
                    trust=TrustClass.USER_CONFIRMED,
                )
            )
            second = ledger.put_memory(
                MemoryRecord(
                    namespace="project",
                    scope="workspace",
                    content="Use Python 3.13",
                    provenance="user correction",
                    trust=TrustClass.USER_CONFIRMED,
                    supersedes_id=first.memory_id,
                )
            )
            active = ledger.list_memory(namespace="project", scope="workspace")
            self.assertEqual([item.memory_id for item in active], [second.memory_id])
            self.assertTrue(ledger.delete_memory(second.memory_id))
            self.assertEqual(ledger.list_memory(namespace="project"), [])

    def test_memory_search_returns_active_ranked_records(self):
        with TemporaryDirectory() as directory:
            ledger = RunLedger(Path(directory) / "state.sqlite3", "session")
            relevant = ledger.put_memory(
                MemoryRecord(
                    namespace="project",
                    scope="workspace",
                    content="Use Python 3.12 for package builds.",
                    provenance="user",
                    trust=TrustClass.USER_CONFIRMED,
                )
            )
            ledger.put_memory(
                MemoryRecord(
                    namespace="project",
                    scope="workspace",
                    content="The release checklist needs a changelog.",
                    provenance="user",
                    trust=TrustClass.USER_CONFIRMED,
                )
            )

            found = ledger.search_memory("python package", scope="workspace")

        self.assertEqual([record.memory_id for record in found], [relevant.memory_id])

    def test_writer_lease_prevents_concurrent_session_mutation(self):
        with TemporaryDirectory() as directory:
            ledger = RunLedger(Path(directory) / "state.sqlite3", "session")
            self.assertTrue(ledger.acquire_lease("run_a", "owner_a"))
            self.assertFalse(ledger.acquire_lease("run_b", "owner_b"))
            self.assertTrue(ledger.release_lease("run_a", "owner_a"))
            self.assertTrue(ledger.acquire_lease("run_b", "owner_b"))


class ReliabilityAndRecoveryTests(TestCase):
    def test_transport_retry_budget_is_separate_and_bounded(self):
        controller = ProviderReliabilityController(
            TransportRetryPolicy(
                max_attempts=3,
                base_delay_seconds=0,
                max_delay_seconds=0,
                jitter_seconds=0,
            )
        )
        attempts = []

        def operation():
            attempts.append(True)
            if len(attempts) < 3:
                raise URLError("temporary")
            return "ok"

        self.assertEqual(controller.call(operation), "ok")
        self.assertEqual(len(attempts), 3)
        self.assertEqual(controller.status()["consecutive_failures"], 0)

    def test_uncertain_local_write_is_reconciled_without_reexecution(self):
        class Engine:
            current_mode = "test"
            MODELS = {"test": {"path": "test-model"}}

        with TemporaryDirectory() as directory:
            root = Path(directory)
            db = root / "state.sqlite3"
            ledger = RunLedger(db, "session")
            state = ledger.begin_run(
                RunState(
                    run_id=new_id("run"),
                    session_id="session",
                    turn_id=new_id("turn"),
                    step_id="step_1",
                    goal="write recovered file",
                )
            )
            receipt = ledger.begin_execution(
                state,
                "write_text_file",
                {"path": "recovered.txt", "content": "already applied"},
            )
            (root / "recovered.txt").write_text("already applied", encoding="utf-8")

            runtime = PydanticAgentRuntime(
                Engine(),
                workspace=root,
                config=RuntimeConfig(
                    state_db_path=db,
                    session_id="session",
                    react_strict_control=True,
                ),
            )
            result = runtime.reconcile_run(state.run_id)

            self.assertEqual(result["resolved"], [receipt.receipt_id])
            self.assertEqual(result["uncertain"], [])
            self.assertEqual(
                (root / "recovered.txt").read_text(encoding="utf-8"),
                "already applied",
            )

    def test_runtime_records_complete_evidence_backed_mutation_run(self):
        class Engine:
            current_mode = "test"
            MODELS = {"test": {"path": "test-model"}}

            def __init__(self):
                self.calls = 0

            def generate_runtime_stream(self, _prompt, **_kwargs):
                responses = [
                    '<tool_call>{"name":"react_dispatch","arguments":'
                    '{"decision":"act","summary":"write requested",'
                    '"goal":"Create result.txt"}}</tool_call>',
                    '<tool_call>{"name":"write_text_file","arguments":'
                    '{"path":"result.txt","content":"done"}}</tool_call>',
                    '<tool_call>{"name":"critique_and_plan","arguments":'
                    '{"progress":"verified write","evidence":["host receipt"],'
                    '"complete":true}}</tool_call>',
                    "Created and verified result.txt.",
                ]
                response = responses[self.calls]
                self.calls += 1
                yield {"type": "token", "content": response}

        with TemporaryDirectory() as directory:
            root = Path(directory)
            db = root / "state.sqlite3"
            runtime = PydanticAgentRuntime(
                Engine(),
                workspace=root,
                config=RuntimeConfig(
                    state_db_path=db,
                    session_id="session",
                    react_strict_control=True,
                ),
            )
            events = list(runtime.generate_stream("Create result.txt containing done"))
            run_id = runtime._run_state.run_id
            ledger_events = runtime._state.ledger.events(run_id)

            self.assertEqual((root / "result.txt").read_text(encoding="utf-8"), "done")
            self.assertEqual(runtime._run_state.lifecycle, RunLifecycle.COMPLETED)
            self.assertEqual(runtime._state.ledger.uncertain_receipts(run_id), [])
            self.assertIn(
                "tool.completed", [event.event_type for event in ledger_events]
            )
            completed = next(
                event for event in ledger_events if event.event_type == "tool.completed"
            )
            self.assertTrue(completed.payload["outcome"]["changed"])
            self.assertTrue(completed.payload["outcome"]["receipt"]["verified"])
            self.assertEqual(events[-1]["type"], "done")

    def test_paused_run_resumes_on_next_message_with_same_run_id(self):
        class Engine:
            current_mode = "test"
            MODELS = {"test": {"path": "test-model"}}

            def __init__(self):
                self.calls = 0

            def generate_runtime_stream(self, _prompt, **_kwargs):
                responses = [
                    '<tool_call>{"name":"react_dispatch","arguments":'
                    '{"decision":"answer","summary":"resume acknowledged"}}</tool_call>',
                    "Resumed answer.",
                ]
                response = responses[self.calls]
                self.calls += 1
                yield {"type": "token", "content": response}

        with TemporaryDirectory() as directory:
            root = Path(directory)
            db = root / "state.sqlite3"
            ledger = RunLedger(db, "session")
            original = ledger.begin_run(
                RunState(
                    run_id=new_id("run"),
                    session_id="session",
                    turn_id=new_id("turn"),
                    goal="resume this mission",
                )
            )
            paused = ledger.transition(
                original, RunLifecycle.WAITING_USER, reason="Need user input"
            )
            runtime = PydanticAgentRuntime(
                Engine(),
                workspace=root,
                config=RuntimeConfig(state_db_path=db, session_id="session"),
            )
            self.assertTrue(runtime.prepare_resume(paused.run_id)["ready"])
            list(runtime.generate_stream("Continue with the supplied answer"))

            self.assertEqual(runtime._run_state.run_id, paused.run_id)
            self.assertEqual(runtime._run_state.lifecycle, RunLifecycle.COMPLETED)
            types = [
                event.event_type
                for event in runtime._state.ledger.events(paused.run_id)
            ]
            self.assertIn("run.resumed", types)
            self.assertIn("run.finished", types)

    def test_resume_rejects_provider_or_model_switch(self):
        class Client:
            provider = "ds2api"
            model = "deepseek/model"

        class Engine:
            backend = "remote_api"
            current_mode = "api"
            api_client = Client()
            MODELS = {"api": {"path": api_client.model}}

        with TemporaryDirectory() as directory:
            root = Path(directory)
            db = root / "state.sqlite3"
            ledger = RunLedger(db, "session")
            original = ledger.begin_run(
                RunState(
                    run_id=new_id("run"),
                    session_id="session",
                    turn_id=new_id("turn"),
                    goal="resume pinned mission",
                ),
                provider="litellm",
                model="claude/model",
            )
            paused = ledger.transition(
                original, RunLifecycle.WAITING_USER, reason="Need user input"
            )
            runtime = PydanticAgentRuntime(
                Engine(),
                workspace=root,
                config=RuntimeConfig(state_db_path=db, session_id="session"),
            )

            result = runtime.prepare_resume(paused.run_id)

            self.assertFalse(result["ready"])
            self.assertIn("pinned to litellm:claude/model", result["error"])


class WorkspaceRaceSafetyTests(TestCase):
    def test_mutation_path_rejects_symlink_components(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            real = root / "real"
            real.mkdir()
            link = root / "link"
            try:
                link.symlink_to(real, target_is_directory=True)
            except OSError:
                self.skipTest("Symlink creation requires platform permission")
            context = WorkspaceContext(root)
            with self.assertRaisesRegex(ValueError, "symbolic links"):
                context.resolve_mutation("link/file.txt")

    def test_parallel_reads_return_deterministic_order_and_reject_writes(self):
        executor = DeterministicReadBatchExecutor(default_tool_registry())
        results = executor.execute(
            [
                ("file_info", {"path": "b.py"}),
                ("file_info", {"path": "a.py"}),
            ],
            lambda _name, arguments: arguments["path"],
        )
        self.assertEqual([item["result"] for item in results], ["b.py", "a.py"])
        with self.assertRaisesRegex(ValueError, "repeat-safe reads"):
            executor.execute(
                [("write_text_file", {"path": "a.py"})],
                lambda *_args: None,
            )
