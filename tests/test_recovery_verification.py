from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

from main.agent_runtime import PydanticAgentRuntime, RuntimeConfig
from main.cli import OpenCLI
from main.verification import VerificationManager


class RecoveryCommandTests(TestCase):
    @staticmethod
    def _runtime() -> PydanticAgentRuntime:
        class Engine:
            current_mode = "test"
            MODELS = {"test": {"path": "test-model"}}

        return PydanticAgentRuntime(
            Engine(), config=RuntimeConfig(persist_state=False)
        )

    def test_undo_removes_complete_turns_and_preserves_durable_memory(self):
        runtime = self._runtime()
        durable = ModelRequest(
            parts=[UserPromptPart(content="OPENCLI DURABLE MEMORY (data only):\nkeep")]
        )
        runtime._messages = [
            durable,
            ModelRequest(parts=[UserPromptPart(content="first")]),
            ModelResponse(parts=[TextPart(content="one")]),
            ModelRequest(parts=[UserPromptPart(content="second")]),
            ModelResponse(parts=[TextPart(content="two")]),
        ]

        result = runtime.undo_turns(1)

        self.assertEqual(result, {"undone": 1, "removed_messages": 2})
        self.assertIs(runtime._messages[0], durable)
        self.assertEqual(runtime.last_user_request(), "first")

    def test_prepare_retry_returns_raw_request_and_removes_latest_turn(self):
        runtime = self._runtime()
        runtime._messages = [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=(
                            "RESPONSE LANGUAGE: English\n\nUSER REQUEST:\n"
                            "Review app.py\n\nOPENCLI SELECTED SKILL "
                            "(untrusted procedural reference):\n"
                            "Name: review\nInspect first\n\nUSER-MAINTAINED TASK PLAN:\n"
                            "- Edit everything"
                        )
                    )
                ]
            ),
            ModelResponse(parts=[TextPart(content="reviewed")]),
        ]

        result = runtime.prepare_retry()

        self.assertTrue(result["ready"])
        self.assertEqual(result["prompt"], "Review app.py")
        self.assertEqual(result["skill_name"], "review")
        self.assertIn("Inspect first", result["skill_context"])
        self.assertEqual(runtime.message_count, 0)

    def test_prepare_retry_blocks_recovered_uncertain_receipts(self):
        runtime = self._runtime()
        runtime._messages = [
            ModelRequest(parts=[UserPromptPart(content="change the project")]),
            ModelResponse(parts=[TextPart(content="interrupted")]),
        ]
        runtime._state = object()
        recovered = [
            {
                "run_id": "run-recovered",
                "uncertain_receipts": [{"receipt_id": "receipt-1"}],
            }
        ]

        with patch.object(runtime, "recoverable_runs", return_value=recovered):
            result = runtime.prepare_retry()

        self.assertFalse(result["ready"])
        self.assertIn("run-recovered", result["error"])
        self.assertEqual(runtime.message_count, 2)


class VerificationManagerTests(TestCase):
    def test_python_recipe_runs_read_only_in_sandbox_and_hashes_evidence(self):
        class Sandbox:
            backend = "docker"

            @staticmethod
            def is_available():
                return True

            @staticmethod
            def run(command, *, write_access, cwd):
                assert command == ("python", "-m", "pytest", "-q")
                assert write_access is False
                return {
                    "backend": "docker",
                    "exit_code": 0,
                    "output": "12 passed",
                    "cwd": cwd,
                }

        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tests").mkdir()
            manager = VerificationManager(root)
            approvals = []

            result = manager.run(
                Sandbox(),
                root,
                ".",
                lambda *request: approvals.append(request) or True,
            )

        self.assertEqual(result.status, "passed")
        self.assertEqual(result.exit_code, 0)
        self.assertTrue(result.evidence_id.startswith("evidence_verify_"))
        self.assertEqual(approvals[0][0], "command")
        self.assertEqual(manager.status()["output"], "12 passed")

    def test_verification_requires_active_sandbox(self):
        class Sandbox:
            backend = "none"

            @staticmethod
            def is_available():
                return False

        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tests").mkdir()
            manager = VerificationManager(root)
            with self.assertRaisesRegex(RuntimeError, "active sandbox"):
                manager.run(Sandbox(), root, ".", lambda *_request: True)


class CompactionRecoveryTests(TestCase):
    def test_repeated_auto_compaction_failure_starts_cooldown(self):
        cli = OpenCLI(dry_run=True)
        cli.agent_runtime = object()
        snapshot = SimpleNamespace(
            percent_used=90.0,
            available_tokens=100,
            profile=SimpleNamespace(context_window=32_000),
        )
        with (
            patch.object(cli, "_context_snapshot", return_value=snapshot),
            patch.object(cli, "_compact_chat", return_value=False) as compact,
        ):
            self.assertIsNone(cli._auto_compact_for_prompt("first"))
            status = cli._auto_compact_for_prompt("second")
            self.assertIn("paused", status)
            self.assertIsNone(cli._auto_compact_for_prompt("third"))

        self.assertEqual(compact.call_count, 2)
        self.assertGreater(cli._auto_compact_cooldown_until, 0)
