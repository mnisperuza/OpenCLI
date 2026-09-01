import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from unittest import TestCase

from main.delegation import DelegationManager


class DelegationManagerTests(TestCase):
    @staticmethod
    def _wait(manager: DelegationManager, job_id: str, timeout: float = 3.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            job = manager.get(job_id)
            if job.status in {"completed", "failed", "cancelled"}:
                return job
            time.sleep(0.01)
        raise AssertionError("delegate did not finish")

    def test_delegate_uses_secret_free_snapshot_and_cannot_change_workspace(self):
        with TemporaryDirectory() as directory:
            base = Path(directory)
            workspace = base / "workspace"
            workspace.mkdir()
            source = workspace / "app.py"
            source.write_text("original", encoding="utf-8")
            (workspace / ".env").write_text("TOKEN=secret", encoding="utf-8")

            def execute(task, snapshot, cancel):
                self.assertEqual(task, "inspect app")
                self.assertFalse(cancel.is_set())
                self.assertFalse((snapshot / ".env").exists())
                (snapshot / "app.py").write_text("delegate", encoding="utf-8")
                return {"result": "found app", "evidence_ids": ["evidence-1"]}

            manager = DelegationManager(workspace, execute, root=base / "state")
            submitted = manager.submit("inspect app")
            completed = self._wait(manager, submitted.job_id)

            self.assertEqual(completed.status, "completed")
            self.assertEqual(completed.result, "found app")
            self.assertEqual(completed.evidence_ids, ["evidence-1"])
            self.assertEqual(source.read_text(encoding="utf-8"), "original")
            self.assertFalse((manager.snapshots / submitted.job_id).exists())

    def test_stop_cancels_queued_delegate(self):
        with TemporaryDirectory() as directory:
            base = Path(directory)
            workspace = base / "workspace"
            workspace.mkdir()
            release = Event()

            def execute(task, _snapshot, cancel):
                if task == "first":
                    release.wait(2)
                return {"result": "cancelled" if cancel.is_set() else "done"}

            manager = DelegationManager(workspace, execute, root=base / "state")
            first = manager.submit("first")
            second = manager.submit("second")
            manager.stop(second.job_id)
            release.set()

            self._wait(manager, first.job_id)
            cancelled = self._wait(manager, second.job_id)
            self.assertEqual(cancelled.status, "cancelled")

    def test_restart_marks_unfinished_job_failed(self):
        with TemporaryDirectory() as directory:
            base = Path(directory)
            workspace = base / "workspace"
            workspace.mkdir()
            state = base / "state"
            digest = __import__("hashlib").sha256(
                str(workspace.resolve()).casefold().encode()
            ).hexdigest()[:12]
            job_dir = state / digest
            job_dir.mkdir(parents=True)
            (job_dir / "jobs.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "jobs": [
                            {
                                "job_id": "old-job",
                                "task": "inspect",
                                "status": "running",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            manager = DelegationManager(
                workspace, lambda *_args: {}, root=state
            )

            recovered = manager.get("old-job")
            self.assertEqual(recovered.status, "failed")
            self.assertIn("exited", recovered.error)
