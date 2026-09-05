"""Durable, bounded, read-only delegation jobs."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping


DelegateExecutor = Callable[[str, Path, threading.Event], Mapping[str, Any]]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class DelegateJob:
    job_id: str
    task: str
    status: str = "queued"
    created_at: str = field(default_factory=_now)
    started_at: str = ""
    completed_at: str = ""
    result: str = ""
    error: str = ""
    evidence_ids: list[str] = field(default_factory=list)
    read_only: bool = True
    max_steps: int = 6


class DelegationManager:
    """Run one background delegate at a time over a bounded workspace snapshot."""

    MAX_TASK_CHARS = 4_000
    MAX_RESULT_CHARS = 24_000
    MAX_FILES = 2_000
    MAX_FILE_BYTES = 2_000_000
    MAX_TOTAL_BYTES = 25_000_000
    KEEP_JOBS = 100
    EXCLUDED_PARTS = {
        ".git", ".fenrir", ".venv", "node_modules", "__pycache__"
    }

    def __init__(
        self,
        workspace: Path,
        executor: DelegateExecutor,
        *,
        root: Path | None = None,
    ) -> None:
        self.workspace = workspace.resolve()
        digest = hashlib.sha256(os.path.normcase(str(self.workspace)).encode()).hexdigest()[:12]
        self.root = (root or Path.home() / ".fenrir" / "delegates") / digest
        self.state_path = self.root / "jobs.json"
        self.snapshots = self.root / "snapshots"
        self.executor = executor
        self._lock = threading.RLock()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fenrir-delegate")
        self._cancel: dict[str, threading.Event] = {}
        self._jobs = self._load()
        self._recover_interrupted()

    def _load(self) -> dict[str, DelegateJob]:
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return {}
        jobs: dict[str, DelegateJob] = {}
        for value in raw.get("jobs", []) if isinstance(raw, dict) else []:
            try:
                job = DelegateJob(**value)
            except (TypeError, ValueError):
                continue
            jobs[job.job_id] = job
        return jobs

    def _save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        jobs = sorted(self._jobs.values(), key=lambda item: item.created_at)[-self.KEEP_JOBS :]
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({"version": 1, "jobs": [asdict(job) for job in jobs]}, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.state_path)

    def _recover_interrupted(self) -> None:
        changed = False
        for job in self._jobs.values():
            if job.status in {"queued", "running", "cancelling"}:
                job.status = "failed"
                job.error = "FenrirAgent exited before delegate completed."
                job.completed_at = _now()
                changed = True
        if changed:
            self._save()

    @classmethod
    def _excluded(cls, relative: str) -> bool:
        parts = PurePosixPath(relative).parts
        if any(part in cls.EXCLUDED_PARTS for part in parts):
            return True
        name = parts[-1].casefold() if parts else ""
        return name == ".env" or name.startswith(".env.") or name.endswith((".pem", ".key"))

    def _snapshot(self, job_id: str) -> Path:
        target = self.snapshots / job_id
        target.mkdir(parents=True, exist_ok=False)
        count = 0
        total = 0
        try:
            for source in sorted(self.workspace.rglob("*")):
                if not source.is_file() or source.is_symlink():
                    continue
                relative = source.relative_to(self.workspace).as_posix()
                if self._excluded(relative):
                    continue
                size = source.stat().st_size
                if size > self.MAX_FILE_BYTES:
                    continue
                count += 1
                total += size
                if count > self.MAX_FILES or total > self.MAX_TOTAL_BYTES:
                    raise ValueError("Workspace snapshot exceeds delegation limits.")
                destination = target / Path(relative)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
        except Exception:
            shutil.rmtree(target, ignore_errors=True)
            raise
        return target

    def submit(self, task: str) -> DelegateJob:
        clean = " ".join(str(task).split()).strip()
        if not clean:
            raise ValueError("Delegate task is required.")
        if len(clean) > self.MAX_TASK_CHARS:
            raise ValueError("Delegate task is too long.")
        job = DelegateJob(job_id=uuid.uuid4().hex[:12], task=clean)
        with self._lock:
            self._jobs[job.job_id] = job
            self._cancel[job.job_id] = threading.Event()
            self._save()
        self._pool.submit(self._run, job.job_id)
        return job

    def _run(self, job_id: str) -> None:
        snapshot: Path | None = None
        with self._lock:
            job = self._jobs[job_id]
            if self._cancel[job_id].is_set():
                job.status = "cancelled"
                job.completed_at = _now()
                self._save()
                return
            job.status = "running"
            job.started_at = _now()
            self._save()
        try:
            snapshot = self._snapshot(job_id)
            raw = dict(self.executor(job.task, snapshot, self._cancel[job_id]))
            with self._lock:
                if self._cancel[job_id].is_set():
                    job.status = "cancelled"
                else:
                    job.status = "completed"
                    job.result = str(raw.get("result", ""))[-self.MAX_RESULT_CHARS :]
                    job.evidence_ids = [str(item) for item in raw.get("evidence_ids", [])][:50]
        except Exception as error:
            with self._lock:
                job.status = "cancelled" if self._cancel[job_id].is_set() else "failed"
                job.error = str(error)[:2_000]
        finally:
            if snapshot is not None:
                shutil.rmtree(snapshot, ignore_errors=True)
            with self._lock:
                job.completed_at = _now()
                self._cancel.pop(job_id, None)
                self._save()

    def stop(self, job_id: str) -> DelegateJob:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(f"Unknown delegate: {job_id}")
            if job.status not in {"queued", "running", "cancelling"}:
                return job
            job.status = "cancelling"
            self._cancel[job_id].set()
            self._save()
            return job

    def get(self, job_id: str) -> DelegateJob:
        with self._lock:
            try:
                return DelegateJob(**asdict(self._jobs[job_id]))
            except KeyError as error:
                raise KeyError(f"Unknown delegate: {job_id}") from error

    def list(self) -> list[DelegateJob]:
        with self._lock:
            return [
                DelegateJob(**asdict(job))
                for job in sorted(
                    self._jobs.values(), key=lambda item: item.created_at, reverse=True
                )
            ]


__all__ = ["DelegateJob", "DelegationManager"]
