"""Durable append-only execution ledger and snapshot projector.

SQLite is deliberately kept behind this boundary.  Existing conversation and
tool-event tables remain readable; new runs use immutable events, typed
snapshots, receipts, artifacts, and memory records.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
from contextlib import closing, contextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional

from .harness_contracts import (
    ExecutionReceipt,
    LedgerEvent,
    MemoryRecord,
    RunLifecycle,
    RunState,
    SecretRedactor,
    ToolOutcome,
    content_hash,
    utc_now,
)


_TRANSITIONS = {
    RunLifecycle.PENDING: {
        RunLifecycle.RUNNING,
        RunLifecycle.CANCELLED,
        RunLifecycle.FAILED,
    },
    RunLifecycle.RUNNING: {
        RunLifecycle.WAITING_APPROVAL,
        RunLifecycle.WAITING_USER,
        RunLifecycle.CANCELLING,
        RunLifecycle.RECOVERING,
        RunLifecycle.COMPLETED,
        RunLifecycle.FAILED,
    },
    RunLifecycle.WAITING_APPROVAL: {
        RunLifecycle.RUNNING,
        RunLifecycle.CANCELLING,
        RunLifecycle.CANCELLED,
        RunLifecycle.FAILED,
    },
    RunLifecycle.WAITING_USER: {
        RunLifecycle.RUNNING,
        RunLifecycle.CANCELLING,
        RunLifecycle.CANCELLED,
        RunLifecycle.FAILED,
    },
    RunLifecycle.CANCELLING: {RunLifecycle.CANCELLED, RunLifecycle.FAILED},
    RunLifecycle.RECOVERING: {
        RunLifecycle.RUNNING,
        RunLifecycle.WAITING_USER,
        RunLifecycle.CANCELLED,
        RunLifecycle.FAILED,
    },
    RunLifecycle.CANCELLED: set(),
    RunLifecycle.COMPLETED: set(),
    RunLifecycle.FAILED: set(),
}


def _json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":")
    )


class RunLedger:
    """Transactional durable state for one FenrirAgent session."""

    def __init__(
        self,
        path: Path,
        session_id: str,
        *,
        artifact_encryption_key: Optional[bytes] = None,
    ):
        self.path = path.resolve()
        self.session_id = str(session_id)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.RLock()
        self._artifact_cipher: Any = None
        if artifact_encryption_key is not None:
            try:
                from cryptography.fernet import Fernet
            except ImportError as error:
                raise RuntimeError(
                    "Artifact encryption requires the cryptography package"
                ) from error
            self._artifact_cipher = Fernet(artifact_encryption_key)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=10, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=10000")
        return connection

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._write_lock, closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except Exception:
                connection.rollback()
                raise
            else:
                connection.commit()

    def _initialize(self) -> None:
        with self._transaction() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS harness_schema (
                    component TEXT PRIMARY KEY,
                    version INTEGER NOT NULL,
                    migrated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL,
                    lifecycle TEXT NOT NULL,
                    goal TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS runs_session_lifecycle
                    ON runs(session_id, lifecycle, updated_at);
                CREATE TABLE IF NOT EXISTS run_events (
                    event_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL,
                    step_id TEXT,
                    sequence INTEGER NOT NULL,
                    parent_event_id TEXT,
                    event_type TEXT NOT NULL,
                    event_json TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(run_id, sequence),
                    FOREIGN KEY(run_id) REFERENCES runs(run_id)
                );
                CREATE INDEX IF NOT EXISTS run_events_run_sequence
                    ON run_events(run_id, sequence);
                CREATE TABLE IF NOT EXISTS execution_receipts (
                    receipt_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    step_id TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    status TEXT NOT NULL,
                    receipt_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    UNIQUE(run_id, idempotency_key),
                    FOREIGN KEY(run_id) REFERENCES runs(run_id)
                );
                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    run_id TEXT,
                    media_type TEXT NOT NULL,
                    byte_size INTEGER NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    origin TEXT NOT NULL,
                    creating_event_id TEXT,
                    completeness TEXT NOT NULL,
                    sensitivity TEXT NOT NULL,
                    redacted INTEGER NOT NULL DEFAULT 0,
                    encrypted INTEGER NOT NULL DEFAULT 0,
                    content BLOB NOT NULL,
                    created_at TEXT NOT NULL,
                    deleted_at TEXT
                );
                CREATE INDEX IF NOT EXISTS artifacts_session_created
                    ON artifacts(session_id, created_at);
                CREATE TABLE IF NOT EXISTS memory_records_v2 (
                    memory_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    namespace TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    trust TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT,
                    superseded_by_id TEXT,
                    deleted_at TEXT
                );
                CREATE INDEX IF NOT EXISTS memory_scope_active
                    ON memory_records_v2(session_id, namespace, scope, deleted_at);
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_search_v1
                    USING fts5(
                        memory_id UNINDEXED,
                        session_id UNINDEXED,
                        namespace UNINDEXED,
                        scope UNINDEXED,
                        content,
                        tokenize='unicode61'
                    );
                CREATE TABLE IF NOT EXISTS workspace_leases (
                    session_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS provider_capabilities (
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    capability_json TEXT NOT NULL,
                    checked_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    PRIMARY KEY(provider, model)
                );
                """
            )
            connection.execute(
                """
                INSERT INTO harness_schema(component, version, migrated_at)
                VALUES ('enterprise_harness', 1, ?)
                ON CONFLICT(component) DO UPDATE SET
                    version = MAX(version, excluded.version),
                    migrated_at = excluded.migrated_at
                """,
                (utc_now().isoformat(),),
            )
            artifact_columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(artifacts)").fetchall()
            }
            if "encrypted" not in artifact_columns:
                connection.execute(
                    "ALTER TABLE artifacts ADD COLUMN encrypted INTEGER NOT NULL DEFAULT 0"
                )

    @staticmethod
    def _event_row(event: LedgerEvent) -> tuple[Any, ...]:
        return (
            event.event_id,
            event.run_id,
            event.session_id,
            event.turn_id,
            event.step_id,
            event.sequence,
            event.parent_event_id,
            event.event_type,
            event.model_dump_json(),
            event.payload_hash,
            event.occurred_at.isoformat(),
        )

    @staticmethod
    def _insert_event(connection: sqlite3.Connection, event: LedgerEvent) -> None:
        connection.execute(
            """
            INSERT INTO run_events(
                event_id, run_id, session_id, turn_id, step_id, sequence,
                parent_event_id, event_type, event_json, payload_hash, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            RunLedger._event_row(event),
        )

    @staticmethod
    def _next_sequence(connection: sqlite3.Connection, run_id: str) -> int:
        row = connection.execute(
            "SELECT COALESCE(MAX(sequence), 0) + 1 AS value FROM run_events WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        return int(row["value"])

    def begin_run(
        self, state: RunState, *, provider: str = "", model: str = ""
    ) -> RunState:
        if state.session_id != self.session_id:
            raise ValueError("Run session does not match ledger session")
        running = state.model_copy(
            update={"lifecycle": RunLifecycle.RUNNING, "updated_at": utc_now()}
        )
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO runs(
                    run_id, session_id, turn_id, lifecycle, goal, state_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    running.run_id,
                    running.session_id,
                    running.turn_id,
                    running.lifecycle.value,
                    running.goal,
                    running.model_dump_json(),
                    running.created_at.isoformat(),
                    running.updated_at.isoformat(),
                ),
            )
            event = LedgerEvent(
                event_type="run.started",
                run_id=running.run_id,
                session_id=running.session_id,
                turn_id=running.turn_id,
                sequence=1,
                provider=provider,
                model=model,
                payload={
                    "goal_hash": content_hash(running.goal),
                    "budgets": running.budgets.model_dump(mode="json"),
                },
            ).with_hash()
            self._insert_event(connection, event)
        return running

    def append_event(
        self,
        event_type: str,
        state: RunState,
        payload: Optional[Mapping[str, Any]] = None,
        *,
        provider: str = "",
        model: str = "",
        parent_event_id: Optional[str] = None,
        evidence_ids: Iterable[str] = (),
        artifact_ids: Iterable[str] = (),
        policy_result: Optional[Mapping[str, Any]] = None,
    ) -> LedgerEvent:
        cleaned, redactions = SecretRedactor.redact_mapping(dict(payload or {}))
        with self._transaction() as connection:
            sequence = self._next_sequence(connection, state.run_id)
            event = LedgerEvent(
                event_type=event_type,
                run_id=state.run_id,
                session_id=state.session_id,
                turn_id=state.turn_id,
                step_id=state.step_id,
                sequence=sequence,
                parent_event_id=parent_event_id,
                provider=provider,
                model=model,
                payload=cleaned,
                policy_result=dict(policy_result or {}),
                redactions=redactions,
                evidence_ids=tuple(evidence_ids),
                artifact_ids=tuple(artifact_ids),
            ).with_hash()
            self._insert_event(connection, event)
        return event

    def save_snapshot(self, state: RunState) -> None:
        updated = state.model_copy(update={"updated_at": utc_now()})
        with self._transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE runs SET lifecycle = ?, turn_id = ?, goal = ?, state_json = ?, updated_at = ?
                WHERE run_id = ? AND session_id = ?
                """,
                (
                    updated.lifecycle.value,
                    updated.turn_id,
                    updated.goal,
                    updated.model_dump_json(),
                    updated.updated_at.isoformat(),
                    updated.run_id,
                    self.session_id,
                ),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Unknown run: {state.run_id}")

    def transition(
        self,
        state: RunState,
        lifecycle: RunLifecycle,
        *,
        reason: str = "",
    ) -> RunState:
        current = RunLifecycle(state.lifecycle)
        if lifecycle == current:
            return state
        if lifecycle not in _TRANSITIONS[current]:
            raise ValueError(
                f"Invalid run transition: {current.value} -> {lifecycle.value}"
            )
        updated = state.model_copy(
            update={
                "lifecycle": lifecycle,
                "stop_reason": reason
                if lifecycle.terminal or lifecycle == RunLifecycle.WAITING_USER
                else state.stop_reason,
                "updated_at": utc_now(),
            }
        )
        event_type = {
            RunLifecycle.WAITING_APPROVAL: "approval.requested",
            RunLifecycle.WAITING_USER: "run.paused",
            RunLifecycle.CANCELLING: "run.cancelling",
            RunLifecycle.CANCELLED: "run.cancelled",
            RunLifecycle.RECOVERING: "run.recovering",
            RunLifecycle.RUNNING: "run.resumed",
            RunLifecycle.COMPLETED: "run.finished",
            RunLifecycle.FAILED: "run.failed",
        }.get(lifecycle, "state.transitioned")
        with self._transaction() as connection:
            sequence = self._next_sequence(connection, state.run_id)
            event = LedgerEvent(
                event_type=event_type,
                run_id=state.run_id,
                session_id=state.session_id,
                turn_id=state.turn_id,
                step_id=state.step_id,
                sequence=sequence,
                payload={
                    "from": current.value,
                    "to": lifecycle.value,
                    "reason": reason,
                },
            ).with_hash()
            self._insert_event(connection, event)
            connection.execute(
                "UPDATE runs SET lifecycle = ?, turn_id = ?, state_json = ?, updated_at = ? WHERE run_id = ?",
                (
                    lifecycle.value,
                    updated.turn_id,
                    updated.model_dump_json(),
                    updated.updated_at.isoformat(),
                    state.run_id,
                ),
            )
        return updated

    def load_state(self, run_id: str) -> Optional[RunState]:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT state_json FROM runs WHERE run_id = ? AND session_id = ?",
                (run_id, self.session_id),
            ).fetchone()
        return RunState.model_validate_json(row["state_json"]) if row else None

    def events(self, run_id: str, *, after_sequence: int = 0) -> list[LedgerEvent]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT event_json FROM run_events
                WHERE run_id = ? AND session_id = ? AND sequence > ?
                ORDER BY sequence
                """,
                (run_id, self.session_id, max(0, int(after_sequence))),
            ).fetchall()
        return [LedgerEvent.model_validate_json(row["event_json"]) for row in rows]

    def incomplete_runs(self) -> list[RunState]:
        terminal = tuple(item.value for item in RunLifecycle if item.terminal)
        placeholders = ",".join("?" for _ in terminal)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                f"""
                SELECT state_json FROM runs
                WHERE session_id = ? AND lifecycle NOT IN ({placeholders})
                ORDER BY updated_at DESC
                """,
                (self.session_id, *terminal),
            ).fetchall()
        return [RunState.model_validate_json(row["state_json"]) for row in rows]

    def mark_abandoned_recovering(self) -> list[RunState]:
        active_run_ids: set[str] = set()
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT run_id FROM workspace_leases WHERE session_id = ? AND expires_at > ?",
                (self.session_id, utc_now().isoformat()),
            ).fetchall()
            active_run_ids = {str(row["run_id"]) for row in rows}
        recovered: list[RunState] = []
        for state in self.incomplete_runs():
            if state.run_id in active_run_ids:
                continue
            if state.lifecycle == RunLifecycle.RUNNING:
                recovered.append(
                    self.transition(
                        state,
                        RunLifecycle.RECOVERING,
                        reason="Process ended before terminal commit",
                    )
                )
            elif state.lifecycle == RunLifecycle.CANCELLING:
                self.transition(
                    state,
                    RunLifecycle.CANCELLED,
                    reason="Cancellation was in progress when the process ended",
                )
        return recovered

    def acquire_lease(
        self,
        run_id: str,
        owner_id: str,
        *,
        ttl_seconds: int = 120,
    ) -> bool:
        now = utc_now()
        expires = now + timedelta(seconds=max(30, int(ttl_seconds)))
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT run_id, owner_id, expires_at FROM workspace_leases WHERE session_id = ?",
                (self.session_id,),
            ).fetchone()
            if (
                row
                and str(row["expires_at"]) > now.isoformat()
                and row["owner_id"] != owner_id
            ):
                return False
            connection.execute(
                """
                INSERT INTO workspace_leases(session_id, run_id, owner_id, expires_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    run_id = excluded.run_id,
                    owner_id = excluded.owner_id,
                    expires_at = excluded.expires_at
                """,
                (self.session_id, run_id, owner_id, expires.isoformat()),
            )
        return True

    def renew_lease(
        self, run_id: str, owner_id: str, *, ttl_seconds: int = 120
    ) -> bool:
        expires = utc_now() + timedelta(seconds=max(30, int(ttl_seconds)))
        with self._transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE workspace_leases SET expires_at = ?
                WHERE session_id = ? AND run_id = ? AND owner_id = ?
                """,
                (expires.isoformat(), self.session_id, run_id, owner_id),
            )
        return cursor.rowcount == 1

    def release_lease(self, run_id: str, owner_id: str) -> bool:
        with self._transaction() as connection:
            cursor = connection.execute(
                """
                DELETE FROM workspace_leases
                WHERE session_id = ? AND run_id = ? AND owner_id = ?
                """,
                (self.session_id, run_id, owner_id),
            )
        return cursor.rowcount == 1

    def begin_execution(
        self,
        state: RunState,
        tool_name: str,
        arguments: Mapping[str, Any],
        *,
        idempotency_key: Optional[str] = None,
    ) -> ExecutionReceipt:
        if not state.step_id:
            raise ValueError("A tool execution requires a step ID")
        key = idempotency_key or content_hash(
            {
                "run_id": state.run_id,
                "step_id": state.step_id,
                "tool": tool_name,
                "arguments": arguments,
            }
        )
        receipt = ExecutionReceipt(
            run_id=state.run_id,
            step_id=state.step_id,
            tool_name=tool_name,
            idempotency_key=key,
        )
        cleaned, redactions = SecretRedactor.redact_mapping(dict(arguments))
        with self._transaction() as connection:
            prior = connection.execute(
                "SELECT receipt_json FROM execution_receipts WHERE run_id = ? AND idempotency_key = ?",
                (state.run_id, key),
            ).fetchone()
            if prior:
                return ExecutionReceipt.model_validate_json(prior["receipt_json"])
            connection.execute(
                """
                INSERT INTO execution_receipts(
                    receipt_id, run_id, step_id, tool_name, idempotency_key,
                    status, receipt_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    receipt.receipt_id,
                    receipt.run_id,
                    receipt.step_id,
                    receipt.tool_name,
                    receipt.idempotency_key,
                    receipt.status,
                    receipt.model_dump_json(),
                    receipt.created_at.isoformat(),
                ),
            )
            event = LedgerEvent(
                event_type="tool.started",
                run_id=state.run_id,
                session_id=state.session_id,
                turn_id=state.turn_id,
                step_id=state.step_id,
                sequence=self._next_sequence(connection, state.run_id),
                payload={
                    "tool": tool_name,
                    "arguments": cleaned,
                    "receipt_id": receipt.receipt_id,
                },
                redactions=redactions,
            ).with_hash()
            self._insert_event(connection, event)
        return receipt

    def complete_execution(
        self,
        state: RunState,
        receipt: ExecutionReceipt,
        outcome: ToolOutcome,
    ) -> ExecutionReceipt:
        completed = receipt.model_copy(
            update={
                "status": outcome.status.value,
                "effect_hash": content_hash(outcome.receipt)
                if outcome.receipt
                else None,
                "reconciliation": dict(outcome.receipt),
                "completed_at": utc_now(),
            }
        )
        with self._transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE execution_receipts
                SET status = ?, receipt_json = ?, completed_at = ?
                WHERE receipt_id = ? AND run_id = ?
                """,
                (
                    completed.status,
                    completed.model_dump_json(),
                    completed.completed_at.isoformat()
                    if completed.completed_at
                    else None,
                    completed.receipt_id,
                    state.run_id,
                ),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Unknown execution receipt: {receipt.receipt_id}")
            event = LedgerEvent(
                event_type="tool.completed",
                run_id=state.run_id,
                session_id=state.session_id,
                turn_id=state.turn_id,
                step_id=state.step_id,
                sequence=self._next_sequence(connection, state.run_id),
                payload={
                    "tool": receipt.tool_name,
                    "receipt_id": receipt.receipt_id,
                    "outcome": outcome.model_dump(mode="json"),
                },
                evidence_ids=outcome.evidence_ids,
                artifact_ids=outcome.artifact_ids,
            ).with_hash()
            self._insert_event(connection, event)
        return completed

    def uncertain_receipts(self, run_id: str) -> list[ExecutionReceipt]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT receipt_json FROM execution_receipts
                WHERE run_id = ? AND status = 'started' ORDER BY created_at
                """,
                (run_id,),
            ).fetchall()
        return [
            ExecutionReceipt.model_validate_json(row["receipt_json"]) for row in rows
        ]

    def store_artifact(
        self,
        content: bytes | str,
        *,
        run_id: Optional[str],
        media_type: str = "text/plain",
        origin: str,
        creating_event_id: Optional[str] = None,
        completeness: str = "full",
        sensitivity: str = "normal",
        redacted: bool = False,
    ) -> str:
        raw = content.encode("utf-8") if isinstance(content, str) else bytes(content)
        digest = hashlib.sha256(raw).hexdigest()
        artifact_id = f"artifact_sha256_{digest}"
        encrypted = self._artifact_cipher is not None and sensitivity != "normal"
        stored = self._artifact_cipher.encrypt(raw) if encrypted else raw
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO artifacts(
                    artifact_id, session_id, run_id, media_type, byte_size,
                    content_sha256, origin, creating_event_id, completeness,
                    sensitivity, redacted, content, created_at
                    , encrypted
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact_id,
                    self.session_id,
                    run_id,
                    media_type,
                    len(raw),
                    digest,
                    origin,
                    creating_event_id,
                    completeness,
                    sensitivity,
                    int(redacted),
                    stored,
                    utc_now().isoformat(),
                    int(encrypted),
                ),
            )
        return artifact_id

    def read_artifact(
        self, artifact_id: str, *, offset: int = 0, limit: int = 20_000
    ) -> Dict[str, Any]:
        offset, limit = max(0, int(offset)), max(1, min(int(limit), 100_000))
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT media_type, byte_size, content_sha256, completeness,
                       sensitivity, redacted, encrypted, content
                FROM artifacts
                WHERE artifact_id = ? AND session_id = ? AND deleted_at IS NULL
                """,
                (artifact_id, self.session_id),
            ).fetchone()
        if not row:
            raise KeyError(f"Unknown artifact: {artifact_id}")
        content = bytes(row["content"])
        if bool(row["encrypted"]):
            if self._artifact_cipher is None:
                raise PermissionError("Artifact is encrypted and no key is configured")
            content = bytes(self._artifact_cipher.decrypt(content))
        excerpt = content[offset : offset + limit]
        return {
            "artifact_id": artifact_id,
            "media_type": row["media_type"],
            "byte_size": row["byte_size"],
            "sha256": row["content_sha256"],
            "completeness": row["completeness"],
            "sensitivity": row["sensitivity"],
            "redacted": bool(row["redacted"]),
            "encrypted": bool(row["encrypted"]),
            "offset": offset,
            "content": excerpt.decode("utf-8", errors="replace"),
            "truncated": offset + len(excerpt) < len(content),
        }

    def put_memory(self, record: MemoryRecord) -> MemoryRecord:
        with self._transaction() as connection:
            if record.supersedes_id:
                prior = connection.execute(
                    """
                    SELECT record_json FROM memory_records_v2
                    WHERE memory_id = ? AND session_id = ? AND deleted_at IS NULL
                    """,
                    (record.supersedes_id, self.session_id),
                ).fetchone()
                if not prior:
                    raise KeyError(
                        f"Unknown memory to supersede: {record.supersedes_id}"
                    )
                connection.execute(
                    "UPDATE memory_records_v2 SET superseded_by_id = ? WHERE memory_id = ?",
                    (record.memory_id, record.supersedes_id),
                )
            connection.execute(
                """
                INSERT INTO memory_records_v2(
                    memory_id, session_id, namespace, scope, trust, record_json,
                    created_at, expires_at, superseded_by_id, deleted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.memory_id,
                    self.session_id,
                    record.namespace,
                    record.scope,
                    record.trust.value,
                    record.model_dump_json(),
                    record.created_at.isoformat(),
                    record.expires_at.isoformat() if record.expires_at else None,
                    record.superseded_by_id,
                    record.deleted_at.isoformat() if record.deleted_at else None,
                ),
            )
            connection.execute(
                """
                INSERT INTO memory_search_v1(
                    memory_id, session_id, namespace, scope, content
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    record.memory_id,
                    self.session_id,
                    record.namespace,
                    record.scope,
                    record.content,
                ),
            )
        return record

    def list_memory(
        self, *, namespace: Optional[str] = None, scope: Optional[str] = None
    ) -> list[MemoryRecord]:
        clauses = ["session_id = ?", "deleted_at IS NULL", "superseded_by_id IS NULL"]
        values: list[Any] = [self.session_id]
        if namespace:
            clauses.append("namespace = ?")
            values.append(namespace)
        if scope:
            clauses.append("scope = ?")
            values.append(scope)
        clauses.append("(expires_at IS NULL OR expires_at > ?)")
        values.append(utc_now().isoformat())
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT record_json FROM memory_records_v2 WHERE "
                + " AND ".join(clauses)
                + " ORDER BY created_at",
                values,
            ).fetchall()
        return [MemoryRecord.model_validate_json(row["record_json"]) for row in rows]

    def search_memory(
        self,
        query: str,
        *,
        limit: int = 5,
        namespace: Optional[str] = None,
        scope: Optional[str] = None,
    ) -> list[MemoryRecord]:
        """Return active memories ranked by local FTS5 relevance.

        Search terms are host-normalized before reaching SQLite so user text
        cannot alter the FTS query grammar.  Historical or deleted records stay
        in the index for lineage, but never appear in recall results.
        """
        terms = re.findall(r"[\w-]{2,}", str(query).casefold(), flags=re.UNICODE)
        if not terms:
            return []
        match = " OR ".join(f'"{term.replace(chr(34), "")}"' for term in terms[:16])
        clauses = [
            "memory_search_v1 MATCH ?",
            "records.session_id = ?",
            "records.deleted_at IS NULL",
            "records.superseded_by_id IS NULL",
            "(records.expires_at IS NULL OR records.expires_at > ?)",
        ]
        values: list[Any] = [match, self.session_id, utc_now().isoformat()]
        if namespace:
            clauses.append("records.namespace = ?")
            values.append(namespace)
        if scope:
            clauses.append("records.scope = ?")
            values.append(scope)
        values.append(max(1, min(int(limit), 20)))
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT records.record_json
                FROM memory_search_v1
                JOIN memory_records_v2 AS records
                    ON records.memory_id = memory_search_v1.memory_id
                WHERE """
                + " AND ".join(clauses)
                + " ORDER BY bm25(memory_search_v1), records.created_at DESC LIMIT ?",
                values,
            ).fetchall()
        return [MemoryRecord.model_validate_json(row["record_json"]) for row in rows]

    def delete_memory(self, memory_id: str) -> bool:
        deleted_at = utc_now()
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT record_json FROM memory_records_v2 WHERE memory_id = ? AND session_id = ? AND deleted_at IS NULL",
                (memory_id, self.session_id),
            ).fetchone()
            if not row:
                return False
            record = MemoryRecord.model_validate_json(row["record_json"]).model_copy(
                update={"deleted_at": deleted_at}
            )
            connection.execute(
                "UPDATE memory_records_v2 SET deleted_at = ?, record_json = ? WHERE memory_id = ?",
                (deleted_at.isoformat(), record.model_dump_json(), memory_id),
            )
        return True

    def export_debug_bundle(self, run_id: str) -> Dict[str, Any]:
        state = self.load_state(run_id)
        if state is None:
            raise KeyError(f"Unknown run: {run_id}")
        events = self.events(run_id)
        return {
            "schema_version": 1,
            "run": state.model_dump(mode="json"),
            "events": [event.model_dump(mode="json") for event in events],
            "uncertain_receipts": [
                receipt.model_dump(mode="json")
                for receipt in self.uncertain_receipts(run_id)
            ],
            "content_redacted": True,
        }

    def cache_provider_capabilities(
        self,
        provider: str,
        model: str,
        capabilities: Mapping[str, Any],
        *,
        ttl_seconds: int = 86_400,
    ) -> None:
        checked = utc_now()
        expires = checked + timedelta(seconds=max(60, int(ttl_seconds)))
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO provider_capabilities(
                    provider, model, capability_json, checked_at, expires_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(provider, model) DO UPDATE SET
                    capability_json = excluded.capability_json,
                    checked_at = excluded.checked_at,
                    expires_at = excluded.expires_at
                """,
                (
                    provider,
                    model,
                    _json(dict(capabilities)),
                    checked.isoformat(),
                    expires.isoformat(),
                ),
            )

    def provider_capabilities(
        self, provider: str, model: str
    ) -> Optional[Dict[str, Any]]:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT capability_json FROM provider_capabilities
                WHERE provider = ? AND model = ? AND expires_at > ?
                """,
                (provider, model, utc_now().isoformat()),
            ).fetchone()
        return json.loads(row["capability_json"]) if row else None


__all__ = ["RunLedger"]
