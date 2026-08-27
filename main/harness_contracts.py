"""Versioned, host-owned contracts for the OpenCLI execution harness.

The language model may propose actions, but only these contracts are allowed to
cross control, execution, evidence, and persistence boundaries.  All models are
strict and JSON serializable so ledger replay never depends on Python objects or
human prose.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterable, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


SCHEMA_VERSION = 1


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    """Return an opaque sortable-enough ID without leaking host information."""
    return f"{prefix}_{uuid.uuid4().hex}"


def content_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class ToolStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    DENIED = "denied"
    RETRYABLE_ERROR = "retryable_error"
    FATAL_ERROR = "fatal_error"
    CANCELLED = "cancelled"

    @property
    def is_failure(self) -> bool:
        return self in {
            ToolStatus.DENIED,
            ToolStatus.RETRYABLE_ERROR,
            ToolStatus.FATAL_ERROR,
            ToolStatus.CANCELLED,
        }


class ErrorCode(str, Enum):
    NONE = "none"
    INVALID_INPUT = "invalid_input"
    OUT_OF_SCOPE = "out_of_scope"
    PROTECTED_RESOURCE = "protected_resource"
    PERMISSION_DENIED = "permission_denied"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    RATE_LIMITED = "rate_limited"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    EXECUTION_FAILED = "execution_failed"
    OUTPUT_LIMIT = "output_limit"
    POLICY_BLOCKED = "policy_blocked"
    INTERNAL_ERROR = "internal_error"


class CapabilityClass(str, Enum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    NETWORK = "network"
    SENSITIVE = "sensitive"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RunLifecycle(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    WAITING_USER = "waiting_user"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    RECOVERING = "recovering"
    COMPLETED = "completed"
    FAILED = "failed"

    @property
    def terminal(self) -> bool:
        return self in {
            RunLifecycle.CANCELLED,
            RunLifecycle.COMPLETED,
            RunLifecycle.FAILED,
        }


class TrustClass(str, Enum):
    USER_CONFIRMED = "user_confirmed"
    TOOL_VERIFIED = "tool_verified"
    MODEL_INFERRED = "model_inferred"
    IMPORTED_UNTRUSTED = "imported_untrusted"


class HarnessModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=False)
    schema_version: int = SCHEMA_VERSION


class ToolOutcome(HarnessModel):
    """The only status envelope consumed by ReAct and completion checks."""

    status: ToolStatus
    summary: str = Field(default="", max_length=2_000)
    evidence_ids: tuple[str, ...] = ()
    artifact_ids: tuple[str, ...] = ()
    changed: bool = False
    receipt: Dict[str, Any] = Field(default_factory=dict)
    retry_after: Optional[float] = Field(default=None, ge=0)
    error_code: ErrorCode = ErrorCode.NONE
    details: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("summary")
    @classmethod
    def clean_summary(cls, value: str) -> str:
        return " ".join(str(value or "").split())[:2_000]

    @property
    def succeeded(self) -> bool:
        return self.status in {ToolStatus.SUCCESS, ToolStatus.PARTIAL}

    @property
    def failed(self) -> bool:
        return self.status.is_failure

    @classmethod
    def success(
        cls,
        summary: str,
        *,
        evidence_ids: Iterable[str] = (),
        artifact_ids: Iterable[str] = (),
        changed: bool = False,
        receipt: Optional[Mapping[str, Any]] = None,
        details: Optional[Mapping[str, Any]] = None,
    ) -> "ToolOutcome":
        return cls(
            status=ToolStatus.SUCCESS,
            summary=summary,
            evidence_ids=tuple(evidence_ids),
            artifact_ids=tuple(artifact_ids),
            changed=changed,
            receipt=dict(receipt or {}),
            details=dict(details or {}),
        )

    @classmethod
    def from_event(cls, event: Mapping[str, Any]) -> "ToolOutcome":
        """Normalize explicit legacy fields without interpreting summary prose."""
        embedded = event.get("outcome")
        if isinstance(embedded, cls):
            return embedded
        if isinstance(embedded, Mapping):
            return cls.model_validate(embedded)
        explicit = event.get("status")
        if explicit is not None:
            parsed_status = (
                explicit
                if isinstance(explicit, ToolStatus)
                else ToolStatus(str(explicit))
            )
            raw_code = event.get("error_code", ErrorCode.NONE)
            parsed_code = (
                raw_code
                if isinstance(raw_code, ErrorCode)
                else ErrorCode(str(raw_code))
            )
            return cls(
                status=parsed_status,
                summary=str(event.get("summary", "")),
                evidence_ids=tuple(event.get("evidence_ids") or ()),
                artifact_ids=tuple(event.get("artifact_ids") or ()),
                changed=bool(event.get("changed", False)),
                receipt=dict(event.get("receipt") or {}),
                retry_after=event.get("retry_after"),
                error_code=parsed_code,
            )
        if event.get("permission_denied"):
            status, code = ToolStatus.DENIED, ErrorCode.PERMISSION_DENIED
        elif event.get("cancelled"):
            status, code = ToolStatus.CANCELLED, ErrorCode.CANCELLED
        elif event.get("error"):
            status = (
                ToolStatus.RETRYABLE_ERROR
                if event.get("recoverable")
                else ToolStatus.FATAL_ERROR
            )
            code = ErrorCode.EXECUTION_FAILED
        elif isinstance(event.get("exit_code"), int) and event["exit_code"] != 0:
            status, code = ToolStatus.RETRYABLE_ERROR, ErrorCode.EXECUTION_FAILED
        else:
            status, code = ToolStatus.SUCCESS, ErrorCode.NONE
        return cls(
            status=status, summary=str(event.get("summary", "")), error_code=code
        )


class ToolManifest(HarnessModel):
    name: str = Field(min_length=1, max_length=128)
    version: str = "1.0.0"
    description: str = ""
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    output_schema: Dict[str, Any] = Field(default_factory=dict)
    capability: CapabilityClass
    risk: RiskLevel = RiskLevel.LOW
    approval_category: Optional[str] = None
    timeout_seconds: float = Field(default=30.0, gt=0, le=3600)
    cancellable: bool = True
    idempotent: bool = True
    reconcile: str = "repeat_safe"
    max_output_chars: int = Field(default=20_000, ge=0)
    redact_output: bool = True
    compensation: Optional[str] = None


class RunBudgets(HarnessModel):
    max_model_requests: int = Field(default=12, ge=1)
    max_tool_steps: int = Field(default=10, ge=1)
    max_input_tokens: Optional[int] = Field(default=None, ge=1)
    max_output_tokens: Optional[int] = Field(default=None, ge=1)
    deadline_at: Optional[datetime] = None
    max_cost_usd: Optional[float] = Field(default=None, ge=0)


class RunState(HarnessModel):
    run_id: str
    session_id: str
    turn_id: str
    step_id: Optional[str] = None
    lifecycle: RunLifecycle = RunLifecycle.PENDING
    goal: str = Field(default="", max_length=4_000)
    user_constraints: tuple[str, ...] = ()
    success_criteria: tuple[str, ...] = ()
    active_plan: tuple[Dict[str, Any], ...] = ()
    phase: str = "idle"
    allowed_transitions: tuple[str, ...] = ()
    budgets: RunBudgets = Field(default_factory=RunBudgets)
    model_requests: int = 0
    tool_steps: int = 0
    active_proposal: Dict[str, Any] = Field(default_factory=dict)
    policy_decision: Dict[str, Any] = Field(default_factory=dict)
    approval: Dict[str, Any] = Field(default_factory=dict)
    tool_receipt: Dict[str, Any] = Field(default_factory=dict)
    verified_facts: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    artifact_ids: tuple[str, ...] = ()
    changed_resources: tuple[str, ...] = ()
    failure_counters: Dict[str, int] = Field(default_factory=dict)
    stagnation_score: float = Field(default=0.0, ge=0)
    cancellation_requested: bool = False
    stop_reason: str = ""
    memory_checkpoint_id: Optional[str] = None
    compaction_checkpoint_id: Optional[str] = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class LedgerEvent(HarnessModel):
    event_id: str = Field(default_factory=lambda: new_id("evt"))
    event_type: str
    run_id: str
    session_id: str
    turn_id: str
    step_id: Optional[str] = None
    sequence: int = Field(default=0, ge=0)
    parent_event_id: Optional[str] = None
    occurred_at: datetime = Field(default_factory=utc_now)
    provider: str = ""
    model: str = ""
    payload: Dict[str, Any] = Field(default_factory=dict)
    payload_hash: str = ""
    policy_result: Dict[str, Any] = Field(default_factory=dict)
    redactions: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    artifact_ids: tuple[str, ...] = ()

    def with_hash(self) -> "LedgerEvent":
        if self.payload_hash:
            return self
        return self.model_copy(update={"payload_hash": content_hash(self.payload)})


class ExecutionReceipt(HarnessModel):
    receipt_id: str = Field(default_factory=lambda: new_id("rcpt"))
    run_id: str
    step_id: str
    tool_name: str
    idempotency_key: str
    status: str = "started"
    precondition_hash: Optional[str] = None
    effect_hash: Optional[str] = None
    reconciliation: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    completed_at: Optional[datetime] = None


class MemoryRecord(HarnessModel):
    memory_id: str = Field(default_factory=lambda: new_id("mem"))
    namespace: str
    scope: str
    content: str = Field(max_length=40_000)
    provenance: str
    trust: TrustClass
    source_event_ids: tuple[str, ...] = ()
    sensitivity: str = "normal"
    created_at: datetime = Field(default_factory=utc_now)
    expires_at: Optional[datetime] = None
    supersedes_id: Optional[str] = None
    superseded_by_id: Optional[str] = None
    deleted_at: Optional[datetime] = None


class CompactionCapsule(HarnessModel):
    checkpoint_id: str = Field(default_factory=lambda: new_id("compact"))
    goal: str
    user_constraints: tuple[str, ...] = ()
    success_criteria: tuple[str, ...] = ()
    decisions: tuple[str, ...] = ()
    completed_work: tuple[str, ...] = ()
    changed_resources: tuple[str, ...] = ()
    verified_facts: tuple[str, ...] = ()
    failures_and_rejected_approaches: tuple[str, ...] = ()
    open_questions: tuple[str, ...] = ()
    active_plan: tuple[str, ...] = ()
    next_action: str = ""
    evidence_and_artifact_references: tuple[str, ...] = ()
    source_event_start: Optional[int] = None
    source_event_end: Optional[int] = None
    created_at: datetime = Field(default_factory=utc_now)


class CompletionDecision(HarnessModel):
    accepted: bool
    reasons: tuple[str, ...] = ()
    evidence_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    unverified: tuple[str, ...] = ()


class SecretRedactor:
    """Small deterministic redactor used before persistence and diagnostics."""

    _PATTERNS = (
        re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,;]+"),
        re.compile(r"(?i)((?:api[_-]?key|token|password|secret)\s*[:=]\s*)[^\s,;]+"),
        re.compile(r"\b(?:sk|gsk|ghp|hf)_[A-Za-z0-9_-]{12,}\b"),
    )

    @classmethod
    def redact_text(cls, value: str) -> tuple[str, tuple[str, ...]]:
        text = str(value)
        labels: list[str] = []
        for index, pattern in enumerate(cls._PATTERNS, 1):

            def replace(match: re.Match[str]) -> str:
                labels.append(f"secret_pattern_{index}")
                prefix = match.group(1) if match.lastindex else ""
                return prefix + "[redacted]"

            text = pattern.sub(replace, text)
        return text, tuple(sorted(set(labels)))

    @classmethod
    def redact_mapping(
        cls, value: Mapping[str, Any]
    ) -> tuple[Dict[str, Any], tuple[str, ...]]:
        labels: list[str] = []
        sensitive_keys = ("authorization", "api_key", "apikey", "password", "secret")

        def clean(item: Any, key: str = "") -> Any:
            folded = key.casefold().replace("-", "_")
            key_is_sensitive = any(marker in folded for marker in sensitive_keys) or (
                folded == "token"
                or (folded.endswith("_token") and not folded.endswith("_tokens"))
            )
            if key and key_is_sensitive:
                labels.append(f"sensitive_key:{key}")
                return "[redacted]"
            if isinstance(item, Mapping):
                return {
                    str(child_key): clean(child, str(child_key))
                    for child_key, child in item.items()
                }
            if isinstance(item, (list, tuple)):
                return [clean(child) for child in item]
            if isinstance(item, str):
                redacted, found = cls.redact_text(item)
                labels.extend(found)
                return redacted
            if item is None or isinstance(item, (bool, int, float)):
                return item
            return str(item)

        return clean(value), tuple(sorted(set(labels)))


__all__ = [
    "CapabilityClass",
    "CompactionCapsule",
    "CompletionDecision",
    "ErrorCode",
    "ExecutionReceipt",
    "HarnessModel",
    "LedgerEvent",
    "MemoryRecord",
    "RiskLevel",
    "RunBudgets",
    "RunLifecycle",
    "RunState",
    "SCHEMA_VERSION",
    "SecretRedactor",
    "ToolManifest",
    "ToolOutcome",
    "ToolStatus",
    "TrustClass",
    "content_hash",
    "new_id",
    "utc_now",
]
