"""Stable internal contracts shared by FenrirAgent runtime components.

These protocols describe boundaries only. They keep local and hosted backends,
tool implementations, permission prompts, and session stores interchangeable
without making a provider-specific API part of the CLI layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Sequence


@dataclass(frozen=True)
class ModelDescriptor:
    """Model identity and capabilities needed by the runtime."""

    key: str
    display_name: str
    backend: str
    context_window: Optional[int] = None
    supports_tools: bool = False
    supports_vision: bool = False
    supports_strict_schema: bool = False
    supports_cancellation: bool = False


@dataclass(frozen=True)
class ToolDescriptor:
    """User-visible, permission-aware tool metadata."""

    name: str
    category: str
    mutates_workspace: bool = False
    requires_permission: Optional[str] = None
    version: str = "1.0.0"
    risk: str = "low"
    idempotent: bool = True


@dataclass(frozen=True)
class PermissionRequestData:
    """Dependency-free form of one requested capability."""

    category: str
    action: str
    target: str
    reason: str
    workspace: Path


class ModelBackend(Protocol):
    """Minimal local or hosted model backend contract."""

    backend: str

    def stream_chat(
        self,
        messages: Sequence[Any],
        tools: Sequence[Mapping[str, Any]],
        tool_choice: Any = "auto",
    ) -> Any: ...


class ToolProvider(Protocol):
    """Expose stable tool metadata to CLI and model adapters."""

    @property
    def available_tools(self) -> Sequence[str]: ...


class PermissionGate(Protocol):
    """Approve or deny one sensitive action."""

    def request(self, category: str, action: str, target: str, reason: str) -> bool: ...


class SessionStore(Protocol):
    """Persist user-controlled session records."""

    def create(self) -> Any: ...

    def save(self, record: Any, transcript: str) -> None: ...

    def load(self, path: Path) -> str: ...


class SandboxBackend(Protocol):
    """Execute bounded argv commands behind an enforced isolation boundary."""

    backend: str

    def is_available(self) -> bool: ...

    def run(
        self,
        command: Sequence[str],
        *,
        write_access: bool = False,
        timeout_seconds: Optional[int] = None,
        cwd: str = ".",
    ) -> Mapping[str, Any]: ...

    def status(self) -> Mapping[str, Any]: ...


class AgentLoopController(Protocol):
    """Own deterministic budgets around model-selected tool actions."""

    def begin_turn(self, goal: str) -> None: ...

    def dispatch(self, decision: str, *, summary: str = "") -> Mapping[str, Any]: ...

    def before_tool(self, name: str, arguments: Any) -> int: ...

    def after_tool(self, event: Mapping[str, Any]) -> None: ...

    def submit_critique(self, critique: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def loop_context(self) -> Mapping[str, Any]: ...

    def fallback_to_user(self, reason: str) -> Mapping[str, Any]: ...

    def status(self) -> Mapping[str, Any]: ...


__all__ = [
    "ModelBackend",
    "ModelDescriptor",
    "PermissionGate",
    "PermissionRequestData",
    "AgentLoopController",
    "SandboxBackend",
    "SessionStore",
    "ToolDescriptor",
    "ToolProvider",
]
