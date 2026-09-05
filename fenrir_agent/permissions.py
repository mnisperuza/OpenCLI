"""Workspace-scoped permission policy for FenrirAgent tools."""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, Optional, Set


class PermissionDecision(str, Enum):
    ALLOW_ONCE = "allow_once"
    ALLOW_SESSION = "allow_session"
    ALWAYS_ALLOW = "always_allow"
    DENY = "deny"


@dataclass(frozen=True)
class PermissionRequest:
    category: str
    action: str
    target: str
    reason: str
    workspace: Path


ApprovalCallback = Callable[[PermissionRequest], PermissionDecision]


class PermissionManager:
    """Resolve one-time, session, and persistent workspace permissions."""

    CATEGORIES = ("web", "api", "file_read", "file_write", "command")

    def __init__(
        self,
        workspace: Path,
        state_file: Optional[Path] = None,
        approval_callback: Optional[ApprovalCallback] = None,
    ):
        self.workspace = workspace.resolve()
        self.state_file = state_file or (
            Path.home() / ".fenrir" / "permissions.json"
        )
        self.approval_callback = approval_callback
        self.session_allowed: Set[str] = set()
        self.web_enabled = True
        # Pydantic AI may dispatch independent tool calls concurrently. Only one
        # terminal permission prompt may own stdin at a time.
        self._request_lock = threading.RLock()

    @property
    def workspace_key(self) -> str:
        return os.path.normcase(str(self.workspace))

    def _load(self) -> Dict[str, object]:
        try:
            data = json.loads(self.state_file.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return {"version": 1, "workspaces": {}}
        if not isinstance(data, dict) or not isinstance(
            data.get("workspaces"), dict
        ):
            return {"version": 1, "workspaces": {}}
        return data

    def _save(self, data: Dict[str, object]) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_file.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(data, indent=2, sort_keys=True), encoding="utf-8"
        )
        temporary.replace(self.state_file)

    def persistent_allowed(self) -> Set[str]:
        data = self._load()
        workspace = data["workspaces"].get(self.workspace_key, {})
        if not isinstance(workspace, dict):
            return set()
        allowed = workspace.get("allow", [])
        return {
            str(category)
            for category in allowed
            if category in self.CATEGORIES
        }

    def _persist_allow(self, category: str) -> None:
        data = self._load()
        workspaces = data["workspaces"]
        workspace = workspaces.setdefault(self.workspace_key, {})
        allowed = set(workspace.get("allow", []))
        allowed.add(category)
        workspace["allow"] = sorted(allowed)
        self._save(data)

    def set_persistent_allow(self, category: str, allowed: bool) -> None:
        """Set one workspace permission without resetting unrelated choices."""
        if category not in self.CATEGORIES:
            raise ValueError(f"Unknown permission category: {category}")
        data = self._load()
        workspaces = data["workspaces"]
        workspace = workspaces.setdefault(self.workspace_key, {})
        values = set(workspace.get("allow", []))
        if allowed:
            values.add(category)
        else:
            values.discard(category)
        if values:
            workspace["allow"] = sorted(values)
        else:
            workspaces.pop(self.workspace_key, None)
        self._save(data)

    def request(
        self, category: str, action: str, target: str, reason: str
    ) -> bool:
        with self._request_lock:
            return self._request_locked(category, action, target, reason)

    def _request_locked(
        self, category: str, action: str, target: str, reason: str
    ) -> bool:
        if category not in self.CATEGORIES:
            raise ValueError(f"Unknown permission category: {category}")
        if category == "web" and not self.web_enabled:
            return False
        if category in self.session_allowed:
            return True
        if category in self.persistent_allowed():
            return True
        if self.approval_callback is None:
            return False

        request = PermissionRequest(
            category=category,
            action=action,
            target=str(target),
            reason=reason,
            workspace=self.workspace,
        )
        try:
            decision = PermissionDecision(self.approval_callback(request))
        except (ValueError, TypeError, EOFError, KeyboardInterrupt):
            return False
        if decision == PermissionDecision.ALLOW_SESSION:
            self.session_allowed.add(category)
        elif decision == PermissionDecision.ALWAYS_ALLOW:
            self._persist_allow(category)
        return decision != PermissionDecision.DENY

    def reset(self) -> None:
        self.session_allowed.clear()
        data = self._load()
        data["workspaces"].pop(self.workspace_key, None)
        self._save(data)

    def status(self) -> Dict[str, object]:
        return {
            "workspace": str(self.workspace),
            "web_enabled": self.web_enabled,
            "session_allowed": sorted(self.session_allowed),
            "persistent_allowed": sorted(self.persistent_allowed()),
        }


__all__ = [
    "PermissionDecision",
    "PermissionManager",
    "PermissionRequest",
]
