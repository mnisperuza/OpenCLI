"""User-visible, session-scoped task plans shared by CLI, TUI, and agent."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import uuid


PLAN_STATUSES = ("pending", "in_progress", "completed", "dismissed")


@dataclass
class TaskPlanItem:
    id: str
    text: str
    status: str = "pending"


class TaskPlanStore:
    """Persist reviewed plans outside normal workspace file tools."""

    def __init__(self, workspace: Path, session_id: str, root: Path | None = None):
        digest = hashlib.sha256(
            os.path.normcase(str(workspace.resolve())).encode("utf-8")
        ).hexdigest()[:12]
        self.path = (
            root or Path.home() / ".fenrir" / "plans"
        ) / digest / f"{session_id}.json"

    def load(self) -> list[TaskPlanItem]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return []
        if not isinstance(payload, dict) or payload.get("version") != 1:
            return []
        items: list[TaskPlanItem] = []
        for value in payload.get("items", []):
            if not isinstance(value, dict):
                continue
            text = " ".join(str(value.get("text", "")).split()).strip()
            status = str(value.get("status", "pending"))
            if text and status in PLAN_STATUSES:
                items.append(TaskPlanItem(str(value.get("id") or uuid.uuid4().hex[:8]), text, status))
        return items

    def save(self, items: list[TaskPlanItem]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "items": [asdict(item) for item in items],
        }
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    @staticmethod
    def add(items: list[TaskPlanItem], text: str) -> TaskPlanItem:
        cleaned = " ".join(text.split()).strip()
        if not cleaned:
            raise ValueError("Plan item cannot be empty")
        item = TaskPlanItem(uuid.uuid4().hex[:8], cleaned)
        items.append(item)
        return item

    def add_item(self, text: str) -> TaskPlanItem:
        """Add and persist one item."""
        items = self.load()
        item = self.add(items, text)
        self.save(items)
        return item

    def replace(self, steps: list[str]) -> list[TaskPlanItem]:
        """Replace plan with a bounded ordered set of concrete steps."""
        if not 1 <= len(steps) <= 30:
            raise ValueError("A plan must contain between 1 and 30 steps")
        items: list[TaskPlanItem] = []
        for step in steps:
            self.add(items, step)
        self.save(items)
        return items

    def clear(self) -> None:
        """Persist an empty plan without deleting session metadata."""
        self.save([])

    def update_status(self, item_id: str, status: str) -> TaskPlanItem:
        """Update one persisted item for an agent or UI action."""
        if status not in PLAN_STATUSES:
            raise ValueError(f"Unknown plan status: {status}")
        items = self.load()
        for item in items:
            if item.id == item_id:
                item.status = status
                self.save(items)
                return item
        raise ValueError(f"Task-plan item not found: {item_id}")


__all__ = ["PLAN_STATUSES", "TaskPlanItem", "TaskPlanStore"]
