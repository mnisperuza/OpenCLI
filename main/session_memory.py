"""User-controlled, per-workspace OpenCLI session archives."""

from __future__ import annotations

import hashlib
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional


@dataclass
class SessionRecord:
    session_id: str
    path: Path
    created_at: datetime
    notes: List[str] = field(default_factory=list)
    transcript: str = ""


class SessionMemoryStore:
    """Save chat transcripts as reviewable Markdown; load only on request."""

    MAX_LOADED_CHARS = 100_000

    def __init__(self, workspace: Path, root: Optional[Path] = None):
        self.workspace = workspace.resolve()
        digest = hashlib.sha256(
            os.path.normcase(str(self.workspace)).encode("utf-8")
        ).hexdigest()[:12]
        self.directory = (root or Path.home() / ".opencli" / "sessions") / digest

    def create(self, now: Optional[datetime] = None) -> SessionRecord:
        created = now or datetime.now().astimezone()
        stamp = created.strftime("%Y-%m-%d_%H-%M-%S")
        session_id = f"{stamp}_{uuid.uuid4().hex[:8]}"
        record = SessionRecord(
            session_id=session_id,
            path=self.directory / f"{session_id}.md",
            created_at=created,
        )
        self.save(record, "")
        return record

    def save(self, record: SessionRecord, transcript: str) -> None:
        record.transcript = transcript
        self.directory.mkdir(parents=True, exist_ok=True)
        notes = "\n".join(f"- {note}" for note in record.notes) or "_No notes._"
        text = (
            "# OpenCLI Session\n\n"
            f"- Session: `{record.session_id}`\n"
            f"- Created: `{record.created_at.isoformat(timespec='seconds')}`\n"
            f"- Workspace: `{self.workspace}`\n\n"
            "## Notes\n\n"
            f"{notes}\n\n"
            "## Transcript\n\n"
            f"{transcript.strip()}\n"
        )
        temporary = record.path.with_suffix(".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(record.path)

    def list(self) -> List[Path]:
        if not self.directory.is_dir():
            return []
        return sorted(self.directory.glob("*.md"), key=lambda path: path.stat().st_mtime, reverse=True)

    def load(self, path: Path) -> str:
        resolved = path.resolve()
        try:
            resolved.relative_to(self.directory.resolve())
        except ValueError as error:
            raise ValueError("Session must belong to current workspace") from error
        if not resolved.is_file() or resolved.suffix.casefold() != ".md":
            raise ValueError("Unknown session archive")
        content = resolved.read_text(encoding="utf-8", errors="replace")
        if len(content) > self.MAX_LOADED_CHARS:
            content = content[-self.MAX_LOADED_CHARS :]
        return content

    def remember(self, record: SessionRecord, note: str, transcript: str) -> None:
        cleaned = " ".join(note.split()).strip()
        if not cleaned:
            raise ValueError("Memory note cannot be empty")
        if len(cleaned) > 2_000:
            raise ValueError("Memory note exceeds 2,000 characters")
        record.notes.append(cleaned)
        self.save(record, transcript)


__all__ = ["SessionMemoryStore", "SessionRecord"]
