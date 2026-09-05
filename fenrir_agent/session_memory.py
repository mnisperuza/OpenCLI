"""User-controlled, per-workspace FenrirAgent session archives."""

from __future__ import annotations

import hashlib
import os
import re
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
    title: str = ""
    current_directory: str = "."
    notes: List[str] = field(default_factory=list)
    compactions: List["CompactionRecord"] = field(default_factory=list)
    tool_archives: List["ToolArchiveRecord"] = field(default_factory=list)
    transcript: str = ""


@dataclass(frozen=True)
class CompactionRecord:
    """Reviewable checkpoint made before active chat history is shortened."""

    created_at: datetime
    summary: str
    source_transcript: str


@dataclass(frozen=True)
class ToolArchiveRecord:
    """Full tool payload removed from active model context."""

    created_at: datetime
    content: str


class SessionMemoryStore:
    """Save chat transcripts as reviewable Markdown; load only on request."""

    MAX_LOADED_CHARS = 100_000
    MAX_CONTEXT_CHARS = 24_000
    MAX_TOOL_ARCHIVE_CHARS = 250_000
    MAX_TOOL_ARCHIVES = 20
    MAX_TITLE_CHARS = 60

    @staticmethod
    def sanitize_durable_context(content: str) -> str:
        """Remove diagnostic failures from model-readable durable context."""
        kept: List[str] = []
        for block in re.split(r"\n\s*\n", str(content or "")):
            stripped = block.strip()
            folded = stripped.casefold()
            if not stripped:
                continue
            if folded.startswith((
                "tool validation error:",
                "traceback (most recent call last):",
                "error log:",
                "exception:",
            )):
                continue
            if folded.startswith("tool result [") and any(marker in folded for marker in (
                '"error":', "'error':", "traceback (most recent call last):",
                '"status": "failed"', '"status": "fatal_error"',
                '"status": "retryable_error"', "permission denied", "not a file:",
            )):
                continue
            if folded.startswith("failures and rejected approaches"):
                continue
            kept.append(stripped)
        return "\n\n".join(kept)

    def __init__(self, workspace: Path, root: Optional[Path] = None):
        self.workspace = workspace.resolve()
        digest = hashlib.sha256(
            os.path.normcase(str(self.workspace)).encode("utf-8")
        ).hexdigest()[:12]
        self.directory = (root or Path.home() / ".fenrir" / "sessions") / digest

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
        transcript = self.sanitize_durable_context(transcript)
        record.transcript = transcript
        self.directory.mkdir(parents=True, exist_ok=True)
        notes = "\n".join(f"- {note}" for note in record.notes) or "_No notes._"
        compact_history = self._render_compactions(record.compactions)
        tool_archive = self._render_tool_archives(record.tool_archives)
        text = (
            "# FenrirAgent Session\n\n"
            f"- Session: `{record.session_id}`\n"
            f"- Created: `{record.created_at.isoformat(timespec='seconds')}`\n"
            f"- Title: {self.clean_title(record.title) or '_Untitled_'}\n"
            f"- Workspace: `{self.workspace}`\n"
            f"- Current directory: `{self.clean_current_directory(record.current_directory)}`\n\n"
            "## Notes\n\n"
            f"{notes}\n\n"
            "## Compact History\n\n"
            f"{compact_history}\n\n"
            "## Tool Result Archive\n\n"
            f"{tool_archive}\n\n"
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
        resolved = self._validate_archive(path)
        content = resolved.read_text(encoding="utf-8", errors="replace")
        if len(content) > self.MAX_LOADED_CHARS:
            content = content[-self.MAX_LOADED_CHARS :]
        return content

    def load_record(self, path: Path) -> SessionRecord:
        """Restore session metadata without discarding prior archives on save."""
        resolved = self._validate_archive(path)
        content = resolved.read_text(encoding="utf-8", errors="replace")
        session_id = self._metadata_value(content, "Session") or resolved.stem
        created = self._parse_created(self._metadata_value(content, "Created"), resolved)
        raw_title = self._metadata_value(content, "Title")
        return SessionRecord(
            session_id=session_id.strip("`"),
            path=resolved,
            created_at=created,
            title="" if raw_title == "_Untitled_" else self.clean_title(raw_title),
            current_directory=self.clean_current_directory(
                self._metadata_value(content, "Current directory")
            ),
            notes=self._parse_notes(self._markdown_section(content, "Notes")),
            compactions=self._parse_compactions(
                self._markdown_section(content, "Compact History")
            ),
            tool_archives=self._parse_tool_archives(
                self._markdown_section(content, "Tool Result Archive")
            ),
            transcript=self._markdown_section(content, "Transcript"),
        )

    def load_context(self, path: Path, max_chars: Optional[int] = None) -> str:
        """Load compact, user-reviewable archive context instead of full history.

        Historical archives can be much larger than a model context window.  Keep
        explicit notes, newest compact checkpoint, and recent transcript tail.
        The caller must still mark this data as untrusted historical text.
        """
        content = self._validate_archive(path).read_text(
            encoding="utf-8", errors="replace"
        )
        budget = max(1_000, min(max_chars or self.MAX_CONTEXT_CHARS, self.MAX_LOADED_CHARS))
        notes = self._markdown_section(content, "Notes")
        compact_history = self._markdown_section(content, "Compact History")
        transcript = self.sanitize_durable_context(
            self._markdown_section(content, "Transcript")
        )
        checkpoint = self._latest_checkpoint(compact_history)
        sections = []
        if notes and notes != "_No notes._":
            sections.append(f"Durable user notes:\n{notes}")
        if checkpoint:
            sections.append(f"Latest compact checkpoint:\n{checkpoint}")
        prefix = "\n\n".join(sections)
        remaining = max(500, budget - len(prefix) - 40)
        if len(transcript) > remaining:
            transcript = "…" + transcript[-remaining:]
        if transcript:
            sections.append(f"Recent transcript:\n{transcript}")
        return "\n\n".join(sections).strip()

    def load_capsule(self, path: Path, max_chars: Optional[int] = None) -> str:
        """Return bounded session reference suitable for one active chat import."""
        record = self.load_record(path)
        context = self.load_context(path, max_chars)
        title = record.title or record.session_id
        return f"SESSION CAPSULE: {title}\n\n{context}".strip()

    def set_title(self, record: SessionRecord, title: str, transcript: str) -> str:
        cleaned = self.clean_title(title)
        if not cleaned:
            raise ValueError("Session title cannot be empty")
        record.title = cleaned
        self.save(record, transcript)
        return cleaned

    @classmethod
    def clean_title(cls, title: str | None) -> str:
        cleaned = " ".join(str(title or "").split()).strip()
        cleaned = cleaned.replace("`", "").replace("#", "")
        if any(ord(character) < 32 for character in cleaned):
            raise ValueError("Session title contains control characters")
        if len(cleaned) > cls.MAX_TITLE_CHARS:
            cleaned = cleaned[: cls.MAX_TITLE_CHARS].rstrip()
        return cleaned

    @staticmethod
    def clean_current_directory(path: str | None) -> str:
        cleaned = str(path or ".").strip().replace("\\", "/")
        if not cleaned or cleaned == ".":
            return "."
        if cleaned.startswith("/") or ":" in cleaned:
            return "."
        parts = [part for part in cleaned.split("/") if part and part != "."]
        if any(part == ".." for part in parts):
            return "."
        return "/".join(parts) or "."

    def remember(self, record: SessionRecord, note: str, transcript: str) -> None:
        cleaned = " ".join(note.split()).strip()
        if not cleaned:
            raise ValueError("Memory note cannot be empty")
        if len(cleaned) > 2_000:
            raise ValueError("Memory note exceeds 2,000 characters")
        if cleaned not in record.notes:
            record.notes.append(cleaned)
        self.save(record, transcript)

    def forget_notes(self, record: SessionRecord, transcript: str) -> None:
        record.notes.clear()
        self.save(record, transcript)

    def record_compaction(
        self,
        record: SessionRecord,
        *,
        summary: str,
        source_transcript: str,
        transcript: str,
    ) -> None:
        record.compactions.append(
            CompactionRecord(
                created_at=datetime.now().astimezone(),
                summary=self.sanitize_durable_context(summary),
                source_transcript=self.sanitize_durable_context(source_transcript),
            )
        )
        self.save(record, transcript)

    def archive_tool_results(self, record: SessionRecord, content: str) -> None:
        cleaned = self.sanitize_durable_context(content)
        if cleaned:
            if len(cleaned) > self.MAX_TOOL_ARCHIVE_CHARS:
                removed = len(cleaned) - self.MAX_TOOL_ARCHIVE_CHARS
                cleaned = (
                    cleaned[: self.MAX_TOOL_ARCHIVE_CHARS]
                    + f"\n\n[Archive truncated: {removed:,} additional characters omitted.]"
                )
            record.tool_archives.append(
                ToolArchiveRecord(datetime.now().astimezone(), cleaned)
            )
            if len(record.tool_archives) > self.MAX_TOOL_ARCHIVES:
                del record.tool_archives[: -self.MAX_TOOL_ARCHIVES]

    def _validate_archive(self, path: Path) -> Path:
        resolved = path.resolve()
        try:
            resolved.relative_to(self.directory.resolve())
        except ValueError as error:
            raise ValueError("Session must belong to current workspace") from error
        if not resolved.is_file() or resolved.suffix.casefold() != ".md":
            raise ValueError("Unknown session archive")
        return resolved

    @staticmethod
    def _markdown_section(content: str, name: str) -> str:
        marker = f"## {name}\n"
        start = content.rfind(marker) if name == "Transcript" else content.find(marker)
        if start < 0:
            return ""
        start += len(marker)
        end = content.find("\n## ", start)
        return content[start : None if end < 0 else end].strip()

    @staticmethod
    def _metadata_value(content: str, name: str) -> str:
        prefix = f"- {name}: "
        for line in content.splitlines():
            if line.startswith(prefix):
                return line[len(prefix) :].strip().strip("`")
        return ""

    @staticmethod
    def _parse_created(value: str, path: Path) -> datetime:
        try:
            return datetime.fromisoformat(value.strip("`"))
        except ValueError:
            return datetime.fromtimestamp(path.stat().st_mtime).astimezone()

    @staticmethod
    def _parse_notes(section: str) -> List[str]:
        if not section or section == "_No notes._":
            return []
        return [line[2:].strip() for line in section.splitlines() if line.startswith("- ")]

    @staticmethod
    def _entries(section: str) -> List[tuple[datetime, str]]:
        if not section or section.startswith("_No "):
            return []
        entries = []
        for block in section.split("\n### "):
            cleaned = block.removeprefix("### ").strip()
            stamp, separator, content = cleaned.partition("\n\n")
            if not separator:
                continue
            try:
                entries.append((datetime.fromisoformat(stamp.strip()), content))
            except ValueError:
                continue
        return entries

    @classmethod
    def _parse_compactions(cls, section: str) -> List[CompactionRecord]:
        records = []
        for created, content in cls._entries(section):
            marker = "Memory capsule:\n"
            source_marker = "\n\nArchived transcript before compaction:\n"
            if not content.startswith(marker):
                continue
            summary, _, source = content[len(marker) :].partition(source_marker)
            records.append(CompactionRecord(created, summary.strip(), source.strip()))
        return records

    @classmethod
    def _parse_tool_archives(cls, section: str) -> List[ToolArchiveRecord]:
        return [ToolArchiveRecord(created, content.strip()) for created, content in cls._entries(section)]

    @staticmethod
    def _latest_checkpoint(compact_history: str) -> str:
        if not compact_history or compact_history == "_No compactions._":
            return ""
        entries = compact_history.split("\n### ")
        latest = entries[-1]
        summary_marker = "Memory capsule:\n"
        start = latest.find(summary_marker)
        if start < 0:
            return ""
        source_marker = "\n\nArchived transcript before compaction:"
        return latest[start + len(summary_marker) : latest.find(source_marker, start) if source_marker in latest else None].strip()

    @staticmethod
    def _render_compactions(compactions: List[CompactionRecord]) -> str:
        if not compactions:
            return "_No compactions._"
        rendered = []
        for item in compactions:
            rendered.append(
                "### " + item.created_at.isoformat(timespec="seconds") + "\n\n"
                "Memory capsule:\n"
                f"{item.summary}\n\n"
                "Archived transcript before compaction:\n"
                f"{item.source_transcript}"
            )
        return "\n\n".join(rendered)

    @staticmethod
    def _render_tool_archives(archives: List[ToolArchiveRecord]) -> str:
        if not archives:
            return "_No pruned tool results._"
        return "\n\n".join(
            "### " + item.created_at.isoformat(timespec="seconds") + "\n\n" + item.content
            for item in archives
        )


__all__ = [
    "CompactionRecord",
    "SessionMemoryStore",
    "SessionRecord",
    "ToolArchiveRecord",
]
