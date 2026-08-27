"""Safe logical workspace navigation shared by CLI, tools, and sessions."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class WorkspaceContext:
    """Keep a session-local current directory inside one trusted root."""

    root: Path
    current_directory: Path | None = None
    allowed_roots: tuple[Path, ...] = field(init=False)

    def __post_init__(self) -> None:
        self.root = self.root.resolve()
        self.allowed_roots = (self.root,)
        self.current_directory = self.root

    def resolve(self, path: str = ".") -> Path:
        """Resolve an absolute or current-directory-relative path within root."""
        cleaned = str(path or ".").strip()
        if not cleaned:
            cleaned = "."
        candidate = Path(cleaned).expanduser()
        if not candidate.is_absolute():
            candidate = self.current_directory / candidate
        resolved = candidate.resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as error:
            raise ValueError("Path must stay inside trusted workspace") from error
        return resolved

    def set_current_directory(self, path: str) -> Path:
        target = self.resolve(path)
        if not target.is_dir():
            raise ValueError(f"Not a directory: {path}")
        self.current_directory = target
        return target

    def relative_path(self, path: Path | None = None) -> str:
        target = (path or self.current_directory).resolve()
        relative = target.relative_to(self.root).as_posix()
        return "." if relative == "." else relative

    def state(self) -> dict[str, object]:
        return {
            "workspace": str(self.root),
            "current_directory": self.relative_path(),
            "allowed_roots": [str(root) for root in self.allowed_roots],
        }


__all__ = ["WorkspaceContext"]
