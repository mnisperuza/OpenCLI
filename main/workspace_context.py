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
        """Resolve an absolute, cwd-relative, or returned root-relative path.

        Tool results expose paths relative to ``root``. When current directory is
        nested, accepting those paths verbatim prevents accidental duplication such
        as ``src/src/app.py`` on a follow-up read.
        """
        cleaned = str(path or ".").strip()
        if not cleaned:
            cleaned = "."
        candidate = Path(cleaned).expanduser()
        if not candidate.is_absolute():
            candidate = self._relative_anchor(candidate) / candidate
        resolved = candidate.resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as error:
            raise ValueError("Path must stay inside trusted workspace") from error
        return resolved

    def _relative_anchor(self, candidate: Path) -> Path:
        """Choose root for reusable tool paths; current directory otherwise."""
        current_relative = self.current_directory.relative_to(self.root)
        current_parts = tuple(part.casefold() for part in current_relative.parts)
        candidate_parts = tuple(part.casefold() for part in candidate.parts)
        if (
            current_parts
            and len(candidate_parts) >= len(current_parts)
            and candidate_parts[:len(current_parts)] == current_parts
        ):
            return self.root
        return self.current_directory

    def resolve_mutation(self, path: str) -> Path:
        """Resolve a write target while rejecting symlinked path components.

        Atomic replacement protects the final write; rejecting symlinks closes
        the common workspace-escape and path-swap route before that boundary.
        """
        cleaned = str(path or "").strip()
        if not cleaned:
            raise ValueError("Mutation path cannot be empty")
        candidate = Path(cleaned).expanduser()
        if not candidate.is_absolute():
            candidate = self._relative_anchor(candidate) / candidate
        lexical = candidate.absolute()
        try:
            relative = lexical.relative_to(self.root)
        except ValueError as error:
            raise ValueError("Path must stay inside trusted workspace") from error
        cursor = self.root
        for part in relative.parts:
            cursor = cursor / part
            if cursor.exists() and cursor.is_symlink():
                raise ValueError("Mutation paths cannot traverse symbolic links")
        parent = lexical.parent.resolve()
        try:
            parent.relative_to(self.root)
        except ValueError as error:
            raise ValueError("Mutation path parent must stay inside trusted workspace") from error
        return lexical

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
