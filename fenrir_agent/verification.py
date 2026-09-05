"""Sandbox-only verification recipes and evidence records."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from .sandbox import SandboxManager


PermissionCallback = Callable[[str, str, str, str], bool]


@dataclass(frozen=True)
class VerificationRecipe:
    name: str
    command: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class VerificationResult:
    recipe: str
    command: tuple[str, ...]
    backend: str
    status: str
    exit_code: Optional[int]
    output: str
    evidence_id: str
    created_at: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "recipe": self.recipe,
            "command": self.command,
            "backend": self.backend,
            "status": self.status,
            "exit_code": self.exit_code,
            "output": self.output,
            "evidence_id": self.evidence_id,
            "created_at": self.created_at,
        }


class VerificationManager:
    """Choose explicit project checks and run them only in a selected sandbox."""

    MAX_OUTPUT_CHARS = 20_000

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.resolve()
        self.last_result: Optional[VerificationResult] = None

    def recipes(self, current_directory: Path) -> tuple[VerificationRecipe, ...]:
        root = current_directory.resolve()
        try:
            root.relative_to(self.workspace)
        except ValueError:
            return ()
        recipes: list[VerificationRecipe] = []
        if (root / "tests").is_dir() or (root / "pytest.ini").is_file():
            recipes.append(
                VerificationRecipe(
                    "python-tests",
                    ("python", "-m", "pytest", "-q"),
                    "Run the Python test suite.",
                )
            )
        if (root / "package.json").is_file():
            recipes.append(
                VerificationRecipe(
                    "node-tests", ("npm", "test"), "Run the package test script."
                )
            )
        if (root / "Cargo.toml").is_file():
            recipes.append(
                VerificationRecipe(
                    "rust-tests", ("cargo", "test"), "Run the Rust test suite."
                )
            )
        return tuple(recipes)

    def select(self, current_directory: Path, requested: str = "auto") -> VerificationRecipe:
        available = self.recipes(current_directory)
        if not available:
            raise ValueError("No supported verification recipe was detected here.")
        requested = requested.casefold().strip() or "auto"
        if requested == "auto":
            return available[0]
        for recipe in available:
            if recipe.name == requested:
                return recipe
        raise ValueError(
            "Unknown verification recipe. Available: "
            + ", ".join(recipe.name for recipe in available)
        )

    def run(
        self,
        sandbox: SandboxManager,
        current_directory: Path,
        relative_directory: str,
        permission_callback: PermissionCallback,
        requested: str = "auto",
    ) -> VerificationResult:
        if not sandbox.is_available():
            raise RuntimeError(
                "Verification requires an active sandbox. Use /sandbox on."
            )
        recipe = self.select(current_directory, requested)
        command_text = " ".join(recipe.command)
        if not permission_callback(
            "command",
            "verify",
            command_text,
            f"{recipe.reason} Verification is sandbox-only and read-only to the host.",
        ):
            raise PermissionError("Verification permission denied.")
        raw = sandbox.run(
            recipe.command,
            write_access=False,
            cwd=relative_directory,
        )
        output = str(raw.get("output", ""))[-self.MAX_OUTPUT_CHARS :]
        exit_code = raw.get("exit_code")
        status = "passed" if exit_code == 0 and not raw.get("error") else "failed"
        evidence_payload = (
            f"{recipe.name}\0{command_text}\0{exit_code}\0{output}"
        ).encode("utf-8", errors="replace")
        result = VerificationResult(
            recipe=recipe.name,
            command=recipe.command,
            backend=str(raw.get("backend", sandbox.backend)),
            status=status,
            exit_code=int(exit_code) if exit_code is not None else None,
            output=output or str(raw.get("error", "")),
            evidence_id="evidence_verify_" + hashlib.sha256(evidence_payload).hexdigest()[:24],
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self.last_result = result
        return result

    def status(self) -> dict[str, Any]:
        return self.last_result.as_dict() if self.last_result else {"status": "not_run"}


__all__ = ["VerificationManager", "VerificationRecipe", "VerificationResult"]
