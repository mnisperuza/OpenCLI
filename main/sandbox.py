"""Minimal Docker-backed command sandbox for OpenCLI.

The container is created only for a requested command.  Docker is the security
boundary; command allowlists are deliberately not treated as one.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence


@dataclass(frozen=True)
class SandboxConfig:
    image: str = "python:3.12-slim"
    cpus: str = "1.0"
    memory: str = "1g"
    pids_limit: int = 256
    timeout_seconds: int = 30
    max_output_chars: int = 50_000


class DockerSandbox:
    """Run argv commands in a constrained, ephemeral Docker container."""

    def __init__(self, workspace: Path, config: SandboxConfig | None = None):
        self.workspace = workspace.resolve()
        self.config = config or SandboxConfig()

    @staticmethod
    def is_available() -> bool:
        docker = shutil.which("docker")
        if not docker:
            return False
        try:
            result = subprocess.run(
                [docker, "info"],
                capture_output=True,
                timeout=3,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return result.returncode == 0

    def run(
        self,
        command: Sequence[str],
        *,
        write_access: bool = False,
        timeout_seconds: int | None = None,
    ) -> Dict[str, Any]:
        """Execute an argv command without a host shell or network access."""
        argv = [str(part) for part in command]
        if not argv or not argv[0].strip():
            raise ValueError("Command must contain an executable")
        if not self.is_available():
            return {"error": "Docker is unavailable; sandbox command was not run."}

        docker = shutil.which("docker")
        assert docker is not None
        mount = f"type=bind,src={self.workspace},dst=/workspace"
        if not write_access:
            mount += ",readonly"
        invocation = [
            docker,
            "run",
            "--rm",
            "--network",
            "none",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=256m",
            "--pids-limit",
            str(self.config.pids_limit),
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--cpus",
            self.config.cpus,
            "--memory",
            self.config.memory,
            "--workdir",
            "/workspace",
            "--mount",
            mount,
            self.config.image,
            *argv,
        ]
        try:
            completed = subprocess.run(
                invocation,
                shell=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds or self.config.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {"error": "Sandbox command timed out."}
        except OSError as error:
            return {"error": f"Sandbox command failed to start: {error}"}

        output = (completed.stdout or "") + (completed.stderr or "")
        truncated = len(output) > self.config.max_output_chars
        if truncated:
            output = output[-self.config.max_output_chars :]
        return {
            "exit_code": completed.returncode,
            "output": output,
            "truncated": truncated,
            "write_access": write_access,
        }


__all__ = ["DockerSandbox", "SandboxConfig"]
