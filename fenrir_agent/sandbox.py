"""Isolated command backends for Docker and user-controlled E2B sandboxes."""

from __future__ import annotations

import fnmatch
import hashlib
import os
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Optional, Sequence

from .interfaces import SandboxBackend


@dataclass(frozen=True)
class SandboxConfig:
    """Shared resource, output, and workspace-sync limits."""

    image: str = "python:3.12-slim"
    cpus: str = "1.0"
    memory: str = "1g"
    pids_limit: int = 256
    timeout_seconds: int = 30
    lifetime_seconds: int = 900
    max_output_chars: int = 50_000
    max_sync_files: int = 2_000
    max_sync_file_bytes: int = 2_000_000
    max_sync_total_bytes: int = 25_000_000
    excluded_patterns: tuple[str, ...] = (
        ".git", ".git/**", ".fenrir", ".fenrir/**", ".env", ".env.*",
        "*.pem", "*.key", "**/secrets*", "__pycache__",
        "**/__pycache__/**", ".venv", ".venv/**", "node_modules",
        "node_modules/**",
    )


class DockerSandbox:
    """Run argv commands in constrained, ephemeral Docker containers."""

    backend = "docker"

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
                [docker, "info"], capture_output=True, timeout=3, check=False
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return result.returncode == 0

    def status(self) -> Dict[str, Any]:
        return {
            "backend": self.backend, "available": self.is_available(),
            "image": self.config.image, "lifecycle": "ephemeral per command",
            "network": "disabled",
        }

    def _excluded(self, relative: str) -> bool:
        normalized = relative.replace("\\", "/")
        parts = PurePosixPath(normalized).parts
        if any(part.casefold() in {".git", ".fenrir", ".opencli", ".venv", "node_modules", "__pycache__"} for part in parts):
            return True
        name = parts[-1].casefold() if parts else ""
        if name == ".env" or name.startswith(".env.") or name.endswith((".pem", ".key")):
            return True
        return any(fnmatch.fnmatchcase(normalized.casefold(), pattern.casefold()) for pattern in self.config.excluded_patterns)

    def _snapshot_workspace(self, destination: Path) -> None:
        """Copy only bounded, non-secret workspace files into Docker input."""
        count = total = 0
        for source in sorted(self.workspace.rglob("*")):
            relative = source.relative_to(self.workspace).as_posix()
            if self._excluded(relative) or source.is_symlink():
                continue
            target = destination / relative
            if source.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not source.is_file():
                continue
            size = source.stat().st_size
            if size > self.config.max_sync_file_bytes:
                continue
            count += 1
            total += size
            if count > self.config.max_sync_files or total > self.config.max_sync_total_bytes:
                raise ValueError("Workspace exceeds Docker snapshot limits")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

    @staticmethod
    def _remove_timed_out_container(docker: str, cid_file: Path) -> bool:
        """Best-effort cleanup for a daemon-owned container after client timeout."""
        try:
            container_id = cid_file.read_text(encoding="utf-8").strip()
            if not container_id:
                return False
            subprocess.run(
                [docker, "rm", "--force", container_id],
                capture_output=True,
                timeout=10,
                check=False,
            )
            return True
        except (OSError, subprocess.TimeoutExpired):
            return False

    def run(
        self, command: Sequence[str], *, write_access: bool = False,
        timeout_seconds: int | None = None, cwd: str = ".",
    ) -> Dict[str, Any]:
        """Execute argv in a filtered workspace snapshot without host network."""
        argv = [str(part) for part in command]
        if not argv or not argv[0].strip():
            raise ValueError("Command must contain an executable")
        if not self.is_available():
            return {"error": "Docker is unavailable; sandbox command was not run."}
        logical_cwd = PurePosixPath(str(cwd).replace("\\", "/"))
        if logical_cwd.is_absolute() or ".." in logical_cwd.parts:
            raise ValueError("Sandbox cwd must stay inside the workspace")

        docker = shutil.which("docker")
        assert docker is not None
        with tempfile.TemporaryDirectory(prefix="fenrir-docker-") as directory:
            snapshot = Path(directory)
            cid_file = snapshot / "container.cid"
            try:
                self._snapshot_workspace(snapshot)
            except (OSError, ValueError) as error:
                return {"error": f"Docker snapshot failed: {error}", "backend": self.backend}
            mount = f"type=bind,src={snapshot},dst=/workspace"
            if not write_access:
                mount += ",readonly"
            workdir = "/workspace"
            if str(logical_cwd) not in {"", "."}:
                workdir += "/" + str(logical_cwd)
            invocation = [
                docker, "run", "--rm", "--network", "none", "--read-only",
                "--cidfile", str(cid_file),
                "--tmpfs", "/tmp:rw,noexec,nosuid,size=256m", "--pids-limit",
                str(self.config.pids_limit), "--cap-drop", "ALL", "--security-opt",
                "no-new-privileges", "--cpus", self.config.cpus, "--memory",
                self.config.memory, "--workdir", workdir, "--mount", mount,
                self.config.image, *argv,
            ]
            try:
                completed = subprocess.run(
                    invocation, shell=False, capture_output=True, text=True,
                    encoding="utf-8", errors="replace",
                    timeout=timeout_seconds or self.config.timeout_seconds, check=False,
                )
            except subprocess.TimeoutExpired:
                cleaned = self._remove_timed_out_container(docker, cid_file)
                return {
                    "error": "Sandbox command timed out.",
                    "backend": self.backend,
                    "cleanup_attempted": cleaned,
                }
            except OSError as error:
                return {"error": f"Sandbox command failed to start: {error}", "backend": self.backend}
        output = (completed.stdout or "") + (completed.stderr or "")
        truncated = len(output) > self.config.max_output_chars
        if truncated:
            output = output[-self.config.max_output_chars :]
        return {
            "backend": self.backend, "exit_code": completed.returncode,
            "output": output, "truncated": truncated,
            "write_access": write_access, "cwd": str(cwd),
            "changes_persisted": False,
        }


class E2BSandbox:
    """Connect to an E2B sandbox owned or explicitly created by the user.

    E2B remains optional. FenrirAgent never stores API keys. Workspace transfer is
    explicit, bounded, excludes secrets, and never propagates remote deletions.
    """

    backend = "e2b"
    remote_root = "/workspace"

    def __init__(
        self, workspace: Path, config: SandboxConfig | None = None,
        sandbox: Any = None,
    ):
        self.workspace = workspace.resolve()
        self.config = config or SandboxConfig()
        self._sandbox = sandbox
        self._baseline: Dict[str, str] = {}

    @staticmethod
    def sdk_available() -> bool:
        try:
            import e2b  # noqa: F401
        except ImportError:
            return False
        return True

    @staticmethod
    def _sdk() -> Any:
        try:
            from e2b import Sandbox
        except ImportError as error:
            raise RuntimeError(
                "E2B SDK unavailable. Install with: pip install 'fenrir-agent[sandbox]'"
            ) from error
        return Sandbox

    @classmethod
    def create(
        cls, workspace: Path, *, template: str | None = None,
        allow_network: bool = False, config: SandboxConfig | None = None,
    ) -> "E2BSandbox":
        active_config = config or SandboxConfig()
        sandbox = cls._sdk().create(
            template=template, timeout=active_config.lifetime_seconds,
            secure=True, allow_internet_access=allow_network,
            metadata={"client": "fenrir-agent"},
        )
        return cls(workspace, active_config, sandbox)

    @classmethod
    def connect(
        cls, workspace: Path, sandbox_id: str, *,
        config: SandboxConfig | None = None,
    ) -> "E2BSandbox":
        identifier = sandbox_id.strip()
        if not identifier or any(char.isspace() for char in identifier):
            raise ValueError("Invalid E2B sandbox ID")
        active_config = config or SandboxConfig()
        sandbox = cls._sdk().connect(
            identifier, timeout=active_config.lifetime_seconds
        )
        return cls(workspace, active_config, sandbox)

    @property
    def sandbox_id(self) -> str | None:
        value = getattr(self._sandbox, "sandbox_id", None)
        return str(value) if value else None

    def is_available(self) -> bool:
        if self._sandbox is None:
            return False
        try:
            return bool(self._sandbox.is_running())
        except Exception:
            return False

    def status(self) -> Dict[str, Any]:
        return {
            "backend": self.backend, "available": self.is_available(),
            "sandbox_id": self.sandbox_id, "remote_root": self.remote_root,
            "synced_files": len(self._baseline),
            "sync_policy": "explicit push/pull; remote deletions ignored",
        }

    def kill(self) -> bool:
        if self._sandbox is None:
            return False
        try:
            stopped = bool(self._sandbox.kill())
        finally:
            self._sandbox = None
        return stopped

    @staticmethod
    def _hash(content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    def _excluded(self, relative: str) -> bool:
        normalized = relative.replace("\\", "/")
        while normalized.startswith("./"):
            normalized = normalized[2:]
        parts = PurePosixPath(normalized).parts
        if any(part in {".git", ".fenrir", ".venv", "node_modules", "__pycache__"} for part in parts):
            return True
        name = parts[-1].casefold() if parts else ""
        if name == ".env" or name.startswith(".env.") or name.endswith((".pem", ".key")):
            return True
        return any(fnmatch.fnmatch(normalized, pattern) for pattern in self.config.excluded_patterns)

    def _workspace_files(self) -> Iterable[tuple[str, Path, bytes]]:
        count = 0
        total = 0
        for path in sorted(self.workspace.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(self.workspace).as_posix()
            if self._excluded(relative):
                continue
            size = path.stat().st_size
            if size > self.config.max_sync_file_bytes:
                continue
            count += 1
            total += size
            if count > self.config.max_sync_files:
                raise ValueError("Workspace exceeds E2B sync file limit")
            if total > self.config.max_sync_total_bytes:
                raise ValueError("Workspace exceeds E2B sync byte limit")
            yield relative, path, path.read_bytes()

    def push_workspace(self) -> Dict[str, Any]:
        """Upload bounded workspace snapshot. Must be called by user command."""
        if not self.is_available():
            return {"error": "E2B sandbox is unavailable."}
        uploaded = 0
        total = 0
        baseline: Dict[str, str] = {}
        for relative, _path, content in self._workspace_files():
            self._sandbox.files.write(f"{self.remote_root}/{relative}", content)
            baseline[relative] = self._hash(content)
            uploaded += 1
            total += len(content)
        self._baseline = baseline
        return {"backend": self.backend, "uploaded": uploaded, "bytes": total}

    @staticmethod
    def _entry_path(entry: Any) -> str:
        if isinstance(entry, dict):
            return str(entry.get("path") or entry.get("name") or "")
        return str(getattr(entry, "path", "") or getattr(entry, "name", ""))

    @staticmethod
    def _entry_is_file(entry: Any) -> bool:
        value = entry.get("type") if isinstance(entry, dict) else getattr(entry, "type", "")
        return "file" in str(value).casefold()

    def _remote_files(self) -> Iterable[tuple[str, bytes]]:
        entries = self._sandbox.files.list(self.remote_root, depth=20)
        count = 0
        total = 0
        for entry in entries:
            if not self._entry_is_file(entry):
                continue
            remote_path = self._entry_path(entry).replace("\\", "/")
            prefix = self.remote_root.rstrip("/") + "/"
            if not remote_path.startswith(prefix):
                continue
            relative = remote_path[len(prefix) :]
            relative_path = PurePosixPath(relative)
            if (
                not relative
                or relative_path.is_absolute()
                or ".." in relative_path.parts
                or self._excluded(relative)
            ):
                continue
            content = bytes(self._sandbox.files.read(remote_path, format="bytes"))
            if len(content) > self.config.max_sync_file_bytes:
                continue
            count += 1
            total += len(content)
            if count > self.config.max_sync_files or total > self.config.max_sync_total_bytes:
                raise ValueError("Remote workspace exceeds E2B pull limits")
            yield relative, content

    def pull_workspace(self, *, apply: bool = False) -> Dict[str, Any]:
        """Preview or apply non-conflicting changed files; never delete local files."""
        if not self.is_available():
            return {"error": "E2B sandbox is unavailable."}
        changed: list[str] = []
        conflicts: list[str] = []
        contents: Dict[str, bytes] = {}
        for relative, content in self._remote_files():
            remote_hash = self._hash(content)
            baseline_hash = self._baseline.get(relative)
            if remote_hash == baseline_hash:
                continue
            local = (self.workspace / Path(relative)).resolve()
            try:
                local.relative_to(self.workspace)
            except ValueError:
                continue
            local_hash = self._hash(local.read_bytes()) if local.is_file() else None
            if local_hash != baseline_hash:
                conflicts.append(relative)
                continue
            changed.append(relative)
            contents[relative] = content
        if apply:
            for relative in changed:
                target = (self.workspace / Path(relative)).resolve()
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary: Path | None = None
                try:
                    with tempfile.NamedTemporaryFile(
                        mode="wb",
                        prefix=f".{target.name}.fenrir-e2b-",
                        suffix=".tmp",
                        dir=target.parent,
                        delete=False,
                    ) as output:
                        output.write(contents[relative])
                        temporary = Path(output.name)
                    os.replace(temporary, target)
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
                self._baseline[relative] = self._hash(contents[relative])
        return {
            "backend": self.backend, "applied": apply, "changed": changed,
            "conflicts": conflicts, "deletions_ignored": True,
        }

    def run(
        self, command: Sequence[str], *, write_access: bool = False,
        timeout_seconds: int | None = None, cwd: str = ".",
    ) -> Dict[str, Any]:
        """Run safely quoted argv in connected E2B Linux sandbox."""
        argv = [str(part) for part in command]
        if not argv or not argv[0].strip():
            raise ValueError("Command must contain an executable")
        if not self.is_available():
            return {"error": "E2B sandbox is unavailable; command was not run."}
        logical = PurePosixPath(str(cwd).replace("\\", "/"))
        if logical.is_absolute() or ".." in logical.parts:
            raise ValueError("Sandbox cwd must stay inside /workspace")
        remote_cwd = self.remote_root if str(logical) in {"", "."} else f"{self.remote_root}/{logical}"
        try:
            result = self._sandbox.commands.run(
                shlex.join(argv), cwd=remote_cwd,
                timeout=timeout_seconds or self.config.timeout_seconds,
            )
        except Exception as error:
            exit_code = getattr(error, "exit_code", None)
            if isinstance(exit_code, int):
                output = str(getattr(error, "stdout", "") or "") + str(
                    getattr(error, "stderr", "") or ""
                )
                if not output:
                    output = str(error)
                truncated = len(output) > self.config.max_output_chars
                return {
                    "backend": self.backend,
                    "sandbox_id": self.sandbox_id,
                    "exit_code": exit_code,
                    "output": output[-self.config.max_output_chars :],
                    "truncated": truncated,
                    "write_access": write_access,
                    "cwd": str(cwd),
                }
            return {"error": f"E2B command failed: {error}", "backend": self.backend}
        output = str(getattr(result, "stdout", "") or "") + str(getattr(result, "stderr", "") or "")
        truncated = len(output) > self.config.max_output_chars
        if truncated:
            output = output[-self.config.max_output_chars :]
        return {
            "backend": self.backend, "sandbox_id": self.sandbox_id,
            "exit_code": int(getattr(result, "exit_code", 0)), "output": output,
            "truncated": truncated, "write_access": write_access, "cwd": str(cwd),
        }


class SandboxManager:
    """Select one active backend. Lifecycle changes remain user-only commands."""

    backend = "none"

    def __init__(self, workspace: Path, config: SandboxConfig | None = None):
        self.workspace = workspace.resolve()
        self.config = config or SandboxConfig()
        self.active: Optional[SandboxBackend] = None

    def use_docker(self, image: str | None = None) -> Dict[str, Any]:
        config = replace(self.config, image=image.strip()) if image else self.config
        if any(char.isspace() for char in config.image):
            raise ValueError("Invalid Docker image")
        backend = DockerSandbox(self.workspace, config)
        if not backend.is_available():
            return {"error": "Docker is unavailable. Install and start Docker Desktop."}
        self.active = backend
        self.backend = backend.backend
        return backend.status()

    def create_e2b(self, template: str | None = None, *, allow_network: bool = False) -> Dict[str, Any]:
        backend = E2BSandbox.create(
            self.workspace, template=template, allow_network=allow_network,
            config=self.config,
        )
        self.active = backend
        self.backend = backend.backend
        return backend.status()

    def connect_e2b(self, sandbox_id: str) -> Dict[str, Any]:
        backend = E2BSandbox.connect(self.workspace, sandbox_id, config=self.config)
        self.active = backend
        self.backend = backend.backend
        return backend.status()

    def disable(self) -> None:
        self.active = None
        self.backend = "none"

    def stop(self) -> Dict[str, Any]:
        if self.active is None:
            return {"backend": "none", "stopped": False}
        if isinstance(self.active, E2BSandbox):
            stopped = self.active.kill()
            self.disable()
            return {"backend": "e2b", "stopped": stopped}
        self.disable()
        return {"backend": "docker", "stopped": True, "note": "Docker commands are ephemeral"}

    def is_available(self) -> bool:
        return self.active is not None and self.active.is_available()

    def status(self) -> Dict[str, Any]:
        if self.active is None:
            return {
                "backend": "none", "available": False,
                "docker_installed": shutil.which("docker") is not None,
                "e2b_sdk_available": E2BSandbox.sdk_available(),
            }
        return self.active.status()

    def push_workspace(self) -> Dict[str, Any]:
        if not isinstance(self.active, E2BSandbox):
            return {"error": "Workspace push is only used by E2B."}
        return self.active.push_workspace()

    def pull_workspace(self, *, apply: bool = False) -> Dict[str, Any]:
        if not isinstance(self.active, E2BSandbox):
            return {"error": "Workspace pull is only used by E2B."}
        return self.active.pull_workspace(apply=apply)

    def run(
        self, command: Sequence[str], *, write_access: bool = False,
        timeout_seconds: int | None = None, cwd: str = ".",
    ) -> Dict[str, Any]:
        if self.active is None:
            return {"error": "No sandbox backend selected."}
        return self.active.run(
            command, write_access=write_access,
            timeout_seconds=timeout_seconds, cwd=cwd,
        )


__all__ = [
    "DockerSandbox", "E2BSandbox", "SandboxBackend", "SandboxConfig",
    "SandboxManager",
]
