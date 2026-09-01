"""Public Python API for OpenCLI.

The command-line application is the primary interface.  This module exposes
stable entry points for integrations without making callers depend on OpenCLI's
internal ``main`` package layout.
"""

from main._version import __version__
from main.interfaces import (
    AgentLoopController,
    ModelBackend,
    ModelDescriptor,
    PermissionGate,
    PermissionRequestData,
    SandboxBackend,
    SessionStore,
    ToolDescriptor,
    ToolProvider,
)

__all__ = [
    "__version__",
    "OpenCLI",
    "OpenCLIEngine",
    "main",
    "ModelBackend",
    "ModelDescriptor",
    "PermissionGate",
    "PermissionRequestData",
    "SessionStore",
    "ToolDescriptor",
    "ToolProvider",
    "AgentLoopController",
    "SandboxBackend",
]


def __getattr__(name: str):
    """Load UI and model runtime lazily so lightweight imports stay cheap."""
    if name in {"OpenCLI", "main"}:
        from main.cli import OpenCLI, main

        return {"OpenCLI": OpenCLI, "main": main}[name]
    if name == "OpenCLIEngine":
        from main.engine import OpenCLIEngine

        return OpenCLIEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
