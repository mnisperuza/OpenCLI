"""
OpenCLI — A calm, local AI assistant
"""

from ._version import __version__
__author__ = "Matias Nisperuza"

__all__ = [
    'main', 'OpenCLI', 'get_engine', 'OpenCLIEngine',
    'get_interrupt_handler', 'PydanticAgentRuntime', 'get_agent_runtime',
    'ModelBackend', 'ModelDescriptor', 'PermissionGate',
    'PermissionRequestData', 'SessionStore', 'ToolDescriptor', 'ToolProvider',
    'AgentLoopController', 'SandboxBackend',
    'ModelCapabilityProfile', 'ModelProfileRegistry',
]


def __getattr__(name):
    """Keep `import main` fast; load CLI and ML runtime only on demand."""
    if name in {"main", "OpenCLI"}:
        from main import cli
        return getattr(cli, name)
    if name in {"get_engine", "OpenCLIEngine", "get_interrupt_handler"}:
        from main import engine
        return getattr(engine, name)
    if name in {"PydanticAgentRuntime", "get_agent_runtime"}:
        from main import agent_runtime
        return getattr(agent_runtime, name)
    if name in {
        "ModelBackend", "ModelDescriptor", "PermissionGate",
        "PermissionRequestData", "SessionStore", "ToolDescriptor",
        "ToolProvider", "AgentLoopController", "SandboxBackend",
    }:
        from main import interfaces
        return getattr(interfaces, name)
    if name in {"ModelCapabilityProfile", "ModelProfileRegistry"}:
        from main import model_profiles
        return getattr(model_profiles, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
