"""
Fenrir Agent - a calm, local AI assistant.
"""

from ._version import __version__
__author__ = "Matias Nisperuza"

__all__ = [
    'main', 'FenrirAgent', 'get_engine', 'FenrirAgentEngine',
    'get_interrupt_handler', 'PydanticAgentRuntime', 'get_agent_runtime',
    'ModelBackend', 'ModelDescriptor', 'PermissionGate',
    'PermissionRequestData', 'SessionStore', 'ToolDescriptor', 'ToolProvider',
    'AgentLoopController', 'SandboxBackend',
    'ModelCapabilityProfile', 'ModelProfileRegistry',
]


def __getattr__(name):
    """Keep ``import fenrir_agent`` fast; load runtime components on demand."""
    if name in {"main", "FenrirAgent"}:
        from fenrir_agent import cli
        return getattr(cli, name)
    if name in {"get_engine", "FenrirAgentEngine", "get_interrupt_handler"}:
        from fenrir_agent import engine
        return getattr(engine, name)
    if name in {"PydanticAgentRuntime", "get_agent_runtime"}:
        from fenrir_agent import agent_runtime
        return getattr(agent_runtime, name)
    if name in {
        "ModelBackend", "ModelDescriptor", "PermissionGate",
        "PermissionRequestData", "SessionStore", "ToolDescriptor",
        "ToolProvider", "AgentLoopController", "SandboxBackend",
    }:
        from fenrir_agent import interfaces
        return getattr(interfaces, name)
    if name in {"ModelCapabilityProfile", "ModelProfileRegistry"}:
        from fenrir_agent import model_profiles
        return getattr(model_profiles, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
