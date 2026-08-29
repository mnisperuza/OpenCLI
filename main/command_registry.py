"""Shared slash-command metadata for terminal command discovery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class CommandSpec:
    command: str
    usage: str
    description: str
    category: str
    aliases: tuple[str, ...] = ()

    @property
    def completion(self) -> str:
        return self.command + (" " if self.usage != self.command else "")

    def matches(self, query: str) -> bool:
        needle = query.casefold().strip()
        values = (self.command, self.usage, self.description, *self.aliases)
        return not needle or any(needle in value.casefold() for value in values)


COMMAND_SPECS: tuple[CommandSpec, ...] = (
    CommandSpec("/help", "/help", "Show command reference", "General", ("/h",)),
    CommandSpec("/info", "/info", "Show OpenCLI project information", "General"),
    CommandSpec("/status", "/status", "Show runtime status", "General"),
    CommandSpec("/context", "/context", "Show context usage", "General"),
    CommandSpec("/usage", "/usage", "Show session token usage", "General"),
    CommandSpec("/prompt-size", "/prompt-size", "Show fixed prompt cost", "General"),
    CommandSpec("/pwd", "/pwd", "Show logical workspace directory", "Workspace"),
    CommandSpec("/cd", "/cd PATH", "Change logical workspace directory", "Workspace"),
    CommandSpec("/roots", "/roots", "Show allowed filesystem roots", "Workspace"),
    CommandSpec("/compact", "/compact [status|auto on|auto off]", "Compact older context", "Context"),
    CommandSpec("/agent", "/agent [status]", "Show agent runtime state", "Agent"),
    CommandSpec("/tools", "/tools", "List agent tools", "Agent"),
    CommandSpec("/tools-on", "/tools-on", "Enable agent tools", "Agent"),
    CommandSpec("/tools-off", "/tools-off", "Disable agent tools", "Agent"),
    CommandSpec("/tool-auto", "/tool-auto on|off", "Set proactive local routing", "Agent"),
    CommandSpec("/plan", "/plan [show|add STEP|set ID STATUS|clear]", "Manage session task plan", "Agent"),
    CommandSpec("/react", "/react [on|off|status]", "Control strict ReAct dispatcher", "Agent"),
    CommandSpec("/think", "/think PROMPT", "Send with supported reasoning mode", "Agent"),
    CommandSpec("/thinking", "/thinking off|low|medium|high|status", "Set native reasoning effort", "Agent"),
    CommandSpec("/web", "/web on|off|always|ask", "Set web access policy", "Tools"),
    CommandSpec("/sandbox", "/sandbox ACTION", "Control isolated command backend", "Tools"),
    CommandSpec("/permissions", "/permissions [reset]", "Show workspace permissions", "Tools"),
    CommandSpec("/history", "/history", "Show recent agent history", "Sessions"),
    CommandSpec("/new", "/new", "Start clean session", "Sessions", ("/newchat",)),
    CommandSpec("/session-name", "/session-name TEXT", "Name current session", "Sessions"),
    CommandSpec("/memory", "/memory [clear|notes|forget|list|current|records|correct|delete|export]", "Manage session memory", "Sessions", ("/mem",)),
    CommandSpec("/remember", "/remember TEXT", "Save durable user note", "Sessions"),
    CommandSpec("/harness", "/harness [status|runs|reconcile RUN_ID|resume RUN_ID|debug RUN_ID]", "Inspect durable harness state", "Sessions"),
    CommandSpec("/model", "/model [NAME]", "Select local or saved model", "Models"),
    CommandSpec("/model-add", "/model-add", "Add GGUF model profile", "Models", ("/modeladd",)),
    CommandSpec("/model-rm", "/model-rm", "Remove saved model profile", "Models", ("/modelrm",)),
    CommandSpec("/api", "/api", "Connect hosted model provider", "Models"),
    CommandSpec("/api-md", "/api-md", "Change hosted model", "Models"),
    CommandSpec("/api-del", "/api-del", "Remove API profile", "Models"),
    CommandSpec("/paste", "/paste", "Toggle multiline paste mode", "Input", ("/multiline",)),
    CommandSpec("/clear", "/clear", "Clear visible conversation", "General", ("/cls",)),
    CommandSpec("/endserver", "/endserver", "Unload local model server", "Models"),
    CommandSpec("/exit", "/exit", "Save and exit", "General", ("/quit", "/q")),
)


def match_commands(query: str, *, limit: int = 8) -> tuple[CommandSpec, ...]:
    """Return stable filtered command metadata for slash autocomplete."""
    matches: Iterable[CommandSpec] = (spec for spec in COMMAND_SPECS if spec.matches(query))
    return tuple(list(matches)[:limit])


__all__ = ["COMMAND_SPECS", "CommandSpec", "match_commands"]
