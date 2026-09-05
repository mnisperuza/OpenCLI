"""Shared slash-command metadata for terminal command discovery."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import get_close_matches
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
    CommandSpec("/info", "/info", "Show Fenrir Agent project information", "General"),
    CommandSpec("/status", "/status", "Show runtime status", "General"),
    CommandSpec("/context", "/context", "Show context usage", "General"),
    CommandSpec("/usage", "/usage", "Show session token usage", "General"),
    CommandSpec("/prompt-size", "/prompt-size", "Show fixed prompt cost", "General"),
    CommandSpec("/pwd", "/pwd", "Show logical workspace directory", "Workspace"),
    CommandSpec("/cd", "/cd PATH", "Change logical workspace directory", "Workspace"),
    CommandSpec("/roots", "/roots", "Show allowed filesystem roots", "Workspace"),
    CommandSpec("/compact", "/compact [status|auto on|auto off]", "Compact older context", "Context"),
    CommandSpec("/agent", "/agent [status]", "Show agent runtime state", "Agent"),
    CommandSpec("/tools", "/tools [enable|disable TOOLSET|reset]", "List or configure agent toolsets", "Agent"),
    CommandSpec("/tools-on", "/tools-on", "Enable agent tools", "Agent"),
    CommandSpec("/tools-off", "/tools-off", "Disable agent tools", "Agent"),
    CommandSpec("/skills", "/skills [list|reload|path|show NAME|enable NAME|disable NAME]", "Manage reusable procedural skills", "Agent"),
    CommandSpec("/skill", "/skill NAME [TASK]", "Load one skill for a model turn", "Agent"),
    CommandSpec("/retry", "/retry", "Replay the previous safe user turn", "Agent"),
    CommandSpec("/undo", "/undo [N]", "Remove recent conversation turns", "Sessions"),
    CommandSpec("/verify", "/verify [auto|recipes|status|RECIPE]", "Run sandbox verification and record evidence", "Tools"),
    CommandSpec("/delegate", "/delegate TASK | /delegate stop ID", "Run or stop isolated read-only agent work", "Agent"),
    CommandSpec("/delegates", "/delegates [ID]", "Show delegated work and evidence", "Agent"),
    CommandSpec("/tool-auto", "/tool-auto on|off", "Set proactive local routing", "Agent"),
    CommandSpec("/plan", "/plan [show|add STEP|set ID STATUS|clear]", "Manage session task plan", "Agent"),
    CommandSpec("/react", "/react [on|off|status]", "Control host-managed ReAct harness", "Agent"),
    CommandSpec("/react-trace", "/react-trace [on|off|status]", "Show or hide detailed ReAct steps", "Agent"),
    CommandSpec("/think", "/think PROMPT", "Send with supported reasoning mode", "Agent"),
    CommandSpec("/thinking", "/thinking off|low|medium|high|status", "Set native reasoning effort", "Agent"),
    CommandSpec("/web", "/web on|off|always|ask", "Set web access policy", "Tools"),
    CommandSpec("/search", "/search fast|deep|status", "Set default web search depth", "Tools"),
    CommandSpec("/sandbox", "/sandbox [on|codex|docker|e2b|status|off]", "Control isolated command backend", "Tools"),
    CommandSpec("/permissions", "/permissions [reset]", "Show workspace permissions", "Tools"),
    CommandSpec("/history", "/history", "Show recent agent history", "Sessions"),
    CommandSpec("/new", "/new", "Start clean session", "Sessions", ("/newchat",)),
    CommandSpec("/session-name", "/session-name TEXT", "Name current session", "Sessions"),
    CommandSpec("/memory", "/memory [search QUERY|clear|notes|forget|list|current|records|correct|delete|export]", "Manage session memory", "Sessions", ("/mem",)),
    CommandSpec("/remember", "/remember TEXT", "Save durable user note", "Sessions"),
    CommandSpec("/harness", "/harness [status|mode legacy|mode v2|runs|reconcile RUN_ID|resume RUN_ID|debug RUN_ID]", "Inspect or select durable harness state", "Sessions"),
    CommandSpec("/model", "/model [NAME]", "Select local or saved model", "Models"),
    CommandSpec("/model-add", "/model-add", "Add GGUF model profile", "Models", ("/modeladd",)),
    CommandSpec("/model-rm", "/model-rm", "Remove saved model profile", "Models", ("/modelrm",)),
    CommandSpec("/api", "/api", "Connect API model provider", "Models"),
    CommandSpec("/api-md", "/api-md", "Change hosted model", "Models"),
    CommandSpec("/api-del", "/api-del", "Remove API profile", "Models"),
    CommandSpec("/paste", "/paste", "Toggle multiline paste mode", "Input", ("/multiline",)),
    CommandSpec("/clear", "/clear", "Clear visible conversation", "General", ("/cls",)),
    CommandSpec("/endserver", "/endserver", "Unload local model server", "Models"),
    CommandSpec("/exit", "/exit", "Save and exit", "General", ("/quit", "/q")),
)


_COMPLETION_VARIANTS: dict[str, tuple[tuple[str, str], ...]] = {
    "/search": (
        ("fast", "Use compact top-result web search"),
        ("deep", "Use bounded multi-source web research"),
        ("status", "Show selected web-search depth"),
    ),
    "/harness": (
        ("status", "Show active harness state"),
        ("mode legacy", "Use the compatibility harness for the next turn"),
        ("mode v2", "Use classified-turn harness v2 for the next turn"),
        ("runs", "Show recoverable durable runs"),
    ),
}


def match_commands(query: str, *, limit: int = 8) -> tuple[CommandSpec, ...]:
    """Return stable filtered command metadata for slash autocomplete."""
    needle = query.casefold().strip()
    for command, variants in _COMPLETION_VARIANTS.items():
        if needle.startswith(command):
            items = tuple(
                CommandSpec(
                    f"{command} {value}",
                    f"{command} {value}",
                    description,
                    "Tools",
                )
                for value, description in variants
                if not needle or f"{command} {value}".startswith(needle)
            )
            return items[:limit]
    matches: Iterable[CommandSpec] = (spec for spec in COMMAND_SPECS if spec.matches(query))
    return tuple(list(matches)[:limit])


def command_usage_for_input(value: str) -> tuple[str, bool]:
    """Return canonical usage and whether the slash command name is known."""
    token = value.strip().split(maxsplit=1)[0].casefold() if value.strip() else "/"
    by_name = {
        name.casefold(): spec
        for spec in COMMAND_SPECS
        for name in (spec.command, *spec.aliases)
    }
    if token in by_name:
        return by_name[token].usage, True
    candidates = tuple(spec.command for spec in COMMAND_SPECS)
    closest = get_close_matches(token, candidates, n=1, cutoff=0.5)
    if closest:
        spec = next(item for item in COMMAND_SPECS if item.command == closest[0])
        return spec.usage, False
    return "/help", False


__all__ = [
    "COMMAND_SPECS", "CommandSpec", "command_usage_for_input", "match_commands",
]
