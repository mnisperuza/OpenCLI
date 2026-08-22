"""UI-neutral events shared by terminal and Textual renderers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class AgentEvent:
    """One observable event from an OpenCLI agent turn."""

    type: str
    content: str = ""
    name: str = ""
    arguments: Mapping[str, Any] = field(default_factory=dict)
    summary: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_chunk(cls, chunk: Mapping[str, Any]) -> "AgentEvent":
        return cls(
            type=str(chunk.get("type", "status")),
            content=str(chunk.get("content", "")),
            name=str(chunk.get("name", "")),
            arguments=chunk.get("arguments", {}) if isinstance(chunk.get("arguments", {}), Mapping) else {},
            summary=str(chunk.get("summary", "")),
            input_tokens=chunk.get("input_tokens") if isinstance(chunk.get("input_tokens"), int) else None,
            output_tokens=chunk.get("output_tokens") if isinstance(chunk.get("output_tokens"), int) else None,
            details=chunk.get("details", {}) if isinstance(chunk.get("details", {}), Mapping) else {},
        )


__all__ = ["AgentEvent"]
