"""UI-neutral events shared by terminal and Textual renderers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class AgentEvent:
    """One observable event from an FenrirAgent agent turn."""

    type: str
    content: str = ""
    name: str = ""
    arguments: Mapping[str, Any] = field(default_factory=dict)
    summary: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    details: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1
    event_id: str = ""
    run_id: str = ""
    turn_id: str = ""
    step_id: str = ""

    @classmethod
    def from_chunk(cls, chunk: Mapping[str, Any]) -> "AgentEvent":
        return cls(
            type=str(chunk.get("type", "status")),
            schema_version=int(chunk.get("schema_version", 1)),
            event_id=str(chunk.get("event_id", "")),
            run_id=str(chunk.get("run_id", "")),
            turn_id=str(chunk.get("turn_id", "")),
            step_id=str(chunk.get("step_id", "")),
            content=str(chunk.get("content", "")),
            name=str(chunk.get("name", "")),
            arguments=chunk.get("arguments", {}) if isinstance(chunk.get("arguments", {}), Mapping) else {},
            summary=str(chunk.get("summary", "")),
            input_tokens=chunk.get("input_tokens") if isinstance(chunk.get("input_tokens"), int) else None,
            output_tokens=chunk.get("output_tokens") if isinstance(chunk.get("output_tokens"), int) else None,
            details=chunk.get("details", {}) if isinstance(chunk.get("details", {}), Mapping) else {},
        )


__all__ = ["AgentEvent"]
