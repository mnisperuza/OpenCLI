"""Bounded structured-output ladder for weak and local model backends."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional, Protocol


class StructuredOutputMode(str, Enum):
    PROVIDER_NATIVE = "provider_native"
    BACKEND_GRAMMAR = "backend_grammar"
    OUTLINES = "outlines"
    STRICT_JSON = "strict_json"
    REPAIRED_JSON = "repaired_json"
    FAILED = "failed"


@dataclass(frozen=True)
class StructuredOutputResult:
    value: Any
    mode: StructuredOutputMode
    repaired: bool = False
    error: str = ""


class ConstrainedDecoder(Protocol):
    name: str

    def available(self) -> bool: ...

    def generate(self, prompt: str, schema: Mapping[str, Any]) -> str: ...


class StructuredOutputLadder:
    """Strict parser plus conservative syntax repair; never evaluates model code."""

    _TRAILING_COMMA = re.compile(r",\s*([}\]])")

    def __init__(
        self, decoder: Optional[ConstrainedDecoder] = None, *, max_chars: int = 100_000
    ):
        self.decoder = decoder
        self.max_chars = max(1_000, int(max_chars))

    @staticmethod
    def _unfence(text: str) -> str:
        value = text.strip()
        if value.startswith("```") and value.endswith("```"):
            lines = value.splitlines()
            if len(lines) >= 3:
                return "\n".join(lines[1:-1]).strip()
        return value

    @classmethod
    def _repair(cls, text: str) -> str:
        repaired = cls._unfence(text).strip()
        repaired = cls._TRAILING_COMMA.sub(r"\1", repaired)
        return repaired

    def parse_json(self, text: str) -> StructuredOutputResult:
        bounded = str(text)[: self.max_chars]
        strict = self._unfence(bounded)
        try:
            return StructuredOutputResult(
                json.loads(strict), StructuredOutputMode.STRICT_JSON
            )
        except json.JSONDecodeError as first_error:
            repaired = self._repair(strict)
            if repaired == strict:
                return StructuredOutputResult(
                    None, StructuredOutputMode.FAILED, error=str(first_error)
                )
            try:
                return StructuredOutputResult(
                    json.loads(repaired),
                    StructuredOutputMode.REPAIRED_JSON,
                    repaired=True,
                )
            except json.JSONDecodeError as error:
                return StructuredOutputResult(
                    None, StructuredOutputMode.FAILED, error=str(error)
                )

    def generate(
        self, prompt: str, schema: Mapping[str, Any]
    ) -> StructuredOutputResult:
        if self.decoder is None or not self.decoder.available():
            return StructuredOutputResult(
                None,
                StructuredOutputMode.FAILED,
                error="No constrained decoder available",
            )
        try:
            raw = self.decoder.generate(prompt, schema)
        except Exception as error:
            return StructuredOutputResult(
                None, StructuredOutputMode.FAILED, error=type(error).__name__
            )
        parsed = self.parse_json(raw)
        if parsed.mode == StructuredOutputMode.FAILED:
            return parsed
        mode = (
            StructuredOutputMode.OUTLINES
            if self.decoder.name == "outlines"
            else StructuredOutputMode.BACKEND_GRAMMAR
        )
        return StructuredOutputResult(parsed.value, mode, parsed.repaired)


__all__ = [
    "ConstrainedDecoder",
    "StructuredOutputLadder",
    "StructuredOutputMode",
    "StructuredOutputResult",
]
