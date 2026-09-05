"""Privacy-first OpenTelemetry facade with a dependency-free fallback."""

from __future__ import annotations

import hashlib
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Mapping, Optional

from .harness_contracts import SecretRedactor


@dataclass
class LocalSpan:
    name: str
    attributes: Dict[str, Any]
    started_at: float = field(default_factory=time.monotonic)
    ended_at: Optional[float] = None
    status: str = "ok"

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[str(key)] = value

    def record_exception(self, error: Exception) -> None:
        self.status = "error"
        self.attributes["error.type"] = type(error).__name__

    def end(self) -> None:
        if self.ended_at is None:
            self.ended_at = time.monotonic()
            self.attributes["duration_ms"] = round(
                (self.ended_at - self.started_at) * 1000, 3
            )


class HarnessTelemetry:
    """Central semantic mapping; prompt/tool content is excluded by default."""

    def __init__(self, *, enabled: bool = False, include_content: bool = False):
        self.enabled = bool(enabled)
        self.include_content = bool(include_content)
        self.completed: list[LocalSpan] = []
        self._tracer: Any = None
        if self.enabled:
            try:
                from opentelemetry import trace

                self._tracer = trace.get_tracer("fenrir.harness")
            except ImportError:
                self._tracer = None

    def safe_attributes(
        self, attributes: Optional[Mapping[str, Any]]
    ) -> Dict[str, Any]:
        cleaned: Dict[str, Any] = {}
        for key, value in (attributes or {}).items():
            name = str(key)
            if any(
                marker in name.casefold()
                for marker in (
                    "prompt",
                    "content",
                    "arguments",
                    "output",
                    "secret",
                    "token",
                )
            ):
                rendered = str(value)
                if self.include_content:
                    redacted, _ = SecretRedactor.redact_text(rendered)
                    cleaned[name] = redacted
                else:
                    cleaned[name + ".sha256"] = hashlib.sha256(
                        rendered.encode("utf-8")
                    ).hexdigest()
                    cleaned[name + ".chars"] = len(rendered)
                continue
            if isinstance(value, (str, bool, int, float)):
                redacted, _ = SecretRedactor.redact_text(str(value))
                cleaned[name] = redacted if isinstance(value, str) else value
        return cleaned

    def start_span(
        self, name: str, attributes: Optional[Mapping[str, Any]] = None
    ) -> Any:
        safe = self.safe_attributes(attributes)
        if self._tracer is not None:
            span = self._tracer.start_span(name, attributes=safe)
            return span
        return LocalSpan(name=name, attributes=safe)

    def end_span(self, span: Any, error: Optional[Exception] = None) -> None:
        if error is not None:
            try:
                span.record_exception(error)
            except AttributeError:
                pass
        span.end()
        if isinstance(span, LocalSpan):
            self.completed.append(span)

    @contextmanager
    def span(
        self, name: str, attributes: Optional[Mapping[str, Any]] = None
    ) -> Iterator[Any]:
        active = self.start_span(name, attributes)
        try:
            yield active
        except Exception as error:
            self.end_span(active, error)
            raise
        else:
            self.end_span(active)

    def metrics(self) -> Dict[str, Any]:
        durations = [span.attributes.get("duration_ms", 0.0) for span in self.completed]
        return {
            "span_count": len(self.completed),
            "error_count": sum(span.status == "error" for span in self.completed),
            "duration_ms": round(sum(float(value) for value in durations), 3),
            "content_tracing": self.include_content,
        }


__all__ = ["HarnessTelemetry", "LocalSpan"]
