"""Provider capability reporting, bounded transport retry, and circuit breaking."""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, TypeVar
from urllib.error import HTTPError, URLError


T = TypeVar("T")


@dataclass(frozen=True)
class ProviderCapabilities:
    provider: str
    model: str
    native_tools: bool = True
    named_tool_choice: Optional[bool] = None
    parallel_tools: Optional[bool] = None
    strict_json_schema: Optional[bool] = None
    grammar_constrained: Optional[bool] = None
    streaming_tool_arguments: Optional[bool] = None
    context_window: Optional[int] = None
    max_output_tokens: Optional[int] = None
    tokenizer: str = "unknown"
    reasoning: Optional[bool] = None
    vision: Optional[bool] = None
    cancellation: Optional[bool] = None
    observed_failures: int = 0
    checked_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class TransportRetryPolicy:
    max_attempts: int = 3
    base_delay_seconds: float = 0.4
    max_delay_seconds: float = 8.0
    jitter_seconds: float = 0.25


class ProviderCircuitOpen(RuntimeError):
    pass


@dataclass(frozen=True)
class FallbackCandidate:
    provider: str
    model: str
    capabilities: ProviderCapabilities
    permits_sensitive_context: bool = False


class ModelFallbackPolicy:
    """Choose only configured candidates that satisfy capability and data policy."""

    def __init__(
        self, candidates: tuple[FallbackCandidate, ...] = (), *, max_hops: int = 2
    ):
        self.candidates = candidates[: max(0, int(max_hops))]

    def choose(
        self,
        *,
        attempted: set[tuple[str, str]],
        requires_tools: bool,
        requires_vision: bool,
        sensitive_context: bool,
    ) -> Optional[FallbackCandidate]:
        for candidate in self.candidates:
            identity = (candidate.provider, candidate.model)
            if identity in attempted:
                continue
            capabilities = candidate.capabilities
            if requires_tools and not capabilities.native_tools:
                continue
            if requires_vision and capabilities.vision is not True:
                continue
            if sensitive_context and not candidate.permits_sensitive_context:
                continue
            return candidate
        return None


class ProviderReliabilityController:
    """Separate transport retries from agent/tool repair budgets."""

    def __init__(
        self,
        retry_policy: Optional[TransportRetryPolicy] = None,
        *,
        failure_threshold: int = 4,
        cooldown_seconds: float = 30.0,
    ):
        self.policy = retry_policy or TransportRetryPolicy()
        self.failure_threshold = max(1, int(failure_threshold))
        self.cooldown_seconds = max(1.0, float(cooldown_seconds))
        self._consecutive_failures = 0
        self._opened_at: Optional[float] = None
        self._lock = threading.RLock()

    @staticmethod
    def _retryable(error: Exception) -> bool:
        if isinstance(error, HTTPError):
            return error.code == 429 or 500 <= error.code < 600
        return isinstance(error, (TimeoutError, URLError, ConnectionError, OSError))

    @staticmethod
    def _retry_after(error: Exception) -> Optional[float]:
        if not isinstance(error, HTTPError):
            return None
        value = error.headers.get("Retry-After") if error.headers else None
        if value is None:
            return None
        try:
            return max(0.0, min(float(value), 60.0))
        except (TypeError, ValueError):
            return None

    def _before_call(self) -> None:
        with self._lock:
            if self._opened_at is None:
                return
            if time.monotonic() - self._opened_at >= self.cooldown_seconds:
                self._opened_at = None
                self._consecutive_failures = 0
                return
            remaining = self.cooldown_seconds - (time.monotonic() - self._opened_at)
            raise ProviderCircuitOpen(
                f"Provider circuit is open; retry after {remaining:.1f} seconds"
            )

    def _success(self) -> None:
        with self._lock:
            self._consecutive_failures = 0
            self._opened_at = None

    def _failure(self) -> None:
        with self._lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.failure_threshold:
                self._opened_at = time.monotonic()

    def call(self, operation: Callable[[], T]) -> T:
        self._before_call()
        attempts = max(1, int(self.policy.max_attempts))
        for attempt in range(1, attempts + 1):
            try:
                result = operation()
            except Exception as error:
                self._failure()
                if attempt >= attempts or not self._retryable(error):
                    raise
                delay = self._retry_after(error)
                if delay is None:
                    delay = min(
                        self.policy.max_delay_seconds,
                        self.policy.base_delay_seconds * (2 ** (attempt - 1)),
                    )
                    delay += random.uniform(0, self.policy.jitter_seconds)
                time.sleep(delay)
                continue
            self._success()
            return result
        raise RuntimeError("Unreachable retry state")

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "consecutive_failures": self._consecutive_failures,
                "circuit_open": self._opened_at is not None,
                "failure_threshold": self.failure_threshold,
                "cooldown_seconds": self.cooldown_seconds,
                "max_transport_attempts": self.policy.max_attempts,
            }


__all__ = [
    "ProviderCapabilities",
    "ProviderCircuitOpen",
    "ProviderReliabilityController",
    "TransportRetryPolicy",
    "FallbackCandidate",
    "ModelFallbackPolicy",
]
