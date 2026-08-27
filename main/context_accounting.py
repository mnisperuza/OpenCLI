"""Model-aware prompt estimates and session usage accounting."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional

from .model_profiles import ModelCapabilityProfile


def tiktoken_counter(
    model_id: str,
    encoding_name: Optional[str] = None,
) -> Optional[Callable[[str], int]]:
    """Return an exact tiktoken counter only for a known model or encoding."""
    try:
        import tiktoken
    except ImportError:
        return None
    try:
        encoding = (
            tiktoken.get_encoding(encoding_name)
            if encoding_name
            else tiktoken.encoding_for_model(model_id)
        )
    except (KeyError, ValueError):
        return None

    def count(text: str) -> int:
        return len(encoding.encode(text, disallowed_special=()))

    return count


@dataclass(frozen=True)
class ContextSnapshot:
    profile: ModelCapabilityProfile
    components: Mapping[str, int]
    used_tokens: int
    output_reserve: int
    available_tokens: int
    percent_used: float
    estimated: bool


@dataclass
class SessionUsage:
    turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    last_input_tokens: int = 0
    last_output_tokens: int = 0
    estimated_turns: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class ContextAccountingService:
    """Estimate context occupancy and retain session token totals."""

    def __init__(
        self,
        profile: ModelCapabilityProfile,
        tokenizer: Optional[Callable[[str], int]] = None,
    ):
        self.profile = profile
        self.tokenizer = tokenizer
        self.usage = SessionUsage()

    def set_profile(
        self,
        profile: ModelCapabilityProfile,
        tokenizer: Optional[Callable[[str], int]] = None,
    ) -> None:
        self.profile = profile
        self.tokenizer = tokenizer

    def reset_usage(self) -> None:
        self.usage = SessionUsage()

    def count_text(self, text: str) -> tuple[int, bool]:
        if not text:
            return 0, self.tokenizer is None
        if self.tokenizer is not None:
            try:
                return max(0, int(self.tokenizer(text))), False
            except (TypeError, ValueError, RuntimeError):
                pass
        return max(1, math.ceil(len(text.encode("utf-8")) / 4)), True

    def snapshot(
        self,
        components: Mapping[str, str],
        *,
        output_reserve: Optional[int] = None,
    ) -> ContextSnapshot:
        counts: Dict[str, int] = {}
        estimated = False
        for name, text in components.items():
            count, item_estimated = self.count_text(str(text or ""))
            counts[name] = count
            estimated = estimated or item_estimated
        used = sum(counts.values())
        safe_reserve = max(64, self.profile.context_window // 2)
        reserve = min(
            output_reserve or self.profile.max_output_tokens,
            safe_reserve,
        )
        available = max(0, self.profile.context_window - used - reserve)
        percent = min(100.0, used * 100.0 / self.profile.context_window)
        return ContextSnapshot(
            profile=self.profile,
            components=counts,
            used_tokens=used,
            output_reserve=reserve,
            available_tokens=available,
            percent_used=percent,
            estimated=estimated,
        )

    def record_turn(
        self,
        input_tokens: int,
        output_tokens: int,
        *,
        estimated: bool,
    ) -> None:
        self.usage.turns += 1
        self.usage.last_input_tokens = max(0, int(input_tokens))
        self.usage.last_output_tokens = max(0, int(output_tokens))
        self.usage.input_tokens += self.usage.last_input_tokens
        self.usage.output_tokens += self.usage.last_output_tokens
        if estimated:
            self.usage.estimated_turns += 1


def format_token_count(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}m"
    if value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return str(value)


__all__ = [
    "ContextAccountingService",
    "ContextSnapshot",
    "SessionUsage",
    "format_token_count",
    "tiktoken_counter",
]
