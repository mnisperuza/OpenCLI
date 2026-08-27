"""Deterministic guardrails around Pydantic AI model/tool cycles.

Models choose actions. Code owns budgets, repeated-action detection, failure
limits, and evidence. Private chain-of-thought is neither requested nor stored.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping


class ReactLoopLimitError(RuntimeError):
    """Raised before another unsafe or unproductive tool action can execute."""


class ReactPhase(str, Enum):
    """Host-owned phases. Models may propose data, never arbitrary transitions."""

    IDLE = "idle"
    DISPATCH = "dispatch"
    PLAN = "plan"
    ACT = "act"
    OBSERVE = "observe"
    CRITIQUE = "critique"
    FINISH = "finish"
    ASK_USER = "ask_user"
    HALTED = "halted"


@dataclass(frozen=True)
class ReactLoopPolicy:
    max_steps: int = 10
    max_repeated_action: int = 2
    max_consecutive_failures: int = 3
    single_action_per_model_step: bool = True
    max_timeline_events: int = 30


@dataclass(frozen=True)
class ReactCritique:
    """Small public reflection record; deliberately excludes hidden reasoning."""

    progress: str
    evidence: tuple[str, ...] = ()
    blocker: str = ""
    next_action: str = ""
    complete: bool = False
    needs_user: bool = False


@dataclass(frozen=True)
class ReactTimelineEvent:
    step: int
    phase: str
    summary: str


@dataclass
class ReactLoopState:
    goal: str = ""
    requested: bool = False
    paths: tuple[str, ...] = ()
    max_steps: int = 0
    steps: int = 0
    consecutive_failures: int = 0
    action_counts: Dict[str, int] = field(default_factory=dict)
    last_tool: str = ""
    halted_reason: str = ""
    phase: ReactPhase = ReactPhase.IDLE
    timeline: list[ReactTimelineEvent] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    critique: ReactCritique | None = None


class ReactLoopController:
    """Observe tool events and enforce one bounded ReAct turn."""

    _FAILURE_MARKERS = (
        "error", "failed", "denied", "unavailable", "timed out",
        "not found", "unchanged", "conflict",
    )

    def __init__(self, policy: ReactLoopPolicy | None = None):
        self.policy = policy or ReactLoopPolicy()
        self.enabled = True
        self.state = ReactLoopState()

    def begin_turn(self, goal: str) -> None:
        """Reset one user turn at the dispatcher boundary."""
        phase = ReactPhase.DISPATCH if self.enabled else ReactPhase.IDLE
        self.state = ReactLoopState(goal=self._clean(goal, 500), phase=phase)
        self._record(phase, "User turn opened")

    @staticmethod
    def _clean(value: Any, limit: int = 500) -> str:
        return " ".join(str(value or "").split())[:limit]

    def _record(self, phase: ReactPhase, summary: str) -> None:
        event = ReactTimelineEvent(
            step=self.state.steps,
            phase=phase.value,
            summary=self._clean(summary),
        )
        self.state.timeline.append(event)
        overflow = len(self.state.timeline) - self.policy.max_timeline_events
        if overflow > 0:
            del self.state.timeline[:overflow]

    def dispatch(self, decision: str, *, summary: str = "") -> Dict[str, Any]:
        """Apply a validated every-turn routing decision."""
        if self.state.phase != ReactPhase.DISPATCH:
            raise ValueError("ReAct dispatch is only valid at the start of a turn.")
        route = self._clean(decision, 20).casefold()
        phases = {
            "answer": ReactPhase.FINISH,
            "act": ReactPhase.PLAN,
            "ask_user": ReactPhase.ASK_USER,
        }
        if route not in phases:
            raise ValueError("ReAct dispatch must be answer, act, or ask_user.")
        self.state.phase = phases[route]
        self._record(self.state.phase, summary or f"Dispatch: {route}")
        return self.status()

    def start_task(
        self,
        goal: str,
        *,
        paths: tuple[str, ...] = (),
        max_steps: int | None = None,
    ) -> Dict[str, Any]:
        """Start a bounded model-requested task without granting any permission."""
        if self.state.requested:
            raise ValueError(
                "A ReAct task is already active for this turn; continue it or finish."
            )
        cleaned_goal = self._clean(goal, 500)
        if not cleaned_goal:
            raise ValueError("ReAct goal cannot be empty.")
        requested_limit = self.policy.max_steps if max_steps is None else int(max_steps)
        if requested_limit < 1:
            raise ValueError("ReAct max_steps must be at least 1.")
        limit = min(requested_limit, self.policy.max_steps)
        timeline = list(self.state.timeline)
        self.state = ReactLoopState(
            goal=cleaned_goal,
            requested=True,
            paths=paths,
            max_steps=limit,
            phase=ReactPhase.PLAN,
            timeline=timeline,
        )
        self._record(ReactPhase.PLAN, "Task started; create or inspect plan")
        return self.status()

    def _step_limit(self) -> int:
        return self.state.max_steps or self.policy.max_steps

    def instruction_block(self) -> str:
        """Public workflow template; asks for actions/evidence, never hidden reasoning."""
        if not self.enabled:
            return "ReAct controller is disabled; tool permissions and mutation evidence still apply."
        return (
            "REACT WORKFLOW: Every turn starts with react_dispatch. Choose answer "
            "for direct conversation, act when external evidence/actions are needed, "
            "or ask_user only for a real blocker. An act route must include a concise "
            "goal and may include relevant workspace paths and a requested step "
            "budget; it grants no permissions and OpenCLI caps it. Once active, "
            "select exactly one useful tool. After its observation, call "
            "critique_and_plan with public progress, evidence, blocker, and next "
            "action or completion state. Repeat or finish. Never reveal or request "
            "private chain-of-thought. Limits: "
            f"{self.policy.max_steps} tool steps, identical action at most "
            f"{self.policy.max_repeated_action} times, "
            f"{self.policy.max_consecutive_failures} consecutive failures."
        )

    @staticmethod
    def _fingerprint(name: str, arguments: Any) -> str:
        payload = json.dumps(arguments, sort_keys=True, default=str, separators=(",", ":"))
        return hashlib.sha256(f"{name}\0{payload}".encode("utf-8")).hexdigest()

    def before_tool(self, name: str, arguments: Any) -> int:
        if not self.enabled or not self.state.requested:
            return 0
        if self.state.halted_reason:
            raise ReactLoopLimitError(self.state.halted_reason)
        if self.state.phase not in {ReactPhase.PLAN, ReactPhase.ACT}:
            raise ReactLoopLimitError(
                f"ReAct cannot run a tool during phase: {self.state.phase.value}."
            )
        step_limit = self._step_limit()
        if self.state.steps >= step_limit:
            self.state.halted_reason = (
                f"ReAct stopped after {step_limit} tool steps. "
                "Refine the request or continue in a new turn."
            )
            self.state.phase = ReactPhase.HALTED
            self._record(ReactPhase.HALTED, self.state.halted_reason)
            raise ReactLoopLimitError(self.state.halted_reason)
        fingerprint = self._fingerprint(name, arguments)
        count = self.state.action_counts.get(fingerprint, 0) + 1
        self.state.action_counts[fingerprint] = count
        if count > self.policy.max_repeated_action:
            self.state.halted_reason = (
                f"ReAct stopped repeated action: {name}. Choose new evidence or ask user."
            )
            self.state.phase = ReactPhase.HALTED
            self._record(ReactPhase.HALTED, self.state.halted_reason)
            raise ReactLoopLimitError(self.state.halted_reason)
        self.state.steps += 1
        self.state.last_tool = name
        self.state.phase = ReactPhase.ACT
        self._record(ReactPhase.ACT, f"Tool: {name}")
        return self.state.steps

    def after_tool(self, event: Dict[str, Any]) -> None:
        if not self.enabled or not self.state.requested:
            return
        raw_summary = self._clean(event.get("summary", ""), 500)
        summary = raw_summary.casefold()
        self.state.phase = ReactPhase.OBSERVE
        self._record(ReactPhase.OBSERVE, raw_summary or "Tool returned no summary")
        if raw_summary:
            self.state.evidence.append(raw_summary)
            self.state.evidence = self.state.evidence[-8:]
        nonzero_exit = summary.startswith("exit ") and summary != "exit 0"
        failed = bool(event.get("error")) or nonzero_exit or any(
            marker in summary for marker in self._FAILURE_MARKERS
        )
        self.state.consecutive_failures = (
            self.state.consecutive_failures + 1 if failed else 0
        )
        if self.state.consecutive_failures >= self.policy.max_consecutive_failures:
            self.state.halted_reason = (
                f"ReAct stopped after {self.policy.max_consecutive_failures} consecutive "
                "tool failures. Change approach or request user input."
            )
            self.state.phase = ReactPhase.HALTED
            self._record(ReactPhase.HALTED, self.state.halted_reason)
        else:
            self.state.phase = ReactPhase.CRITIQUE
            self._record(ReactPhase.CRITIQUE, "Critique progress and choose next transition")

    def submit_critique(self, critique: ReactCritique | Mapping[str, Any]) -> Dict[str, Any]:
        """Validate bounded self-reflection and deterministically select next phase."""
        if self.state.phase != ReactPhase.CRITIQUE:
            raise ValueError("Critique is only valid after observing a tool result.")
        if isinstance(critique, Mapping):
            critique = ReactCritique(
                progress=self._clean(critique.get("progress"), 500),
                evidence=tuple(
                    self._clean(item, 300)
                    for item in critique.get("evidence", ())
                    if self._clean(item, 300)
                )[:8],
                blocker=self._clean(critique.get("blocker"), 300),
                next_action=self._clean(critique.get("next_action"), 300),
                complete=bool(critique.get("complete", False)),
                needs_user=bool(critique.get("needs_user", False)),
            )
        if not critique.progress:
            raise ValueError("Critique progress cannot be empty.")
        if critique.complete and critique.needs_user:
            raise ValueError("Critique cannot be complete and need user input.")
        if not critique.complete and not critique.needs_user and not critique.next_action:
            raise ValueError("An unfinished critique requires next_action.")
        self.state.critique = critique
        if critique.complete:
            next_phase = ReactPhase.FINISH
        elif critique.needs_user:
            next_phase = ReactPhase.ASK_USER
        else:
            next_phase = ReactPhase.ACT
        self.state.phase = next_phase
        self._record(next_phase, critique.progress)
        return self.status()

    def fallback_to_user(self, reason: str) -> Dict[str, Any]:
        """End model autonomy safely after repeated structured-output failure."""
        cleaned = self._clean(reason, 500) or "Structured ReAct decision failed."
        self.state.phase = ReactPhase.ASK_USER
        self.state.halted_reason = cleaned
        self._record(ReactPhase.ASK_USER, cleaned)
        return self.status()

    def loop_context(self) -> Dict[str, Any]:
        """Public, compact step context suitable for reinjection into model prompts."""
        critique = asdict(self.state.critique) if self.state.critique else None
        return {
            "goal": self.state.goal,
            "phase": self.state.phase.value,
            "step": self.state.steps,
            "max_steps": self._step_limit(),
            "remaining_steps": max(0, self._step_limit() - self.state.steps),
            "consecutive_failures": self.state.consecutive_failures,
            "last_tool": self.state.last_tool,
            "recent_evidence": list(self.state.evidence),
            "critique": critique,
            "halted_reason": self.state.halted_reason,
        }

    def status(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "requested": self.state.requested,
            "phase": self.state.phase.value,
            "goal": self.state.goal,
            "paths": list(self.state.paths),
            "max_steps": self._step_limit(),
            "steps": self.state.steps,
            "failures": self.state.consecutive_failures,
            "last_tool": self.state.last_tool,
            "halted_reason": self.state.halted_reason,
            "timeline": [asdict(event) for event in self.state.timeline],
            "critique": asdict(self.state.critique) if self.state.critique else None,
            "single_action_per_model_step": self.policy.single_action_per_model_step,
        }


__all__ = [
    "ReactCritique", "ReactLoopController", "ReactLoopLimitError",
    "ReactLoopPolicy", "ReactLoopState", "ReactPhase", "ReactTimelineEvent",
]
