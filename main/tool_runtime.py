"""Enterprise tool registry, policy checks, progress, and completion controls."""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from .harness_contracts import (
    CapabilityClass,
    CompletionDecision,
    ErrorCode,
    RiskLevel,
    ToolManifest,
    ToolOutcome,
    ToolStatus,
    content_hash,
)


def evidence_id(tool_name: str, value: Any) -> str:
    return f"evidence_{tool_name}_{content_hash(value)[:24]}"


def mutation_receipt(
    resource: str,
    *,
    pre_hash: Optional[str],
    post_hash: Optional[str],
    verified: bool,
) -> Dict[str, Any]:
    return {
        "resource": resource,
        "pre_hash": pre_hash,
        "post_hash": post_hash,
        "verified": bool(verified),
    }


class ToolRegistry:
    """Stable manifests kept separate from provider-generated schemas."""

    def __init__(self, manifests: Iterable[ToolManifest] = ()):
        self._items: Dict[str, ToolManifest] = {}
        for manifest in manifests:
            self.register(manifest)

    def register(self, manifest: ToolManifest) -> None:
        prior = self._items.get(manifest.name)
        if prior is not None and prior.version != manifest.version:
            raise ValueError(
                f"Tool {manifest.name!r} already registered at {prior.version}"
            )
        self._items[manifest.name] = manifest

    def get(self, name: str) -> ToolManifest:
        try:
            return self._items[name]
        except KeyError as error:
            raise KeyError(f"Unregistered tool: {name}") from error

    def as_dict(self) -> Dict[str, Dict[str, Any]]:
        return {
            name: manifest.model_dump(mode="json")
            for name, manifest in sorted(self._items.items())
        }

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))


@dataclass(frozen=True)
class Toolset:
    """Named capability group exposed through the command-only tool controls."""

    name: str
    description: str
    tools: tuple[str, ...]


class ToolsetRegistry:
    """Host-owned tool grouping; models cannot enable capabilities themselves."""

    def __init__(self, toolsets: Iterable[Toolset] = ()):
        self._items: Dict[str, Toolset] = {}
        for toolset in toolsets:
            name = toolset.name.casefold().strip()
            if not name or name in self._items:
                raise ValueError(f"Invalid or duplicate toolset: {toolset.name!r}")
            self._items[name] = Toolset(name, toolset.description, toolset.tools)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))

    def normalize(self, enabled: Iterable[str]) -> tuple[str, ...]:
        requested = tuple(dict.fromkeys(str(item).casefold().strip() for item in enabled))
        unknown = sorted(set(requested) - set(self._items))
        if unknown:
            raise ValueError(f"Unknown toolset(s): {', '.join(unknown)}")
        return tuple(name for name in self.names if name in requested)

    def enabled_tools(self, enabled: Iterable[str]) -> frozenset[str]:
        active = self.normalize(enabled)
        return frozenset(
            tool for name in active for tool in self._items[name].tools
        )

    def status(self, enabled: Iterable[str]) -> Dict[str, Dict[str, Any]]:
        active = set(self.normalize(enabled))
        return {
            name: {
                "enabled": name in active,
                "description": toolset.description,
                "tools": toolset.tools,
            }
            for name, toolset in sorted(self._items.items())
        }


def default_toolset_registry() -> ToolsetRegistry:
    """Default capability groups, inspired by Hermes toolsets but OpenCLI-owned."""
    return ToolsetRegistry(
        (
            Toolset(
                "workspace",
                "Inspect and modify files inside the approved workspace.",
                (
                    "get_working_directory", "set_working_directory",
                    "list_allowed_roots", "list_files", "read_text_file",
                    "search_text", "file_info", "write_text_file",
                    "edit_text_file", "create_directory",
                ),
            ),
            Toolset(
                "web", "Fast search, deep research, and approved source fetches.",
                ("web_search", "web_fetch"),
            ),
            Toolset(
                "memory", "Recall trusted facts from prior OpenCLI sessions.",
                ("search_memory",),
            ),
            Toolset(
                "planning", "Use ReAct dispatch and persistent task plans.",
                (
                    "react_dispatch", "critique_and_plan", "get_task_plan",
                    "create_task_plan", "add_task_plan_item",
                    "update_task_plan_item",
                ),
            ),
            Toolset(
                "sandbox", "Run approved commands in an isolated backend.",
                ("get_sandbox_status", "run_sandboxed_command"),
            ),
            Toolset(
                "session", "Maintain the current session title.",
                ("set_session_title",),
            ),
        )
    )


DEFAULT_TOOLSETS = default_toolset_registry().names


def default_tool_registry(max_output_chars: int = 20_000) -> ToolRegistry:
    def manifest(
        name: str,
        capability: CapabilityClass,
        *,
        risk: RiskLevel = RiskLevel.LOW,
        approval: Optional[str] = None,
        idempotent: bool = True,
        reconcile: str = "repeat_safe",
        timeout: float = 30,
    ) -> ToolManifest:
        return ToolManifest(
            name=name,
            capability=capability,
            risk=risk,
            approval_category=approval,
            idempotent=idempotent,
            reconcile=reconcile,
            timeout_seconds=timeout,
            max_output_chars=max_output_chars,
        )

    return ToolRegistry(
        [
            manifest("get_working_directory", CapabilityClass.READ),
            manifest(
                "set_working_directory",
                CapabilityClass.WRITE,
                risk=RiskLevel.MEDIUM,
                idempotent=False,
            ),
            manifest("list_allowed_roots", CapabilityClass.READ),
            manifest("list_files", CapabilityClass.READ, approval="file_read"),
            manifest("read_text_file", CapabilityClass.READ, approval="file_read"),
            manifest("search_text", CapabilityClass.READ, approval="file_read"),
            manifest("file_info", CapabilityClass.READ, approval="file_read"),
            manifest(
                "write_text_file",
                CapabilityClass.WRITE,
                risk=RiskLevel.HIGH,
                approval="file_write",
                idempotent=False,
                reconcile="post_image_hash",
            ),
            manifest(
                "edit_text_file",
                CapabilityClass.WRITE,
                risk=RiskLevel.HIGH,
                approval="file_write",
                idempotent=False,
                reconcile="post_image_hash",
            ),
            manifest(
                "create_directory",
                CapabilityClass.WRITE,
                risk=RiskLevel.MEDIUM,
                approval="file_write",
                reconcile="resource_exists",
            ),
            manifest(
                "web_search",
                CapabilityClass.NETWORK,
                risk=RiskLevel.MEDIUM,
                approval="web",
                timeout=45,
            ),
            manifest(
                "web_fetch",
                CapabilityClass.NETWORK,
                risk=RiskLevel.MEDIUM,
                approval="web",
                timeout=45,
            ),
            manifest("search_memory", CapabilityClass.READ),
            manifest("get_sandbox_status", CapabilityClass.READ),
            manifest(
                "run_sandboxed_command",
                CapabilityClass.EXECUTE,
                risk=RiskLevel.HIGH,
                approval="command",
                idempotent=False,
                reconcile="inspect_external_receipt",
                timeout=300,
            ),
            manifest("react_dispatch", CapabilityClass.READ),
            manifest("critique_and_plan", CapabilityClass.READ),
            manifest("get_task_plan", CapabilityClass.READ),
            manifest("create_task_plan", CapabilityClass.WRITE, risk=RiskLevel.MEDIUM),
            manifest(
                "add_task_plan_item", CapabilityClass.WRITE, risk=RiskLevel.MEDIUM
            ),
            manifest(
                "update_task_plan_item", CapabilityClass.WRITE, risk=RiskLevel.MEDIUM
            ),
            manifest("set_session_title", CapabilityClass.WRITE, risk=RiskLevel.LOW),
        ]
    )


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    requires_approval: bool
    reason: str
    capability: str
    risk: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "requires_approval": self.requires_approval,
            "reason": self.reason,
            "capability": self.capability,
            "risk": self.risk,
        }


class ToolPolicy:
    """Deterministic checks performed before provider/tool implementation code."""

    def __init__(self, registry: ToolRegistry, workspace: Path):
        self.registry = registry
        self.workspace = workspace.resolve()

    def evaluate(
        self, name: str, arguments: Mapping[str, Any], *, remaining_steps: int
    ) -> PolicyDecision:
        manifest = self.registry.get(name)
        if remaining_steps <= 0 and name not in {"react_dispatch", "critique_and_plan"}:
            return PolicyDecision(
                False,
                False,
                "Tool budget exhausted",
                manifest.capability.value,
                manifest.risk.value,
            )
        for key in ("path", "cwd"):
            raw = arguments.get(key)
            if not isinstance(raw, str) or not raw:
                continue
            target = (self.workspace / raw).resolve()
            try:
                target.relative_to(self.workspace)
            except ValueError:
                return PolicyDecision(
                    False,
                    False,
                    "Target escapes workspace scope",
                    manifest.capability.value,
                    manifest.risk.value,
                )
        return PolicyDecision(
            True,
            manifest.approval_category is not None,
            "Policy checks passed",
            manifest.capability.value,
            manifest.risk.value,
        )


class ProgressEvaluator:
    """Cheap host-owned progress check; never asks for hidden reasoning."""

    def evaluate(
        self,
        outcome: ToolOutcome,
        *,
        prior_evidence_ids: Sequence[str],
        stagnation_score: float,
        milestone: bool = False,
    ) -> Dict[str, Any]:
        new_evidence = tuple(
            item for item in outcome.evidence_ids if item not in set(prior_evidence_ids)
        )
        reasons: list[str] = []
        if outcome.status != ToolStatus.SUCCESS:
            reasons.append(f"outcome:{outcome.status.value}")
        if outcome.changed:
            reasons.append("mutation")
        if milestone:
            reasons.append("milestone")
        if stagnation_score >= 1.0:
            reasons.append("stagnation")
        if not new_evidence:
            reasons.append("no_new_evidence")
        return {
            "advanced": bool(new_evidence or outcome.changed),
            "new_evidence_ids": list(new_evidence),
            "requires_critique": bool(reasons),
            "reasons": reasons,
        }


class SemanticStagnationDetector:
    """Detect equivalent actions and evidence-free cycles across cosmetic edits."""

    def __init__(self, max_history: int = 20):
        self._history: deque[tuple[str, tuple[str, ...]]] = deque(
            maxlen=max(4, max_history)
        )

    @staticmethod
    def signature(name: str, arguments: Mapping[str, Any]) -> str:
        def normalize(value: Any) -> Any:
            if isinstance(value, Mapping):
                return {
                    str(key).casefold(): normalize(item)
                    for key, item in sorted(
                        value.items(), key=lambda pair: str(pair[0]).casefold()
                    )
                }
            if isinstance(value, (list, tuple)):
                return [normalize(item) for item in value]
            if isinstance(value, str):
                compact = re.sub(r"\s+", " ", value.strip())
                if any(mark in compact for mark in ("/", "\\", ".")):
                    compact = compact.replace("\\", "/").casefold()
                return compact
            return value

        return content_hash(
            {"name": name.casefold(), "arguments": normalize(arguments)}
        )

    def observe(
        self, name: str, arguments: Mapping[str, Any], outcome: ToolOutcome
    ) -> float:
        signature = self.signature(name, arguments)
        evidence = tuple(sorted(outcome.evidence_ids))
        score = 0.0
        for old_signature, old_evidence in self._history:
            if old_signature == signature:
                score += 1.0 if old_evidence == evidence else 0.5
        if len(self._history) >= 3:
            recent = list(self._history)[-3:]
            if len({item[0] for item in recent}) <= 2 and not evidence:
                score += 1.0
        self._history.append((signature, evidence))
        return score


class CompletionValidator:
    """Reject prose-backed completion when evidence or receipts are incomplete."""

    def validate(
        self,
        outcomes: Sequence[ToolOutcome],
        *,
        success_criteria: Sequence[str] = (),
        criterion_evidence: Optional[Mapping[str, Sequence[str]]] = None,
        pending_approval: bool = False,
        pending_tool: bool = False,
        unresolved_fatal_error: bool = False,
        plan_incomplete: bool = False,
    ) -> CompletionDecision:
        reasons: list[str] = []
        unverified: list[str] = []
        evidence = {item for outcome in outcomes for item in outcome.evidence_ids}
        successful = [outcome for outcome in outcomes if outcome.succeeded]
        if not successful:
            reasons.append("No successful tool evidence exists for this task")
        for outcome in outcomes:
            if outcome.changed and not bool(outcome.receipt.get("verified")):
                resource = str(outcome.receipt.get("resource") or "changed resource")
                unverified.append(resource)
        if unverified:
            reasons.append("Mutation receipts require verification")
        if pending_approval:
            reasons.append("An approval is still pending")
        if pending_tool:
            reasons.append("A tool execution is still pending")
        if unresolved_fatal_error:
            reasons.append("A fatal error remains unresolved")
        if plan_incomplete:
            reasons.append("The active plan still has unfinished required items")

        mapping = criterion_evidence or {}
        covered = 0
        for criterion in success_criteria:
            refs = tuple(mapping.get(criterion, ()))
            if refs and any(ref in evidence for ref in refs):
                covered += 1
            else:
                unverified.append(criterion)
        coverage = (
            covered / len(success_criteria)
            if success_criteria
            else (1.0 if successful else 0.0)
        )
        if success_criteria and covered != len(success_criteria):
            reasons.append("Not all success criteria have evidence coverage")
        return CompletionDecision(
            accepted=not reasons,
            reasons=tuple(reasons),
            evidence_coverage=coverage,
            unverified=tuple(dict.fromkeys(unverified)),
        )


class UntrustedContentScanner:
    """Tag likely instruction injection without treating untrusted text as policy."""

    _SIGNALS = {
        "instruction_override": re.compile(
            r"(?i)\b(ignore|disregard|override)\b.{0,40}\b(instruction|system|developer|previous)\b"
        ),
        "role_impersonation": re.compile(
            r"(?im)^\s*(system|developer|assistant)\s*:\s*"
        ),
        "secret_request": re.compile(
            r"(?i)\b(reveal|print|send|upload|exfiltrate)\b.{0,50}\b(secret|token|password|api[_ -]?key)\b"
        ),
        "tool_directive": re.compile(
            r"(?i)<\s*tool_call|\bcall\s+(?:the\s+)?(?:shell|command|write)_?tool\b"
        ),
    }

    @classmethod
    def scan(cls, value: str) -> Dict[str, Any]:
        text = str(value or "")
        signals = [
            name for name, pattern in cls._SIGNALS.items() if pattern.search(text)
        ]
        return {
            "trust": "untrusted_data",
            "injection_signals": signals,
            "instruction_priority": "none",
        }


class DeterministicReadBatchExecutor:
    """Run independent repeat-safe reads concurrently and return input order."""

    def __init__(
        self, registry: ToolRegistry, *, max_workers: int = 4, max_calls: int = 16
    ):
        self.registry = registry
        self.max_workers = max(1, min(int(max_workers), 8))
        self.max_calls = max(1, min(int(max_calls), 64))

    def execute(
        self,
        calls: Sequence[tuple[str, Mapping[str, Any]]],
        invoke: Any,
    ) -> list[Dict[str, Any]]:
        if len(calls) > self.max_calls:
            raise ValueError("Parallel read batch exceeds call budget")
        for name, _arguments in calls:
            manifest = self.registry.get(name)
            if manifest.capability != CapabilityClass.READ or not manifest.idempotent:
                raise ValueError(
                    f"Parallel execution is restricted to repeat-safe reads: {name}"
                )
        ordered: list[Optional[Dict[str, Any]]] = [None] * len(calls)
        with ThreadPoolExecutor(
            max_workers=min(self.max_workers, max(1, len(calls)))
        ) as pool:
            futures = {
                pool.submit(invoke, name, dict(arguments)): (index, name)
                for index, (name, arguments) in enumerate(calls)
            }
            for future in as_completed(futures):
                index, name = futures[future]
                try:
                    value = future.result()
                except Exception as error:
                    outcome = ToolOutcome(
                        status=ToolStatus.RETRYABLE_ERROR,
                        summary=f"{name} raised {type(error).__name__}",
                        error_code=ErrorCode.EXECUTION_FAILED,
                    )
                    ordered[index] = {
                        "name": name,
                        "result": None,
                        "outcome": outcome.model_dump(mode="json"),
                    }
                else:
                    ordered[index] = {
                        "name": name,
                        "result": value,
                        "outcome": ToolOutcome.success(
                            f"{name} completed",
                            evidence_ids=(evidence_id(name, value),),
                        ).model_dump(mode="json"),
                    }
        return [item for item in ordered if item is not None]


__all__ = [
    "CompletionValidator",
    "PolicyDecision",
    "ProgressEvaluator",
    "SemanticStagnationDetector",
    "ToolPolicy",
    "ToolRegistry",
    "Toolset",
    "ToolsetRegistry",
    "DEFAULT_TOOLSETS",
    "default_tool_registry",
    "default_toolset_registry",
    "evidence_id",
    "mutation_receipt",
    "UntrustedContentScanner",
    "DeterministicReadBatchExecutor",
]
