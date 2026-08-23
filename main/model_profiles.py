"""Model capability profiles and workspace-scoped overrides."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

try:
    import tomllib
except ImportError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib


@dataclass(frozen=True)
class ModelCapabilityProfile:
    """Capabilities OpenCLI needs for safe context and feature decisions."""

    key: str
    display_name: str
    model_id: str
    backend: str
    context_window: int
    max_output_tokens: int
    supports_tools: bool
    supports_vision: bool = False
    supports_reasoning: bool = False
    tokenizer: Optional[str] = None
    source: str = "built-in"


FALLBACK_PROFILE = ModelCapabilityProfile(
    key="unknown",
    display_name="Unknown model",
    model_id="unknown",
    backend="unknown",
    context_window=32_768,
    max_output_tokens=4_096,
    supports_tools=False,
    source="conservative fallback",
)


BUILTIN_PROFILES = (
    ModelCapabilityProfile(
        key="ministral-3-14b",
        display_name="Ministral 3 14B Instruct",
        model_id="mistralai/Ministral-3-14B-Instruct-2512-GGUF",
        backend="llama_cpp",
        context_window=32_768,
        max_output_tokens=8_192,
        supports_tools=True,
        supports_vision=True,
    ),
    ModelCapabilityProfile(
        key="gpt-oss-20b",
        display_name="GPT-OSS 20B",
        model_id="unsloth/gpt-oss-20b-GGUF",
        backend="llama_cpp",
        context_window=32_768,
        max_output_tokens=16_384,
        supports_tools=True,
        supports_reasoning=True,
    ),
    ModelCapabilityProfile(
        key="devstral-small-2-24b",
        display_name="Devstral Small 2 24B",
        model_id="bartowski/mistralai_Devstral-Small-2-24B-Instruct-2512-GGUF",
        backend="llama_cpp",
        context_window=32_768,
        max_output_tokens=8_192,
        supports_tools=True,
        supports_vision=True,
    ),
    ModelCapabilityProfile(
        key="qwen3.8-27b",
        display_name="Qwen 3.8 27B",
        model_id="unsloth/Qwen3.8-27B-GGUF",
        backend="llama_cpp",
        context_window=32_768,
        max_output_tokens=16_384,
        supports_tools=True,
        supports_vision=True,
        supports_reasoning=True,
    ),
)


class ModelProfileRegistry:
    """Resolve built-in, inferred, fallback, and workspace override profiles."""

    _OVERRIDE_FIELDS = {
        "display_name",
        "backend",
        "context_window",
        "max_output_tokens",
        "supports_tools",
        "supports_vision",
        "supports_reasoning",
        "tokenizer",
    }

    def __init__(self, workspace: Path, config_path: Optional[Path] = None):
        self.workspace = workspace.resolve()
        self.config_path = config_path or self.workspace / ".opencli" / "config.toml"
        self.warnings: list[str] = []
        self._profiles: Dict[str, ModelCapabilityProfile] = {}
        for profile in BUILTIN_PROFILES:
            self._profiles[profile.key.casefold()] = profile
            self._profiles[profile.model_id.casefold()] = profile
        self._overrides = self._load_overrides()

    def _warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    def _load_overrides(self) -> Dict[str, Dict[str, Any]]:
        if not self.config_path.is_file():
            return {}
        try:
            with self.config_path.open("rb") as config_file:
                data = tomllib.load(config_file)
        except (OSError, tomllib.TOMLDecodeError) as error:
            self._warn(f"Could not load {self.config_path}: {error}")
            return {}
        models = data.get("models", {})
        if not isinstance(models, dict):
            self._warn("[models] in .opencli/config.toml must be a table")
            return {}
        return {
            str(identifier).casefold(): dict(values)
            for identifier, values in models.items()
            if isinstance(values, dict)
        }

    @staticmethod
    def _metadata_profile(
        key: str,
        model_id: str,
        backend: str,
        metadata: Mapping[str, Any],
    ) -> ModelCapabilityProfile:
        context_window = int(metadata.get("context", FALLBACK_PROFILE.context_window))
        max_output = int(metadata.get("max_tokens", FALLBACK_PROFILE.max_output_tokens))
        context_window = min(max(context_window, 512), 1_000_000)
        # A model can advertise output almost as large as its whole context.
        # Reserving that value leaves no room for even a clean prompt.
        max_output = min(max(max_output, 64), max(64, context_window // 2))
        return replace(
            FALLBACK_PROFILE,
            key=key or model_id,
            display_name=str(metadata.get("name") or metadata.get("display_name") or model_id),
            model_id=model_id,
            backend=backend or str(metadata.get("backend") or "unknown"),
            context_window=context_window,
            max_output_tokens=max_output,
            supports_tools=bool(metadata.get("supports_tools", False)),
            supports_vision=bool(metadata.get("supports_vision", False)),
            supports_reasoning=bool(metadata.get("has_thinking", False)),
            source="model metadata",
        )

    def _apply_override(
        self, profile: ModelCapabilityProfile, identifier: str
    ) -> ModelCapabilityProfile:
        values = self._overrides.get(identifier.casefold())
        if values is None:
            return profile
        unknown = sorted(set(values) - self._OVERRIDE_FIELDS)
        if unknown:
            self._warn(
                f"Ignored unknown model override fields for {identifier}: {', '.join(unknown)}"
            )
        changes = {key: values[key] for key in self._OVERRIDE_FIELDS if key in values}
        try:
            if "context_window" in changes:
                changes["context_window"] = int(changes["context_window"])
            if "max_output_tokens" in changes:
                changes["max_output_tokens"] = int(changes["max_output_tokens"])
            for name in ("supports_tools", "supports_vision", "supports_reasoning"):
                if name in changes and not isinstance(changes[name], bool):
                    raise ValueError(f"{name} must be true or false")
            result = replace(profile, **changes, source=f"workspace override: {identifier}")
            if not 512 <= result.context_window <= 1_000_000:
                raise ValueError("context_window must be between 512 and 1000000")
            if not 64 <= result.max_output_tokens <= result.context_window // 2:
                raise ValueError("max_output_tokens must use at most half the context window")
            return result
        except (TypeError, ValueError) as error:
            self._warn(f"Ignored invalid model override for {identifier}: {error}")
            return profile

    def resolve(
        self,
        *,
        key: str,
        model_id: str,
        backend: str,
        provider: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ModelCapabilityProfile:
        profile = self._profiles.get(key.casefold()) or self._profiles.get(model_id.casefold())
        if profile is None:
            if metadata:
                profile = self._metadata_profile(key, model_id, backend, metadata)
            else:
                profile = replace(
                    FALLBACK_PROFILE,
                    key=key or model_id,
                    display_name=model_id or "Unknown model",
                    model_id=model_id or "unknown",
                    backend=backend or "unknown",
                )
        candidates = []
        if provider:
            candidates.append(f"{provider}:{model_id}")
        candidates.extend([model_id, key])
        for identifier in candidates:
            if identifier and identifier.casefold() in self._overrides:
                return self._apply_override(profile, identifier)
        return profile


__all__ = [
    "BUILTIN_PROFILES",
    "FALLBACK_PROFILE",
    "ModelCapabilityProfile",
    "ModelProfileRegistry",
]
