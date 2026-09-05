"""Persistent API provider/model profiles. API keys never enter this file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from .api_providers import PROVIDERS, OpenAICompatibleClient


class ApiProfileRegistry:
    """Store provider/model choices for fast, keyless API startup."""

    def __init__(self, state_file: Optional[Path] = None):
        self.state_file = state_file or Path.home() / ".fenrir" / "api-profiles.json"
        self._profiles, self._default = self._load()

    def _load(self):
        try:
            data = json.loads(self.state_file.read_text(encoding="utf-8"))
            profiles = data.get("profiles", {})
            if not isinstance(profiles, dict):
                return {}, None
            clean = {}
            for key, value in profiles.items():
                if (
                    not isinstance(value, dict)
                    or value.get("provider") not in PROVIDERS
                    or not OpenAICompatibleClient.normalize_model_id(value.get("model", ""))
                ):
                    continue
                profile: Dict[str, Any] = {
                    "provider": value["provider"],
                    "model": value["model"],
                }
                for field in ("context_window", "max_output_tokens"):
                    number = value.get(field)
                    if isinstance(number, int) and number > 0:
                        profile[field] = number
                clean[str(key)] = profile
            default = data.get("default")
            return clean, default if default in clean else None
        except (OSError, TypeError, ValueError, KeyError):
            return {}, None

    def _save(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_file.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({"version": 1, "default": self._default, "profiles": self._profiles}, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.state_file)

    @property
    def profiles(self) -> Dict[str, Dict[str, Any]]:
        return {key: dict(value) for key, value in self._profiles.items()}

    def save(
        self,
        provider: str,
        model: str,
        *,
        context_window: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
    ) -> str:
        model = OpenAICompatibleClient.normalize_model_id(model)
        if provider not in PROVIDERS or not model:
            raise ValueError("Invalid API profile")
        if context_window is not None:
            context_window = int(context_window)
            if not 512 <= context_window <= 1_000_000:
                raise ValueError("Context window must be between 512 and 1,000,000")
        if max_output_tokens is not None:
            max_output_tokens = int(max_output_tokens)
            if not 64 <= max_output_tokens <= 262_144:
                raise ValueError("Output limit must be between 64 and 262,144")
        if (
            context_window is not None
            and max_output_tokens is not None
            and max_output_tokens > context_window // 2
        ):
            raise ValueError("Output limit must use at most half the context window")
        key = f"{provider}:{model}"
        profile: Dict[str, Any] = {"provider": provider, "model": model}
        if context_window is not None:
            profile["context_window"] = context_window
        if max_output_tokens is not None:
            profile["max_output_tokens"] = max_output_tokens
        self._profiles[key] = profile
        self._default = key
        self._save()
        return key

    def default(self) -> Optional[Dict[str, Any]]:
        if self._default is None:
            return None
        profile = self._profiles.get(self._default)
        return dict(profile) if profile else None

    def remove(self, key: str) -> Dict[str, Any]:
        profile = self._profiles.pop(key)
        if self._default == key:
            self._default = next(iter(self._profiles), None)
        self._save()
        return profile


__all__ = ["ApiProfileRegistry"]
