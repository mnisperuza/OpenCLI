"""Persistent API provider/model profiles. API keys never enter this file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from .api_providers import PROVIDERS, OpenAICompatibleClient


class ApiProfileRegistry:
    """Store provider/model choices for fast, keyless API startup."""

    def __init__(self, state_file: Optional[Path] = None):
        self.state_file = state_file or Path.home() / ".opencli" / "api-profiles.json"
        self._profiles, self._default = self._load()

    def _load(self):
        try:
            data = json.loads(self.state_file.read_text(encoding="utf-8"))
            profiles = data.get("profiles", {})
            if not isinstance(profiles, dict):
                return {}, None
            clean = {
                str(key): {"provider": value["provider"], "model": value["model"]}
                for key, value in profiles.items()
                if isinstance(value, dict)
                and value.get("provider") in PROVIDERS
                and OpenAICompatibleClient.normalize_model_id(value.get("model", ""))
            }
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
    def profiles(self) -> Dict[str, Dict[str, str]]:
        return {key: dict(value) for key, value in self._profiles.items()}

    def save(self, provider: str, model: str) -> str:
        model = OpenAICompatibleClient.normalize_model_id(model)
        if provider not in PROVIDERS or not model:
            raise ValueError("Invalid API profile")
        key = f"{provider}:{model}"
        self._profiles[key] = {"provider": provider, "model": model}
        self._default = key
        self._save()
        return key

    def default(self) -> Optional[Dict[str, str]]:
        if self._default is None:
            return None
        profile = self._profiles.get(self._default)
        return dict(profile) if profile else None

    def remove(self, key: str) -> Dict[str, str]:
        profile = self._profiles.pop(key)
        if self._default == key:
            self._default = next(iter(self._profiles), None)
        self._save()
        return profile


__all__ = ["ApiProfileRegistry"]
