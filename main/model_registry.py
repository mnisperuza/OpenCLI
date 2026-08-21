"""Persistent user-added llama.cpp model definitions."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional


class ModelRegistryError(ValueError):
    """Raised when a custom model definition is invalid."""


class ModelRegistry:
    """Store at most ten user-managed GGUF model profiles."""

    MAX_CUSTOM_MODELS = 10

    def __init__(self, state_file: Optional[Path] = None):
        self.state_file = state_file or Path.home() / ".opencli" / "models.json"
        self._models = self._load()

    def _load(self) -> Dict[str, Dict[str, Any]]:
        try:
            data = json.loads(self.state_file.read_text(encoding="utf-8"))
            models = data.get("models", {})
            if isinstance(models, dict):
                return {
                    str(key): value
                    for key, value in models.items()
                    if isinstance(value, dict)
                }
        except (OSError, TypeError, ValueError):
            pass
        return {}

    def _save(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_file.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({"version": 1, "models": self._models}, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.state_file)

    @staticmethod
    def _key(name: str) -> str:
        key = re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-")
        return key[:48]

    @staticmethod
    def _number(value: Any, name: str, minimum: float, maximum: float) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError) as error:
            raise ModelRegistryError(f"{name} must be a number") from error
        if not minimum <= number <= maximum:
            raise ModelRegistryError(
                f"{name} must be between {minimum:g} and {maximum:g}"
            )
        return number

    @property
    def models(self) -> Dict[str, Dict[str, Any]]:
        return {key: dict(value) for key, value in self._models.items()}

    def engine_models(self) -> Dict[str, Dict[str, Any]]:
        return self.models

    def add(
        self,
        *,
        name: str,
        source_type: str,
        path: str,
        llama_file: str = "",
        context: Any = 32768,
        max_tokens: Any = 8192,
        temperature: Any = 0.7,
        has_thinking: bool = False,
        supports_vision: bool = False,
        reserved_keys: Optional[set[str]] = None,
    ) -> str:
        if len(self._models) >= self.MAX_CUSTOM_MODELS:
            raise ModelRegistryError(
                f"Maximum custom models reached ({self.MAX_CUSTOM_MODELS})"
            )
        display_name = name.strip()
        if not 2 <= len(display_name) <= 80:
            raise ModelRegistryError("Model name must be 2-80 characters")
        key = self._key(display_name)
        if not key:
            raise ModelRegistryError("Model name must contain letters or numbers")
        if key in self._models or key in (reserved_keys or set()):
            raise ModelRegistryError("Model name already exists")

        source_type = source_type.strip().casefold()
        source = path.strip()
        if source_type not in {"huggingface", "local"}:
            raise ModelRegistryError("Source type must be huggingface or local")
        if source_type == "huggingface":
            if not re.fullmatch(r"[\w.-]+/[\w.-]+(?::[\w.-]+)?", source):
                raise ModelRegistryError(
                    "Hugging Face source must be owner/repository[:quant]"
                )
            if llama_file and not llama_file.strip().casefold().endswith(".gguf"):
                raise ModelRegistryError("Exact model filename must end in .gguf")
        else:
            local_path = Path(source).expanduser().resolve()
            if local_path.suffix.casefold() != ".gguf" or not local_path.is_file():
                raise ModelRegistryError("Local source must be an existing .gguf file")
            source = str(local_path)

        context_size = int(self._number(context, "Context", 512, 1_000_000))
        output_tokens = int(
            self._number(max_tokens, "Max output tokens", 64, context_size - 1)
        )
        temp = self._number(temperature, "Temperature", 0.0, 2.0)
        self._models[key] = {
            "name": display_name,
            "display_name": display_name,
            "path": source,
            "source_type": source_type,
            "llama_file": llama_file.strip() or None,
            "family": "custom",
            "max_tokens": output_tokens,
            "temp": temp,
            "top_k": 40,
            "repetition_penalty": 1.05,
            "context": context_size,
            "vram": "Custom",
            "note": "User-added GGUF model",
            "usage": "User-added model",
            "has_thinking": bool(has_thinking),
            "supports_vision": bool(supports_vision),
            "backend": "llama_cpp",
            "locked": False,
        }
        self._save()
        return key

    def remove(self, key: str) -> Dict[str, Any]:
        try:
            model = self._models.pop(key)
        except KeyError as error:
            raise ModelRegistryError("Unknown custom model") from error
        self._save()
        return model


__all__ = ["ModelRegistry", "ModelRegistryError"]
