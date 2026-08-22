"""Secure, dependency-light clients for OpenAI-compatible API providers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class ApiProviderError(RuntimeError):
    """Safe provider error that never includes an API key."""


@dataclass(frozen=True)
class ProviderDefinition:
    key: str
    name: str
    base_url: str
    key_url: str
    environment_variable: str


PROVIDERS: Dict[str, ProviderDefinition] = {
    "groq": ProviderDefinition(
        key="groq",
        name="Groq",
        base_url="https://api.groq.com/openai/v1",
        key_url="https://console.groq.com/keys",
        environment_variable="GROQ_API_KEY",
    ),
    "gemini": ProviderDefinition(
        key="gemini",
        name="Gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        key_url="https://aistudio.google.com/apikey",
        environment_variable="GEMINI_API_KEY",
    ),
    "openrouter": ProviderDefinition(
        key="openrouter",
        name="OpenRouter",
        base_url="https://openrouter.ai/api/v1",
        key_url="https://openrouter.ai/settings/keys",
        environment_variable="OPENROUTER_API_KEY",
    ),
}


class OpenAICompatibleClient:
    """Model discovery and native function calling over standard HTTP/SSE."""

    def __init__(
        self,
        provider: str,
        api_key: str,
        model: str = "",
        timeout_seconds: int = 300,
    ):
        try:
            self.definition = PROVIDERS[provider]
        except KeyError as error:
            raise ValueError(f"Unknown API provider: {provider}") from error
        self.api_key = api_key.strip()
        if not self.api_key:
            raise ValueError("API key cannot be empty")
        self.model = self.normalize_model_id(model)
        self.timeout_seconds = timeout_seconds
        self.max_output_tokens: Optional[int] = None

    @staticmethod
    def normalize_model_id(value: str) -> str:
        """Validate model identifiers without altering provider case semantics."""
        model = str(value or "").strip()
        if not model:
            return ""
        if len(model) > 256 or any(character.isspace() for character in model):
            raise ValueError("Invalid API model ID")
        if any(ord(character) < 32 or ord(character) == 127 for character in model):
            raise ValueError("Invalid API model ID")
        return model

    @property
    def provider(self) -> str:
        return self.definition.key

    @property
    def provider_name(self) -> str:
        return self.definition.name

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "OpenCLI/1.5",
        }
        if self.provider == "openrouter":
            headers.update(
                {
                    "HTTP-Referer": "https://github.com/mnisperuza/bert-cli",
                    "X-OpenRouter-Title": "OpenCLI",
                }
            )
        return headers

    def _safe_error(self, error: Exception, body: str = "") -> ApiProviderError:
        detail = " ".join(body.split())[:1_000]
        detail = detail.replace(self.api_key, "[redacted]")
        if isinstance(error, HTTPError):
            message = f"{self.provider_name} API returned HTTP {error.code}"
        elif isinstance(error, URLError):
            message = f"Could not reach {self.provider_name} API"
        else:
            message = f"{self.provider_name} API failed"
        if detail:
            message += f": {detail}"
        return ApiProviderError(message)

    def list_models(self) -> List[str]:
        request = Request(
            f"{self.definition.base_url}/models",
            headers=self._headers(),
            method="GET",
        )
        try:
            with urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise self._safe_error(error, body) from error
        except (OSError, URLError, ValueError, json.JSONDecodeError) as error:
            raise self._safe_error(error) from error

        models = []
        for item in payload.get("data", []):
            if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                continue
            try:
                model_id = self.normalize_model_id(item["id"])
            except ValueError:
                continue
            supported = item.get("supported_parameters")
            if self.provider == "openrouter" and isinstance(supported, list):
                if "tools" not in supported:
                    continue
            if self.provider == "gemini":
                lowered = model_id.casefold()
                if "gemini" not in lowered or any(
                    marker in lowered
                    for marker in ("embedding", "image", "tts", "aqa")
                ):
                    continue
            if model_id:
                models.append(model_id)
        return sorted(set(models), key=str.casefold)

    def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> Generator[Dict[str, Any], None, None]:
        if not self.model:
            raise ApiProviderError("No API model selected")
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }
        if isinstance(self.max_output_tokens, int) and self.max_output_tokens > 0:
            body["max_tokens"] = self.max_output_tokens
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        request = Request(
            f"{self.definition.base_url}/chat/completions",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        calls: Dict[int, Dict[str, str]] = {}
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        payload = json.loads(data)
                        if not isinstance(payload, dict):
                            continue
                        if payload.get("error"):
                            detail = " ".join(str(payload["error"]).split())[:1_000]
                            detail = detail.replace(self.api_key, "[redacted]")
                            raise ApiProviderError(
                                f"{self.provider_name} API error: {detail}"
                            )
                    except (TypeError, json.JSONDecodeError):
                        continue
                    usage = payload.get("usage")
                    if isinstance(usage, dict):
                        input_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
                        output_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
                        if isinstance(input_tokens, int) and isinstance(output_tokens, int):
                            yield {
                                "type": "usage",
                                "input_tokens": input_tokens,
                                "output_tokens": output_tokens,
                            }
                    choices = payload.get("choices") or []
                    if not choices or not isinstance(choices[0], dict):
                        continue
                    delta = choices[0].get("delta", {})
                    content = delta.get("content")
                    if isinstance(content, str) and content:
                        yield {"type": "token", "content": content}
                    for tool_delta in delta.get("tool_calls") or []:
                        if not isinstance(tool_delta, dict):
                            continue
                        index = int(tool_delta.get("index", 0))
                        state = calls.setdefault(
                            index,
                            {"id": "", "name": "", "arguments": ""},
                        )
                        state["id"] += str(tool_delta.get("id") or "")
                        function = tool_delta.get("function") or {}
                        if isinstance(function, dict):
                            state["name"] += str(function.get("name") or "")
                            state["arguments"] += str(
                                function.get("arguments") or ""
                            )
        except ApiProviderError:
            raise
        except HTTPError as error:
            body_text = error.read().decode("utf-8", errors="replace")
            raise self._safe_error(error, body_text) from error
        except (OSError, URLError) as error:
            raise self._safe_error(error) from error

        if calls:
            yield {
                "type": "tool_calls",
                "calls": [calls[index] for index in sorted(calls)],
            }


__all__ = [
    "ApiProviderError",
    "OpenAICompatibleClient",
    "PROVIDERS",
    "ProviderDefinition",
]
