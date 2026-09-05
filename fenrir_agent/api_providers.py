"""Secure, dependency-light clients for OpenAI-compatible API providers."""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .provider_reliability import ProviderCapabilities, ProviderReliabilityController


class ApiProviderError(RuntimeError):
    """Safe provider error that never includes an API key."""


def _origin(url: str) -> tuple[str, str, int | None]:
    parsed = urlparse(url)
    return parsed.scheme.casefold(), (parsed.hostname or "").casefold(), parsed.port


class _SameOriginRedirectHandler(HTTPRedirectHandler):
    """Never forward credential-bearing provider requests to another origin."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if _origin(req.full_url) != _origin(newurl):
            raise URLError("Provider redirect changed origin")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def safe_urlopen(request: Request, timeout: int):
    """Open provider request while allowing only same-origin redirects."""
    return build_opener(_SameOriginRedirectHandler()).open(request, timeout=timeout)


@dataclass(frozen=True)
class ProviderDefinition:
    key: str
    name: str
    base_url: str
    key_url: str
    environment_variable: str
    environment_variable_aliases: tuple[str, ...] = ()
    base_url_environment_variable: str = ""
    models_path: str = "models"

    def api_key_from_environment(self) -> str:
        """Read first configured provider key without persisting it."""
        for name in (self.environment_variable, *self.environment_variable_aliases):
            value = os.environ.get(name, "").strip()
            if value:
                return value
        return ""

    def resolved_base_url(self) -> str:
        """Resolve optional regional endpoint and reject unsafe URLs."""
        value = self.base_url
        if self.base_url_environment_variable:
            value = os.environ.get(self.base_url_environment_variable, value).strip()
        value = value.rstrip("/")
        parsed = urlparse(value)
        loopback_http = (
            parsed.scheme == "http"
            and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
        )
        if (
            (parsed.scheme != "https" and not loopback_http)
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(f"Invalid {self.name} API base URL")
        return value


PROVIDERS: Dict[str, ProviderDefinition] = {
    "groq": ProviderDefinition(
        key="groq",
        name="Groq",
        base_url="https://api.groq.com/openai/v1",
        key_url="https://console.groq.com/keys",
        environment_variable="GROQ_API_KEY",
    ),
    "openai": ProviderDefinition(
        key="openai",
        name="OpenAI",
        base_url="https://api.openai.com/v1",
        key_url="https://platform.openai.com/api-keys",
        environment_variable="OPENAI_API_KEY",
    ),
    "cerebras": ProviderDefinition(
        key="cerebras",
        name="Cerebras",
        base_url="https://api.cerebras.ai/v1",
        key_url="https://cloud.cerebras.ai/platform/api-keys",
        environment_variable="CEREBRAS_API_KEY",
    ),
    "deepseek": ProviderDefinition(
        key="deepseek",
        name="DeepSeek",
        base_url="https://api.deepseek.com",
        key_url="https://platform.deepseek.com/api_keys",
        environment_variable="DEEPSEEK_API_KEY",
    ),
    "xai": ProviderDefinition(
        key="xai",
        name="xAI",
        base_url="https://api.x.ai/v1",
        key_url="https://console.x.ai",
        environment_variable="XAI_API_KEY",
    ),
    "nvidia": ProviderDefinition(
        key="nvidia",
        name="NVIDIA NIM",
        base_url="https://integrate.api.nvidia.com/v1",
        key_url="https://build.nvidia.com",
        environment_variable="NVIDIA_API_KEY",
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
    "qwen": ProviderDefinition(
        key="qwen",
        name="Qwen Cloud (Alibaba Model Studio)",
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        key_url="https://www.alibabacloud.com/help/en/model-studio/get-api-key",
        environment_variable="DASHSCOPE_API_KEY",
        environment_variable_aliases=("QWEN_API_KEY",),
        base_url_environment_variable="QWEN_BASE_URL",
    ),
    "mistral": ProviderDefinition(
        key="mistral",
        name="Mistral AI",
        base_url="https://api.mistral.ai/v1",
        key_url="https://console.mistral.ai/api-keys",
        environment_variable="MISTRAL_API_KEY",
    ),
    "fireworks": ProviderDefinition(
        key="fireworks",
        name="Fireworks AI",
        base_url="https://api.fireworks.ai/inference/v1",
        key_url="https://fireworks.ai/account/api-keys",
        environment_variable="FIREWORKS_API_KEY",
    ),
    "together": ProviderDefinition(
        key="together",
        name="Together AI",
        base_url="https://api.together.xyz/v1",
        key_url="https://api.together.ai/settings/api-keys",
        environment_variable="TOGETHER_API_KEY",
    ),
    "freellmapi": ProviderDefinition(
        key="freellmapi",
        name="FreeLLMAPI Gateway",
        base_url="http://127.0.0.1:3001/v1",
        key_url="http://127.0.0.1:3001",
        environment_variable="FREELLMAPI_API_KEY",
        base_url_environment_variable="FREELLMAPI_BASE_URL",
        models_path="models?ready=true",
    ),
    "litellm": ProviderDefinition(
        key="litellm",
        name="LiteLLM Gateway",
        base_url="http://127.0.0.1:4000",
        key_url="https://docs.litellm.ai/docs/proxy/virtual_keys",
        environment_variable="LITELLM_API_KEY",
        base_url_environment_variable="LITELLM_BASE_URL",
    ),
    "ds2api": ProviderDefinition(
        key="ds2api",
        name="DS2API",
        base_url="http://127.0.0.1:5001/v1",
        key_url="https://github.com/CJackHwang/ds2api",
        environment_variable="DS2API_API_KEY",
        base_url_environment_variable="DS2API_BASE_URL",
    ),
}


class OpenAICompatibleClient:
    """Model discovery and native function calling over standard HTTP/SSE."""

    MAX_DISCOVERY_BYTES = 2_000_000

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
        self.base_url = self.definition.resolved_base_url()
        self.timeout_seconds = timeout_seconds
        self.max_output_tokens: Optional[int] = None
        self.max_stream_chars: Optional[int] = 96_000
        self.reasoning_control = "none"
        self.reasoning_effort = "off"
        self._model_metadata: Dict[str, Dict[str, Any]] = {}
        self.reliability = ProviderReliabilityController()
        self._capability_overrides: Dict[str, Any] = {}
        self._cancel_requested = threading.Event()
        self._response_lock = threading.Lock()
        self._active_response = None

    def cancel(self) -> None:
        """Cancel active SSE request and unblock its reader immediately."""
        self._cancel_requested.set()
        with self._response_lock:
            response = self._active_response
        if response is not None:
            try:
                response.close()
            except (OSError, ValueError):
                pass

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
            "User-Agent": "Fenrir-Agent/2.0",
        }
        if self.provider == "openrouter":
            headers.update(
                {
                    "HTTP-Referer": "https://github.com/mnisperuza/OpenCLI",
                    "X-OpenRouter-Title": "Fenrir Agent",
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

    @classmethod
    def _read_discovery_response(cls, response: Any) -> bytes:
        payload = response.read(cls.MAX_DISCOVERY_BYTES + 1)
        if len(payload) > cls.MAX_DISCOVERY_BYTES:
            raise ValueError("Provider model response exceeds configured limit")
        return payload

    def list_models(self) -> List[str]:
        request = Request(
            f"{self.base_url}/{self.definition.models_path}",
            headers=self._headers(),
            method="GET",
        )
        try:
            response_handle = self.reliability.call(
                lambda: safe_urlopen(request, timeout=30)
            )
            with response_handle as response:
                payload = json.loads(self._read_discovery_response(response).decode("utf-8"))
        except HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise self._safe_error(error, body) from error
        except (OSError, URLError, ValueError, json.JSONDecodeError) as error:
            raise self._safe_error(error) from error

        models = []
        items = payload.get("data", []) if isinstance(payload, dict) else []
        for item in items:
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
                provider_data = item.get("top_provider")
                if not isinstance(provider_data, dict):
                    provider_data = {}
                limits = item.get("limits")
                if not isinstance(limits, dict):
                    limits = {}
                model_metadata = item.get("metadata")
                if not isinstance(model_metadata, dict):
                    model_metadata = {}
                context = self._first_positive_int(
                    item.get("context_length"),
                    item.get("context_window"),
                    item.get("input_token_limit"),
                    item.get("inputTokenLimit"),
                    item.get("max_context_length"),
                    item.get("max_input_tokens"),
                    provider_data.get("context_length"),
                    provider_data.get("context_window"),
                    limits.get("context"),
                    limits.get("context_length"),
                    limits.get("input_tokens"),
                    model_metadata.get("context_length"),
                    model_metadata.get("context_window"),
                )
                output = self._first_positive_int(
                    item.get("max_output_tokens"),
                    item.get("output_token_limit"),
                    item.get("outputTokenLimit"),
                    item.get("maxOutputTokens"),
                    item.get("max_completion_tokens"),
                    provider_data.get("max_completion_tokens"),
                    provider_data.get("max_output_tokens"),
                    limits.get("output_tokens"),
                    limits.get("max_output_tokens"),
                    model_metadata.get("max_output_tokens"),
                )
                metadata: Dict[str, Any] = {
                    "supports_tools": not isinstance(supported, list)
                    or "tools" in supported,
                }
                if context:
                    metadata["context"] = context
                if output:
                    metadata["max_tokens"] = output
                self._model_metadata[model_id] = metadata
        return sorted(set(models), key=str.casefold)

    @staticmethod
    def _first_positive_int(*values: Any) -> Optional[int]:
        for value in values:
            if isinstance(value, bool):
                continue
            try:
                number = int(value)
            except (TypeError, ValueError):
                continue
            if number > 0:
                return number
        return None

    def model_metadata(self, model: Optional[str] = None) -> Dict[str, Any]:
        return dict(self._model_metadata.get(model or self.model, {}))

    def capability_report(self, model: Optional[str] = None) -> Dict[str, Any]:
        model_id = model or self.model
        metadata = self.model_metadata(model_id)
        profile = ProviderCapabilities(
            provider=self.provider,
            model=model_id,
            native_tools=True,
            named_tool_choice=self._capability_overrides.get("named_tool_choice"),
            parallel_tools=self._capability_overrides.get("parallel_tools"),
            strict_json_schema=self._capability_overrides.get("strict_json_schema"),
            streaming_tool_arguments=True,
            context_window=self._first_positive_int(metadata.get("context")),
            max_output_tokens=self._first_positive_int(metadata.get("max_tokens"), self.max_output_tokens),
            tokenizer="provider_reported_or_tiktoken",
            cancellation=True,
            observed_failures=self.reliability.status()["consecutive_failures"],
        )
        return {**profile.as_dict(), "transport": self.reliability.status()}

    def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        tool_choice: Any = "auto",
        _tool_repair_attempt: int = 0,
    ) -> Generator[Dict[str, Any], None, None]:
        if not self.model:
            raise ApiProviderError("No API model selected")
        self._cancel_requested.clear()
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }
        if self.provider == "groq" and _tool_repair_attempt:
            # Groq explicitly recommends a lower temperature after its server
            # rejects malformed tool JSON. The rejected call never executes,
            # so this retry is safe and does not repeat an external action.
            body["temperature"] = 0.2
        if isinstance(self.max_output_tokens, int) and self.max_output_tokens > 0:
            body["max_tokens"] = self.max_output_tokens
        if (
            self.reasoning_control == "api_parameter"
            and self.reasoning_effort in {"low", "medium", "high"}
        ):
            body["reasoning_effort"] = self.reasoning_effort
        if tools:
            body["tools"] = tools
            if tool_choice is not None:
                body["tool_choice"] = tool_choice
        request = Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        calls: Dict[int, Dict[str, Any]] = {}
        emitted_chars = 0
        provider_limited = False
        response_handle = None
        try:
            response_handle = self.reliability.call(
                lambda: safe_urlopen(request, timeout=self.timeout_seconds)
            )
            with self._response_lock:
                self._active_response = response_handle
            with response_handle as response:
                for raw_line in response:
                    if self._cancel_requested.is_set():
                        yield {"type": "cancelled", "content": "Generation cancelled."}
                        return
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
                        limit = self.max_stream_chars
                        if isinstance(limit, int) and limit > 0:
                            remaining = limit - emitted_chars
                            if remaining <= 0:
                                yield {
                                    "type": "output_limit",
                                    "content": "Fenrir Agent stopped oversized API output.",
                                }
                                return
                            if len(content) > remaining:
                                yield {"type": "token", "content": content[:remaining]}
                                yield {
                                    "type": "output_limit",
                                    "content": "Fenrir Agent stopped oversized API output.",
                                }
                                return
                        emitted_chars += len(content)
                        yield {"type": "token", "content": content}
                    if choices[0].get("finish_reason") == "length":
                        provider_limited = True
                    for tool_delta in delta.get("tool_calls") or []:
                        if not isinstance(tool_delta, dict):
                            continue
                        index = int(tool_delta.get("index", 0))
                        state = calls.setdefault(
                            index,
                            {"id": "", "name": "", "arguments": ""},
                        )
                        state["id"] += str(tool_delta.get("id") or "")
                        # Gemini's OpenAI-compatible endpoint attaches its
                        # required reasoning continuity token to the call.
                        # Keep the provider-owned structure byte-for-byte in
                        # the logical sense (never synthesize or transform it)
                        # so the adapter can return it on the next request.
                        extra_content = tool_delta.get("extra_content")
                        if not isinstance(extra_content, dict):
                            extra_content = tool_delta.get("extraContent")
                        if isinstance(extra_content, dict):
                            state["extra_content"] = dict(extra_content)
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
            if (
                self.provider == "groq"
                and tools
                and error.code == 400
                and _tool_repair_attempt < 1
                and self._is_groq_failed_tool_generation(body_text)
            ):
                yield from self.stream_chat(
                    messages, tools, tool_choice, _tool_repair_attempt + 1
                )
                return
            if (
                tools
                and tool_choice is not None
                and tool_choice != "auto"
                and tool_choice != "none"
                and error.code in {400, 404, 422}
            ):
                # Some OpenAI-compatible providers expose tools but reject
                # required/named tool_choice. Strict prompt + host validation remain.
                self._capability_overrides["named_tool_choice"] = False
                yield from self.stream_chat(messages, tools, "auto")
                return
            raise self._safe_error(error, body_text) from error
        except (OSError, URLError, ValueError) as error:
            if self._cancel_requested.is_set():
                yield {"type": "cancelled", "content": "Generation cancelled."}
                return
            raise self._safe_error(error) from error
        finally:
            with self._response_lock:
                if self._active_response is response_handle:
                    self._active_response = None

        if provider_limited:
            yield {
                "type": "output_limit",
                "content": "Provider stopped output at configured token limit.",
            }
        if calls:
            yield {
                "type": "tool_calls",
                "calls": [calls[index] for index in sorted(calls)],
            }

    @staticmethod
    def _is_groq_failed_tool_generation(body: str) -> bool:
        """Recognize Groq's documented rejected-tool payload without guessing."""
        try:
            payload = json.loads(body)
        except (TypeError, json.JSONDecodeError):
            return False
        error = payload.get("error") if isinstance(payload, dict) else None
        return isinstance(error, dict) and isinstance(error.get("failed_generation"), dict)


__all__ = [
    "ApiProviderError",
    "OpenAICompatibleClient",
    "PROVIDERS",
    "ProviderDefinition",
]
