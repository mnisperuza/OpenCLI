"""Structured web retrieval for OpenCLI.

Search and extraction live behind a small application-owned boundary so the
agent runtime is not coupled to DDGS response shapes.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


EventSink = Callable[[Dict[str, Any]], None]
PermissionCallback = Callable[[str, str, str, str], bool]
_TRACKING_PARAMETERS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "msclkid",
}


class WebRetrievalError(RuntimeError):
    """Raised when live web retrieval cannot return a usable observation."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _canonical_url(value: Any) -> Optional[str]:
    raw = str(value or "").strip()
    try:
        parts = urlsplit(raw)
    except ValueError:
        return None
    if parts.scheme.lower() not in {"http", "https"} or not parts.hostname:
        return None

    host = parts.hostname.lower().rstrip(".")
    try:
        port = parts.port
    except ValueError:
        return None
    default_port = (parts.scheme.lower() == "http" and port == 80) or (
        parts.scheme.lower() == "https" and port == 443
    )
    netloc = host if port is None or default_port else f"{host}:{port}"
    query = [
        (key, item)
        for key, item in parse_qsl(parts.query, keep_blank_values=True)
        if not key.lower().startswith("utm_")
        and key.lower() not in _TRACKING_PARAMETERS
    ]
    path = parts.path or "/"
    return urlunsplit(
        (parts.scheme.lower(), netloc, path, urlencode(query, doseq=True), "")
    )


def _is_public_web_url(url: str) -> bool:
    """Reject local/private destinations before asking DDGS to extract them."""
    parts = urlsplit(url)
    host = (parts.hostname or "").lower().rstrip(".")
    if not host or host == "localhost" or host.endswith(".localhost"):
        return False
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(host, None)}
    except socket.gaierror:
        return False
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if not ip.is_global:
            return False
    return bool(addresses)


class WebRetriever:
    """Live metasearch with normalized, bounded, structured observations."""

    def __init__(
        self,
        max_results: int = 10,
        max_content_chars: int = 20_000,
        event_sink: Optional[EventSink] = None,
        client_factory: Optional[Callable[[], Any]] = None,
        permission_callback: Optional[PermissionCallback] = None,
    ):
        self.max_results = max(1, max_results)
        self.max_content_chars = max(1_000, max_content_chars)
        self.event_sink = event_sink
        self._client_factory = client_factory
        self.permission_callback = permission_callback

    def _client(self) -> Any:
        if self._client_factory is not None:
            return self._client_factory()
        try:
            from ddgs import DDGS
        except ImportError as error:
            raise WebRetrievalError(
                "DDGS is unavailable. Install project dependencies."
            ) from error
        return DDGS()

    def _event(self, name: str, arguments: Dict[str, Any]) -> None:
        if self.event_sink:
            self.event_sink(
                {"type": "tool", "name": name, "arguments": arguments}
            )

    def _result(self, name: str, summary: str) -> None:
        if self.event_sink:
            self.event_sink(
                {"type": "tool_result", "name": name, "summary": summary}
            )

    def _allowed(self, action: str, target: str, reason: str) -> bool:
        if self.permission_callback is None:
            return True
        return self.permission_callback("web", action, target, reason)

    def web_search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Search the live web and return ranked, deduplicated source records.

        Args:
            query: Specific web search query.
            max_results: Desired result count, capped by runtime policy.
        """
        query = _clean_text(query)
        if not query:
            raise ValueError("Search query cannot be empty")
        limit = min(max(1, int(max_results)), self.max_results)
        if not self._allowed(
            "web_search", query, "Search live web for current information"
        ):
            self._result("web_search", "permission denied")
            return {
                "query": query,
                "result_count": 0,
                "results": [],
                "permission_denied": True,
            }
        self._event("web_search", {"query": query, "max_results": limit})

        # Fetch extras because canonicalization may collapse tracking variants.
        fetch_limit = min(max(limit * 2, limit), self.max_results * 2)
        try:
            raw_results = self._client().text(
                query,
                region="wt-wt",
                safesearch="moderate",
                max_results=fetch_limit,
                backend="auto",
            )
        except Exception as error:
            raise WebRetrievalError(f"Web search failed: {error}") from error

        results = []
        seen = set()
        for raw in raw_results or []:
            if not isinstance(raw, dict):
                continue
            url = _canonical_url(raw.get("href") or raw.get("url"))
            if not url or url in seen:
                continue
            seen.add(url)
            results.append(
                {
                    "rank": len(results) + 1,
                    "title": _clean_text(raw.get("title")),
                    "url": url,
                    "domain": urlsplit(url).hostname or "",
                    "snippet": _clean_text(
                        raw.get("body") or raw.get("description")
                    ),
                }
            )
            if len(results) >= limit:
                break

        output = {
            "query": query,
            "retrieved_at": _utc_now(),
            "result_count": len(results),
            "results": results,
        }
        self._result("web_search", f"{len(results)} results")
        return output

    def web_fetch(self, url: str) -> Dict[str, Any]:
        """Extract readable content from a public web result for grounding.

        Args:
            url: Public HTTP or HTTPS source URL returned by web_search.
        """
        canonical = _canonical_url(url)
        if not canonical:
            raise ValueError("URL must be a public HTTP or HTTPS destination")
        if not self._allowed(
            "web_fetch", canonical, "Read source content for grounded answer"
        ):
            self._result("web_fetch", "permission denied")
            return {"url": canonical, "permission_denied": True, "content": ""}
        if not _is_public_web_url(canonical):
            raise ValueError("URL must be a public HTTP or HTTPS destination")
        self._event("web_fetch", {"url": canonical})
        extracted = None
        error_text = ""
        # Extraction providers intermittently fail. One retry is enough to
        # cover transient faults without turning a bad page into a tool loop.
        for _attempt in range(2):
            try:
                extracted = self._client().extract(canonical, fmt="text_markdown")
                break
            except Exception as error:
                error_text = str(error)

        if extracted is None:
            output = {
                "url": canonical,
                "retrieved_at": _utc_now(),
                "content": "",
                "truncated": False,
                "error": f"Web fetch failed: {error_text}",
                "recoverable": True,
            }
            self._result("web_fetch", output["error"])
            return output

        content_value = (extracted or {}).get("content", "")
        if isinstance(content_value, bytes):
            content_value = content_value.decode("utf-8", errors="replace")
        content = str(content_value).strip()
        output = {
            "url": canonical,
            "retrieved_at": _utc_now(),
            "content": content[: self.max_content_chars],
            "truncated": len(content) > self.max_content_chars,
        }
        self._result("web_fetch", f"{len(output['content'])} characters")
        return output


__all__ = ["WebRetrievalError", "WebRetriever"]
