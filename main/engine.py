"""
OpenCLI Core Engine

By Matias Nisperuza — 2026

"""

import os
import json
import csv
import gc
import re
import importlib.util
import warnings
import threading
import time
import shutil
import subprocess
import atexit
import tempfile
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Generator, Optional, Dict, List, Tuple, Union

from main.logger import configure_logging

configure_logging()


def get_hf_token() -> Optional[str]:
    """Return optional Hugging Face token from current process environment."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

# ═══════════════════════════════════════════════════════════════════════════════
# ML IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════

import platform

ML_AVAILABLE = False
BNB_AVAILABLE = False
HQQ_AVAILABLE = False
torch = None
IS_WINDOWS = platform.system() == "Windows"
IS_MACOS = platform.system() == "Darwin"
AutoProcessor = None
AutoModelForImageTextToText = None
DynamicCache = None
QuantizedCache = None
HQQQuantizedLayer = None
CacheLayerMixin = None

PIL_AVAILABLE = False
Image = None
ImageGrab = None
try:
    from PIL import Image, ImageGrab
    PIL_AVAILABLE = True
except ImportError:
    pass

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import torch
        from transformers import (
            AutoTokenizer,
            AutoProcessor,
            AutoModelForCausalLM,
            BitsAndBytesConfig,
            DynamicCache,
            QuantizedCache,
        )
        from transformers.cache_utils import HQQQuantizedLayer, CacheLayerMixin
        from transformers import TextIteratorStreamer
        from transformers import logging as tf_logging
        from huggingface_hub.utils import logging as hf_logging
        try:
            from transformers import AutoModelForImageTextToText
        except ImportError:
            AutoModelForImageTextToText = None
        tf_logging.set_verbosity_error()
        hf_logging.set_verbosity_error()
        warnings.filterwarnings(
            "ignore",
            message=r".*[Uu]nauthenticated.*Hugging Face Hub.*",
            module=r"huggingface_hub.*",
        )
        ML_AVAILABLE = True

        BNB_AVAILABLE = importlib.util.find_spec("bitsandbytes") is not None
        HQQ_AVAILABLE = importlib.util.find_spec("hqq") is not None

except ImportError:
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# INTERRUPT HANDLER
# ═══════════════════════════════════════════════════════════════════════════════

class InterruptHandler:
    """Handles ESC key interrupts during generation"""

    def __init__(self):
        self.interrupted = False
        self._lock = threading.Lock()

    def reset(self):
        with self._lock:
            self.interrupted = False

    def interrupt(self):
        with self._lock:
            self.interrupted = True

    def is_interrupted(self) -> bool:
        with self._lock:
            return self.interrupted

# Global interrupt handler
_interrupt_handler = InterruptHandler()

def get_interrupt_handler() -> InterruptHandler:
    return _interrupt_handler


# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM SYSTEM PROMPTS
# ═══════════════════════════════════════════════════════════════════════════════

OPENCLI_CHAT_PROMPT = (
    "You are inside OpenCLI, a helpful AI assistant agentic CLI designed to help people learn, build, understand, create, and solve problems. You can speak many languages fluently."
)

LFM_DIRECT_RESPONSE_PROMPT = (
    OPENCLI_CHAT_PROMPT
    + "\n\nResponse contract: put only user-facing answer inside <final> and "
    "</final>. Never write anything outside these tags. Do not reveal private "
    "planning, drafting notes, self-critique, or statements such as 'I should' "
    "or 'Let me craft a response'."
)

SYSTEM_PROMPTS = {
    "auto": OPENCLI_CHAT_PROMPT,
}

DEFAULT_PROMPT = OPENCLI_CHAT_PROMPT


# ═══════════════════════════════════════════════════════════════════════════════
# MEMORY CLEANUP
# ═══════════════════════════════════════════════════════════════════════════════

def cleanup_memory():
    """Clean GPU/CPU memory"""
    try:
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
        gc.collect()
    except:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# CONTEXT COMPRESSION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ImageAttachment:
    source: str
    image: Any
    path: Optional[Path] = None


@dataclass
class InputPayload:
    prompt: str
    enhanced_prompt: str
    file_contents: List[str] = field(default_factory=list)
    file_paths: List[Path] = field(default_factory=list)
    image_attachments: List[ImageAttachment] = field(default_factory=list)
    clipboard_image_used: bool = False


class FileHandler:
    """Handles file operations with path awareness"""

    SUPPORTED_EXTENSIONS = {
        'code': ['.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.cpp', '.h', '.hpp',
                 '.cs', '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala'],
        'web': ['.html', '.htm', '.css', '.scss', '.sass', '.less', '.vue', '.svelte'],
        'data': ['.json', '.yaml', '.yml', '.xml', '.csv', '.toml', '.ini', '.env'],
        'doc': ['.md', '.txt', '.rst', '.log'],
        'image': ['.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp'],
    }

    def __init__(
        self,
        permission_callback: Optional[
            Callable[[str, str, str, str], bool]
        ] = None,
    ):
        self.current_path = Path.cwd()
        self.last_file = None
        self.permission_callback = permission_callback

    def _allowed(self, action: str, target: str, reason: str) -> bool:
        if self.permission_callback is None:
            return True
        return self.permission_callback("file_read", action, target, reason)

    def extract_paths(self, text: str) -> List[str]:
        """Extract file paths from text (marked with @ or detected)"""
        paths = []

        # Find @path references
        at_paths = re.findall(r'@((?:[A-Za-z]:)?[^\s,;!?\'"]+)', text)
        paths.extend(at_paths)

        # Find Windows / relative file-like patterns
        file_patterns = re.findall(r'(?:[A-Za-z]:)?[\w./\\ -]+\.[\w]+', text)
        for fp in file_patterns:
            candidate = fp.strip()
            if any(candidate.lower().endswith(ext) for exts in self.SUPPORTED_EXTENSIONS.values() for ext in exts):
                paths.append(candidate)

        return list(set(paths))

    def resolve_path(self, path_str: str) -> Optional[Path]:
        """Resolve a path string to absolute path"""
        try:
            path = Path(path_str.strip().strip('"').strip("'"))
            if path.is_absolute():
                return path if path.exists() else None

            # Try relative to current directory
            resolved = self.current_path / path
            if resolved.exists():
                return resolved

            # Try relative to home
            resolved = Path.home() / path
            if resolved.exists():
                return resolved

            return None
        except:
            return None

    def read_file(self, path: Path) -> Tuple[bool, str]:
        """Read a file safely"""
        try:
            if not self._allowed(
                "read_file", str(path.resolve()), "Attach file content to prompt"
            ):
                return False, f"Permission denied: {path}"
            if not path.exists():
                return False, f"File not found: {path}"

            if path.stat().st_size > 1_000_000:  # 1MB limit
                return False, f"File too large: {path}"

            content = path.read_text(encoding='utf-8', errors='ignore')
            self.last_file = path
            return True, content
        except Exception as e:
            return False, f"Error reading file: {e}"

    def load_image(self, path: Path) -> Tuple[bool, Optional[Any], str]:
        """Load an image file for multimodal processing."""
        if not PIL_AVAILABLE or Image is None:
            return False, None, "Pillow is not available"
        try:
            if not self._allowed(
                "load_image", str(path.resolve()), "Attach image to prompt"
            ):
                return False, None, f"Permission denied: {path}"
            if not path.exists():
                return False, None, f"File not found: {path}"
            image = Image.open(path).convert("RGB")
            self.last_file = path
            return True, image, ""
        except Exception as e:
            return False, None, f"Error loading image: {e}"

    def get_clipboard_image(self) -> Optional[ImageAttachment]:
        """Read an image directly from the Windows clipboard when available."""
        if not IS_WINDOWS or not PIL_AVAILABLE or ImageGrab is None:
            return None
        if not self._allowed(
            "read_clipboard", "clipboard", "Attach clipboard image to prompt"
        ):
            return None
        try:
            clipped = ImageGrab.grabclipboard()
        except Exception:
            return None

        if clipped is None:
            return None

        if isinstance(clipped, list):
            for item in clipped:
                candidate = Path(item)
                if candidate.suffix.lower() in self.SUPPORTED_EXTENSIONS["image"]:
                    ok, image, _ = self.load_image(candidate)
                    if ok:
                        return ImageAttachment(source=f"clipboard:{candidate.name}", image=image, path=candidate)
            return None

        if Image is not None and isinstance(clipped, Image.Image):
            return ImageAttachment(source="clipboard", image=clipped.convert("RGB"))

        return None

    def should_check_clipboard_image(self, prompt: str) -> bool:
        """Heuristic for auto-using a clipboard image without manual mode switching."""
        lower = (prompt or "").lower()
        triggers = (
            "clipboard",
            "image",
            "screenshot",
            "photo",
            "diagram",
            "scan",
            "ocr",
            "this picture",
            "this image",
        )
        return not lower.strip() or any(token in lower for token in triggers)

    def get_file_type(self, path: Path) -> str:
        """Get the type category of a file"""
        ext = path.suffix.lower()
        for category, extensions in self.SUPPORTED_EXTENSIONS.items():
            if ext in extensions:
                return category
        return "unknown"

    def navigate_to(self, path_str: str) -> Tuple[bool, str]:
        """Navigate to a directory"""
        try:
            path = Path(path_str).expanduser().resolve()
            if path.is_dir():
                self.current_path = path
                return True, f"Now at: {path}"
            elif path.is_file():
                self.current_path = path.parent
                return True, f"Now at: {path.parent}"
            return False, f"Path not found: {path_str}"
        except Exception as e:
            return False, f"Navigation error: {e}"


# ═══════════════════════════════════════════════════════════════════════════════
# THINKING TREE EXTRACTOR
# ═══════════════════════════════════════════════════════════════════════════════

class ThinkingTreeExtractor:
    """
    Streaming thinking-tag extractor.

    This extractor suppresses generated <think>...</think> blocks from visible output.
    """

    def __init__(self):
        self.current_thinking = ""
        self.in_thinking = False

    def reset(self):
        self.current_thinking = ""
        self.in_thinking = False

    def extract_thinking(self, text: str) -> Tuple[str, str]:
        """
        No-op extract; preserve the visible answer text.
        """
        return "", text

    def process_stream_token(self, token: str) -> Tuple[bool, str]:
        """
        Parse streaming tokens and suppress thinking segments.
        Returns (is_thinking, display_token).
        """
        if token is None:
            return False, ""

        if "<think>" in token:
            token = token.replace("<think>", "")
            self.in_thinking = True

        if "</think>" in token:
            token = token.replace("</think>", "")
            if token:
                self.current_thinking += token
            self.in_thinking = False
            return True, ""

        if self.in_thinking:
            self.current_thinking += token
            return True, ""

        return False, token

    def get_thinking(self) -> str:
        return self.current_thinking


class UntaggedReasoningFilter:
    """Hide plain-text planning from models that do not emit thinking tags."""

    _FINAL_OPEN_TAG = re.compile(r"<final>", re.IGNORECASE)
    _FINAL_CLOSE_TAG = re.compile(r"</final>", re.IGNORECASE)

    _PLANNING_START = re.compile(
        r"^(?:the\s+(?:user|question|prompt)\b|the\s+is\s+asking\b|"
        r"i\s+(?:need|should|must)\b|we\s+need\b)",
        re.IGNORECASE,
    )
    _FINAL_TRANSITION = re.compile(
        r"\b(?:let\s+me|i(?:'ll|\s+will))\s+"
        r"(?:provide|craft|give|offer|write|formulate|produce)\s+"
        r"(?:a\s+)?(?:(?:clear|concise|helpful|direct|final)\s+)?"
        r"(?:answer|response)?\s*:?\s*",
        re.IGNORECASE,
    )
    _PICKED_ANSWER_TRANSITION = re.compile(
        r"\b(?:let\s+me|i(?:'ll|\s+will))\s+"
        r"(?:pick|choose|use|go\s+with)\s+"
        r"(?:a\s+|an\s+)?"
        r"(?:(?:classic|clean|short|simple|quick|good|family[-\s]friendly)\s*,?\s*)*"
        r"(?:joke\s*[:\-]?\s*)?"
        r"(?=(?:don['’]t|why|what|how|when|where|who|did|do|does|can|could|would)\b|"
        r"here(?:'s|\s+is)\b|[\"“])",
        re.IGNORECASE,
    )
    _THOUGHT_TO_ANSWER_TRANSITION = re.compile(
        r"\b(?:let\s+me|i(?:'ll|\s+will))\s+"
        r"(?:think\s+of|come\s+up\s+with)\s+"
        r"(?:a\s+|an\s+)?(?:joke|answer|response|example)[.!:]\s*",
        re.IGNORECASE,
    )

    def reset(self, enabled: bool = False):
        self.enabled = enabled
        self.buffer = ""
        self.filtering = False
        self.passthrough = False
        self.in_final_tag = False
        self.final_tag_closed = False
        self.final_tag_buffer = ""

    def process(self, token: str) -> str:
        if not self.enabled:
            return token

        if self.final_tag_closed:
            return ""

        if self.in_final_tag:
            return self._consume_final_tag(token)

        if self.passthrough:
            return token

        self.buffer += token
        final_tag = self._FINAL_OPEN_TAG.search(self.buffer)
        if final_tag:
            self.in_final_tag = True
            content = self.buffer[final_tag.end():]
            self.buffer = ""
            return self._consume_final_tag(content)

        if not self.filtering:
            sample = self.buffer.lstrip()
            if self._PLANNING_START.match(sample):
                self.filtering = True
            elif len(sample) >= 96:
                self.passthrough = True
                result, self.buffer = self.buffer, ""
                return result
            else:
                return ""

        transition = (
            self._FINAL_TRANSITION.search(self.buffer)
            or self._THOUGHT_TO_ANSWER_TRANSITION.search(self.buffer)
            or self._PICKED_ANSWER_TRANSITION.search(self.buffer)
        )
        if transition:
            self.passthrough = True
            result = self.buffer[transition.end():]
            self.buffer = ""
            return result
        return ""

    def _consume_final_tag(self, text: str) -> str:
        """Return final-tag contents and discard everything after its closing tag."""
        self.final_tag_buffer += text
        closing_tag = self._FINAL_CLOSE_TAG.search(self.final_tag_buffer)
        if not closing_tag:
            return ""
        self.in_final_tag = False
        self.final_tag_closed = True
        result = self.final_tag_buffer[:closing_tag.start()]
        self.final_tag_buffer = ""
        return result

    def finish(self) -> str:
        """Drop unfinished planning rather than exposing it as an answer."""
        if self.in_final_tag:
            result, self.final_tag_buffer = self.final_tag_buffer, ""
            return result
        if not self.enabled or self.passthrough:
            return ""
        self.buffer = ""
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
# OPENCLI ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class OpenCLIEngine:
    MODELS = {

    "auto": {
        "path": "mistralai/Ministral-3-14B-Instruct-2512-GGUF",
        "name": "Ministral 3 14B Instruct",
        "family": "mistral",
        "max_tokens": 8192,
        "temp": 0.05,
        "top_k": 40,
        "repetition_penalty": 1.05,
        "context": 32768,
        "vram": "~8.2GB Q4_K_M + Q4 KV cache",
        "has_thinking": False,
        "supports_vision": True,
        "backend": "llama_cpp",
        "locked": False,
    },
    "ministral-3-14b": {
            "path": "mistralai/Ministral-3-14B-Instruct-2512-GGUF",
            "name": "Ministral 3 14B Instruct",
            "family": "mistral",
            "max_tokens": 8192,
            "temp": 0.05,
            "top_k": 40,
            "repetition_penalty": 1.05,
            "context": 32768,
            "vram": "~8.2GB Q4_K_M + Q4 KV cache",
            "has_thinking": False,
            "supports_vision": True,
            "backend": "llama_cpp",
            "locked": False,
    },
    "gpt-oss-20b": {
        "path": "unsloth/gpt-oss-20b-GGUF",
        "name": "GPT-OSS 20B",
        "family": "auto",
        "max_tokens": 16384,
        "temp": 0.7,
        "top_k": 20,
        "repetition_penalty": 1.0,
        "context": 32768,
        "vram": "~11.6GB Q4_K_M + Q4 KV cache",
        "has_thinking": True,
        "supports_vision": False,
        "backend": "llama_cpp",
        "locked": False,
    },

    "devstral-small-2-24b": {
        "path": "bartowski/mistralai_Devstral-Small-2-24B-Instruct-2512-GGUF",
        "name": "Devstral Small 2 24B",
        "family": "mistral",
        "max_tokens": 8192,
        "temp": 0.15,
        "top_k": 40,
        "repetition_penalty": 1.05,
        "context": 32768,
        "vram": "~14.3GB Q4_K_M + Q4 KV cache",
        "has_thinking": False,
        "supports_vision": True,
        "backend": "llama_cpp",
        "locked": False,
    },

    "qwen3.8-27b": {
        "path": "unsloth/Qwen3.8-27B-GGUF",
        "name": "Qwen 3.8 27B",
        "family": "qwen",
        "max_tokens": 16384,
        "temp": 1.0,
        "top_k": 20,
        "repetition_penalty": 1.0,
        "context": 32768,
        "vram": "~18.0GB Q4_K_M + Q4 KV cache",
        "has_thinking": True,
        "supports_vision": True,
        "backend": "llama_cpp",
        "locked": False,
    },

}

    MODEL_ALIASES = {}

    def __init__(self):
        # Force CUDA detection
        self.device = self._detect_device()
        self.MODELS = {
            key: dict(model_info) for key, model_info in self.MODELS.items()
        }

        self.model = None
        self.tokenizer = None
        self.processor = None
        self.backend = None
        self.api_client = None
        self._llama_process = None
        self._llama_log = None
        self._llama_log_path = None
        atexit.register(self.shutdown)
        self.current_mode = "auto"  # Changed from "mini"
        self.current_quant = "int4"
        self.file_handler = FileHandler()
        self.thinking_extractor = ThinkingTreeExtractor()
        self.untagged_reasoning_filter = UntaggedReasoningFilter()
        self.interrupt_handler = get_interrupt_handler()

        # Generation state
        self._generating = False
        self._current_response = ""

    def _detect_device(self) -> str:
        """Force CUDA detection - be aggressive about finding GPU"""
        if not ML_AVAILABLE or not torch:
            return "cpu"

        # Try CUDA first
        if torch.cuda.is_available():
            try:
                torch.cuda.init()
                device_count = torch.cuda.device_count()
                if device_count > 0:
                    return "cuda"
            except Exception:
                pass

        # Try MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            try:
                return "mps"
            except Exception:
                pass

        return "cpu"

    def get_device_info(self) -> str:
        """Get device info string"""
        if self.backend == "remote_api" and self.api_client is not None:
            return f"API: {self.api_client.provider_name}"
        if self.backend == "llama_cpp":
            return "llama.cpp server"
        if self.device == "cuda" and torch:
            try:
                name = torch.cuda.get_device_name(0)
                return f"GPU: {name}"
            except:
                return "GPU: CUDA"
        elif self.device == "mps":
            return "GPU: Apple Silicon"
        return "CPU"

    @staticmethod
    def _llama_cpp_url() -> str:
        url = os.environ.get(
            "OPENCLI_LLAMA_CPP_URL", "http://127.0.0.1:8080/v1"
        ).rstrip("/")
        if urlparse(url).hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("OPENCLI_LLAMA_CPP_URL must use a local loopback host")
        return url

    @staticmethod
    def _llama_cpp_is_ready(base_url: str) -> bool:
        try:
            with urlopen(f"{base_url}/models", timeout=2) as response:
                response.read(1)
            return True
        except (URLError, OSError):
            return False

    @staticmethod
    def _llama_cpp_model_ids(base_url: str) -> List[str]:
        """Return model IDs reported by an already-running llama.cpp server."""
        try:
            with urlopen(f"{base_url}/models", timeout=2) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (URLError, OSError, UnicodeDecodeError, json.JSONDecodeError):
            return []
        entries = payload.get("data") or payload.get("models") or []
        model_ids = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            model_id = entry.get("id") or entry.get("model") or entry.get("name")
            if model_id:
                model_ids.append(str(model_id))
        return model_ids

    @staticmethod
    def _llama_cpp_model_matches(expected: str, actual: str) -> bool:
        expected_key = expected.casefold().rstrip("/")
        actual_key = actual.casefold().rstrip("/")
        return expected_key in actual_key or actual_key in expected_key

    @staticmethod
    def _find_llama_cpp_executable() -> Optional[str]:
        """Find llama.cpp from PATH or Windows WinGet package storage."""
        executable = shutil.which("llama-server") or shutil.which("llama")
        if executable:
            return executable
        if os.name != "nt":
            return None

        package_root = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
        for name in ("llama-server.exe", "llama.exe"):
            matches = sorted(package_root.glob(f"ggml.llamacpp_*/{name}"))
            if matches:
                return str(matches[-1])
        return None

    @staticmethod
    def _gguf_quant_tag(quant: str) -> str:
        return {
            "int4": "Q4_K_M",
            "int8": "Q8_0",
            "fp16": "F16",
            "fp32": "F32",
        }.get((quant or "int4").lower(), "Q4_K_M")

    @staticmethod
    def _llama_cpp_startup_timeout() -> float:
        """Return bounded server startup time; allow local configuration."""
        raw_value = os.environ.get(
            "OPENCLI_LLAMA_CPP_STARTUP_TIMEOUT", "900"
        )
        try:
            return max(1.0, float(raw_value))
        except ValueError:
            return 900.0

    def _start_llama_cpp_server(
        self, model_info: Dict[str, Any], quant: str
    ) -> Tuple[bool, str]:
        """Start a local llama.cpp server for a selected GGUF model."""
        base_url = self._llama_cpp_url()
        parsed = urlparse(base_url)
        if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            return False, "Automatic GGUF startup only supports a local llama.cpp URL."

        executable = self._find_llama_cpp_executable()
        if not executable:
            return False, "llama.cpp is required for GGUF models. Install it with: winget install llama.cpp"

        port = parsed.port or 8080
        if model_info.get("source_type") == "local":
            command = [executable, "-m", model_info["path"]]
        else:
            command = [executable, "-hf", model_info["path"]]
            llama_file = model_info.get("llama_file")
            if llama_file:
                command.extend(["-hff", llama_file])
            else:
                command[-1] = (
                    f"{model_info['path']}:{self._gguf_quant_tag(quant)}"
                )

        command.extend(
            [
                "--gpu-layers", "auto",
                "--fit", "on",
                "--kv-offload",
                "--cache-type-k", "q4_0",
                "--cache-type-v", "q4_0",
                "--ctx-size", str(model_info.get("context", 32768)),
                *model_info.get("llama_args", []),
                "--port", str(port),
            ]
        )
        startup_info = None
        if os.name == "nt":
            startup_info = subprocess.STARTUPINFO()
            startup_info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startup_info.wShowWindow = subprocess.SW_HIDE
        try:
            self._llama_log_path = Path(tempfile.gettempdir()) / "bert-llama-server.log"
            self._llama_log = open(self._llama_log_path, "w", encoding="utf-8")
            # Keep the server outside this console's Ctrl+C group.  Ctrl+C is a
            # generation interrupt in OpenCLI; it must not kill the loaded model.
            creationflags = 0
            if os.name == "nt":
                creationflags = (
                    subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
                )
            self._llama_process = subprocess.Popen(
                command,
                stdout=self._llama_log,
                stderr=subprocess.STDOUT,
                startupinfo=startup_info,
                creationflags=creationflags,
            )
        except OSError as error:
            if self._llama_log is not None:
                self._llama_log.close()
                self._llama_log = None
            return False, f"Could not start llama.cpp: {error}"

        timeout_seconds = self._llama_cpp_startup_timeout()
        deadline = time.monotonic() + timeout_seconds
        while True:
            if self._llama_cpp_is_ready(base_url):
                return True, ""
            if self._llama_process.poll() is not None:
                self._llama_process = None
                detail = self._llama_log_tail()
                self.shutdown()
                return False, f"llama.cpp stopped during startup. {detail}"
            if time.monotonic() >= deadline:
                detail = self._llama_log_tail()
                self.shutdown()
                message = (
                    "llama.cpp did not become ready within "
                    f"{timeout_seconds:g} seconds."
                )
                return False, f"{message} {detail}".rstrip()
            time.sleep(0.5)


    def _llama_log_tail(self) -> str:
        if self._llama_log is not None:
            self._llama_log.flush()
        if self._llama_log_path is None or not self._llama_log_path.exists():
            return ""
        try:
            lines = self._llama_log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            return " ".join(lines[-3:])[-800:]
        except OSError:
            return ""

    def _load_llama_cpp_model(self, mode: str, model_info: Dict[str, Any], quant: str) -> Tuple[bool, str]:
        """Use a running llama.cpp OpenAI-compatible server for GGUF models."""
        base_url = self._llama_cpp_url()
        server_ready = self._llama_cpp_is_ready(base_url)
        if server_ready:
            model_ids = self._llama_cpp_model_ids(base_url)
            expected = model_info["path"]
            if model_ids and not any(
                self._llama_cpp_model_matches(expected, model_id)
                for model_id in model_ids
            ):
                loaded = ", ".join(model_ids)
                self.shutdown()
                if self._llama_cpp_is_ready(base_url):
                    return False, (
                        f"llama.cpp at {base_url} is serving {loaded}, not {expected}. "
                        "OpenCLI could not stop that server."
                    )
                server_ready = False
        if not server_ready:
            started, message = self._start_llama_cpp_server(model_info, quant)
            if not started:
                return False, message

        self.model = {"backend": "llama_cpp", "base_url": base_url}
        self.backend = "llama_cpp"
        self.current_mode = mode
        self.current_quant = quant
        return True, f"{model_info['name']} connected through llama.cpp"

    def _stop_unmanaged_windows_llama_cpp_server(self) -> None:
        """Stop only a local llama executable listening on OpenCLI's configured port."""
        if os.name != "nt":
            return
        try:
            port = str(urlparse(self._llama_cpp_url()).port or 8080)
            listeners = subprocess.run(
                ["netstat", "-ano", "-p", "tcp"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.splitlines()
        except (OSError, ValueError):
            return

        for line in listeners:
            columns = line.split()
            if len(columns) < 5 or columns[-2].upper() != "LISTENING":
                continue
            if not columns[1].endswith(f":{port}"):
                continue
            pid = columns[-1]
            if not pid.isdigit():
                continue
            try:
                task = subprocess.run(
                    ["tasklist", "/FO", "CSV", "/NH", "/FI", f"PID eq {pid}"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                row = next(csv.reader(task.stdout.splitlines()), [])
                image_name = row[0].casefold() if row else ""
                if image_name not in {"llama-server.exe", "llama.exe"}:
                    continue
                subprocess.run(
                    ["taskkill", "/PID", pid, "/T", "/F"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
            except OSError:
                continue

    def shutdown(self) -> None:
        """Stop the local llama.cpp server and release its loaded model memory."""
        process = self._llama_process
        self._llama_process = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        elif process is None:
            self._stop_unmanaged_windows_llama_cpp_server()
        if self._llama_log is not None:
            self._llama_log.close()
            self._llama_log = None

    def _get_quant_config(self, quant: str):
        """Get quantization config"""
        if not ML_AVAILABLE:
            return None

        if quant == "int4" and BNB_AVAILABLE and self.device == "cuda":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif quant == "int8" and BNB_AVAILABLE and self.device == "cuda":
            return BitsAndBytesConfig(
                load_in_8bit=True,
            )
        return None

    def _new_q4_kv_cache(self):
        """Build an INT4 HQQ cache while preserving hybrid convolution layers."""
        if not HQQ_AVAILABLE:
            raise RuntimeError(
                "Q4 KV cache requires HQQ. Install project dependencies or run: pip install hqq"
            )
        if self.model is None:
            raise RuntimeError("Cannot create KV cache before model load")

        config = self.model.config
        if DynamicCache is not None and HQQQuantizedLayer is not None and CacheLayerMixin is not None:
            cache = DynamicCache(config=config)
            layers = getattr(cache, "layers", None)
            if layers:
                for index, layer in enumerate(layers):
                    if isinstance(layer, CacheLayerMixin):
                        layers[index] = HQQQuantizedLayer(
                            nbits=4,
                            axis_key=1,
                            axis_value=1,
                        )
                return cache

        if QuantizedCache is None:
            raise RuntimeError("Installed Transformers version does not support quantized KV cache")
        return QuantizedCache(
            backend="hqq",
            config=config,
            nbits=4,
            axis_key=1,
            axis_value=1,
        )

    def _move_inputs_to_runtime_device(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Move processor/tokenizer tensors to the active runtime device."""
        if self.device == "cpu":
            return inputs

        target_device = getattr(self.model, "device", None)
        if target_device is None and hasattr(self.model, "hf_device_map"):
            target_device = self.device
        if target_device is None:
            return inputs

        moved = {}
        for key, value in inputs.items():
            moved[key] = value.to(target_device) if hasattr(value, "to") else value
        return moved

    def _build_multimodal_messages(
        self,
        system_prompt: str,
        prompt: str,
        image_attachments: List[ImageAttachment],
    ) -> List[Dict[str, Any]]:
        """Build a processor-friendly multimodal message payload."""
        content: List[Dict[str, Any]] = []
        if prompt.strip():
            content.append({"type": "text", "text": prompt})
        else:
            content.append({"type": "text", "text": "Analyze the attached image."})

        for attachment in image_attachments:
            content.append({"type": "image", "image": attachment.image})

        return [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": content},
        ]

    def _tokenize_payload(
        self,
        payload: InputPayload,
        system_prompt: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        model_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Prepare token or multimodal processor inputs for generation."""
        if payload.image_attachments and self.processor is not None and model_info.get("supports_vision"):
            mm_messages = self._build_multimodal_messages(
                system_prompt,
                payload.enhanced_prompt,
                payload.image_attachments,
            )
            if hasattr(self.processor, "apply_chat_template"):
                prompt_text = self.processor.apply_chat_template(
                    mm_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                prompt_text = payload.enhanced_prompt

            images = [attachment.image for attachment in payload.image_attachments]
            if callable(self.processor):
                try:
                    return self.processor(
                        text=prompt_text,
                        images=images,
                        return_tensors="pt",
                        padding=True,
                    )
                except TypeError:
                    return self.processor(
                        prompt_text,
                        images,
                        return_tensors="pt",
                    )

        if hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                input_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception:
                input_text = f"{system_prompt}\n\nUser: {payload.enhanced_prompt}\nAssistant:"
        else:
            input_text = f"{system_prompt}\n\nUser: {payload.enhanced_prompt}\nAssistant:"

        return self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=model_info.get("context", 4096) - max_tokens
        )

    @staticmethod
    def _system_prompt_for(model_info: Dict[str, Any]) -> str:
        """Return a model override without changing shared model behavior."""
        return model_info.get(
            "system_prompt",
            SYSTEM_PROMPTS.get(model_info.get("family", "auto"), DEFAULT_PROMPT),
        )

    def load_model(self, mode: str = "auto", quant: str = "int4",
                   progress_callback=None) -> Tuple[bool, str]:
        """Load a model with progress updates"""

        # Resolve model alias.
        mode = self.MODEL_ALIASES.get(mode, mode)

        if mode not in self.MODELS:
            return False, f"Unknown model: {mode}"

        model_info = self.MODELS[mode]
        model_path = model_info["path"]

        if model_info.get("locked") or not model_path:
            return False, (
                f"{model_info['name']} is currently locked. "
                "No backend model has been assigned yet."
            )

        if model_info.get("backend") == "llama_cpp":
            self.unload_model()
            return self._load_llama_cpp_model(mode, model_info, quant)

        if not ML_AVAILABLE:
            return False, "PyTorch/Transformers not available"

        # Unload current model
        self.unload_model()

        try:
            if progress_callback:
                progress_callback(f"📦 Loading {model_info['name']} ({model_path})...")

            # Get quantization config
            quant_config = self._get_quant_config(quant)

            # Determine dtype
            if quant in ["fp16"] and self.device in ["cuda", "mps"]:
                dtype = torch.float16
            elif quant == "fp32" or self.device == "cpu":
                dtype = torch.float32
            else:
                dtype = torch.float16 if self.device != "cpu" else torch.float32

            # Load tokenizer / processor
            token = get_hf_token()
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                padding_side='left',
                token=token,
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.processor = None
            if model_info.get("supports_vision") and AutoProcessor is not None:
                try:
                    self.processor = AutoProcessor.from_pretrained(
                        model_path,
                        trust_remote_code=True,
                        token=token,
                    )
                except Exception:
                    self.processor = None

            # Load model
            load_kwargs = {
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }

            if quant_config and self.device == "cuda":
                load_kwargs["quantization_config"] = quant_config
                load_kwargs["device_map"] = "auto"
            else:
                load_kwargs["torch_dtype"] = dtype
                if self.device != "cpu":
                    load_kwargs["device_map"] = "auto"

            model_cls = AutoModelForCausalLM
            if model_info.get("supports_vision") and AutoModelForImageTextToText is not None:
                model_cls = AutoModelForImageTextToText

            self.model = model_cls.from_pretrained(
                model_path,
                token=token,
                **load_kwargs
            )

            # Move to device if needed
            if "device_map" not in load_kwargs and self.device != "cpu":
                self.model = self.model.to(self.device)

            self.model.eval()
            self.backend = "transformers"

            self.current_mode = mode
            self.current_quant = quant
            if progress_callback:
                progress_callback(f"✓ {model_info['name']} loaded ({quant.upper()})")

            return True, f"{model_info['name']} loaded successfully"

        except Exception as e:
            cleanup_memory()
            error_msg = str(e)
            if "CUDA" in error_msg or "memory" in error_msg.lower():
                return False, f"GPU memory error: Try a smaller model or different quantization"
            return False, f"Failed to load model: {error_msg}"

    def register_models(self, models: Dict[str, Dict[str, Any]]) -> None:
        """Register validated user model profiles for this engine instance."""
        self.MODELS.update({key: dict(value) for key, value in models.items()})

    def configure_api(self, client: Any) -> Tuple[bool, str]:
        """Unload local inference and activate one session-only API client."""
        self.unload_model()
        self.api_client = client
        self.model = client
        self.backend = "remote_api"
        self.current_mode = "api"
        self.current_quant = "api"
        self.MODELS["api"] = {
            "path": client.model,
            "name": f"{client.provider_name}: {client.model}",
            "family": "auto",
            "max_tokens": 8192,
            "temp": 0.2,
            "context": 128000,
            "vram": "Remote",
            "has_thinking": False,
            "supports_vision": False,
            "backend": "remote_api",
            "locked": False,
        }
        return True, f"{client.provider_name} API ready: {client.model}"

    def unload_model(self):
        """Unload current model"""
        self.shutdown()
        if self.model is not None:
            del self.model
            self.model = None
        self.backend = None
        self.api_client = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        cleanup_memory()

    def _generate_llama_cpp_stream(
        self, payload: InputPayload, model_info: Dict[str, Any], max_tokens: int
    ) -> Generator[dict, None, None]:
        """Stream text from a llama.cpp OpenAI-compatible server."""
        system_prompt = self._system_prompt_for(model_info)
        request_body = {
            "model": model_info["path"],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": payload.enhanced_prompt},
            ],
            "stream": True,
            "max_tokens": max_tokens,
            "temperature": model_info.get("temp", 0.6),
            "top_p": model_info.get("top_p", 0.9),
            "top_k": model_info.get("top_k", 40),
            "repeat_penalty": model_info.get("repetition_penalty", 1.1),
            "repeat_last_n": model_info.get("repeat_last_n", 128),
            "stop": ["<|im_end|>", "<|endoftext|>", "\nUser:", "\nuser:"],
        }
        request = Request(
            f"{self._llama_cpp_url()}/chat/completions",
            data=json.dumps(request_body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        response_text = ""
        chunks = 0
        streamed_tool_calls: Dict[int, Dict[str, str]] = {}
        try:
            with urlopen(request, timeout=300) as response:
                for raw_line in response:
                    if self.interrupt_handler.is_interrupted():
                        yield {"type": "status", "content": "\n[Generation stopped by user]"}
                        break
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        delta = json.loads(data)["choices"][0].get("delta", {})
                        content = delta.get("content") or ""
                    except (KeyError, IndexError, TypeError, json.JSONDecodeError):
                        continue
                    if content:
                        response_text += content
                        chunks += 1
                        yield {"type": "token", "content": content}
                        if self._is_garbage_output(response_text[-160:]):
                            yield {"type": "status", "content": "\n[Output cleaned]"}
                            self._current_response = self._clean_response(response_text)
                            yield {
                                "type": "done",
                                "content": self._current_response,
                                "thinking": "",
                                "tokens_used": chunks,
                                "response_tokens": chunks,
                            }
                            return
                    for tool_delta in delta.get("tool_calls") or []:
                        if not isinstance(tool_delta, dict):
                            continue
                        index = int(tool_delta.get("index", 0))
                        state = streamed_tool_calls.setdefault(
                            index, {"name": "", "arguments": ""}
                        )
                        function = tool_delta.get("function") or {}
                        if not isinstance(function, dict):
                            continue
                        name = function.get("name")
                        if name:
                            state["name"] += str(name)
                        arguments = function.get("arguments")
                        if isinstance(arguments, dict):
                            state["arguments"] = json.dumps(
                                arguments, ensure_ascii=False
                            )
                        elif arguments:
                            state["arguments"] += str(arguments)
        except (URLError, OSError) as error:
            detail = self._llama_log_tail()
            suffix = f" Server log: {detail}" if detail else ""
            yield {"type": "error", "content": f"llama.cpp request failed: {error}.{suffix}"}
            return

        for index in sorted(streamed_tool_calls):
            tool_call = streamed_tool_calls[index]
            name = tool_call["name"]
            if not name:
                continue
            raw_arguments = tool_call["arguments"] or "{}"
            try:
                arguments = json.loads(raw_arguments)
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(arguments, dict):
                continue
            normalized = json.dumps(
                {"name": name, "arguments": arguments}, ensure_ascii=False
            )
            content = f"<tool_call>{normalized}</tool_call>"
            response_text += content
            chunks += 1
            yield {"type": "token", "content": content}

        self._current_response = self._clean_response(response_text)
        yield {
            "type": "done",
            "content": self._current_response,
            "thinking": "",
            "tokens_used": chunks,
            "response_tokens": chunks,
        }

    def generate_stream(self, prompt: Union[str, InputPayload], max_new_tokens: int = None) -> Generator[dict, None, None]:
        """
        Generate response with streaming.
        Yields dicts with: type (token/thinking/status/done), content, tokens_used
        """
        if not self.model:
            yield {"type": "error", "content": "No model loaded"}
            return

        self._generating = True
        self._current_response = ""
        self.interrupt_handler.reset()
        self.thinking_extractor.reset()

        payload = prompt if isinstance(prompt, InputPayload) else self.prepare_input_payload(prompt)
        model_info = self.MODELS.get(self.current_mode, {})
        self.untagged_reasoning_filter.reset(model_info.get("hide_untagged_reasoning", False))
        max_tokens = max_new_tokens or model_info.get("max_tokens", 2000)

        if self.backend == "llama_cpp":
            yield from self._generate_llama_cpp_stream(payload, model_info, max_tokens)
            self._generating = False
            return

        if not self.tokenizer:
            yield {"type": "error", "content": "No tokenizer loaded"}
            return

        # Build messages
        system_prompt = self._system_prompt_for(model_info)

        messages = [{"role": "system", "content": system_prompt}]
        messages.append({"role": "user", "content": payload.enhanced_prompt})

        try:
            inputs = self._tokenize_payload(
                payload=payload,
                system_prompt=system_prompt,
                messages=messages,
                max_tokens=max_tokens,
                model_info=model_info,
            )

            inputs = self._move_inputs_to_runtime_device(inputs)

            # Setup streamer
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )

            # Generation kwargs
            gen_kwargs = {
                "input_ids": inputs['input_ids'],
                "max_new_tokens": max_tokens,
                "temperature": model_info.get("temp", 0.7),
                "do_sample": model_info.get("do_sample", True),
                "top_p": model_info.get("top_p", 0.9),
                "top_k": model_info.get("top_k", 50),
                "repetition_penalty": model_info.get("repetition_penalty", 1.15),
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "streamer": streamer,
                "use_cache": True,
                "past_key_values": self._new_q4_kv_cache(),
            }
            if "attention_mask" in inputs:
                gen_kwargs["attention_mask"] = inputs["attention_mask"]

            # Start generation in thread
            def generate_thread():
                try:
                    with torch.no_grad():
                        self.model.generate(**gen_kwargs)
                except Exception:
                    pass

            thread = threading.Thread(target=generate_thread)
            thread.start()

            # Stream tokens
            tokens_generated = 0
            full_response = ""

            for token in streamer:
                # Check for interrupt
                if self.interrupt_handler.is_interrupted():
                    yield {"type": "status", "content": "\n[Generation stopped by user]"}
                    break

                is_thinking, display_token = self.thinking_extractor.process_stream_token(token)
                if is_thinking:
                    continue
                if display_token:
                    display_token = self.untagged_reasoning_filter.process(display_token)
                    if not display_token:
                        continue
                    # Check for prompt leak (model outputting system prompt)
                    if "You are OpenCLI" in display_token or "by Biwa" in display_token:
                        continue  # Skip leaked prompt content

                    # Skip role tags
                    if display_token.strip() in ['assistant', 'user', 'system', '###', 'Human:', 'Assistant:']:
                        continue

                    full_response += display_token
                    tokens_generated += 1

                    # Check for garbage output and stop early
                    if self._is_garbage_output(full_response[-100:]):
                        yield {"type": "status", "content": "\n[Output cleaned]"}
                        break

                    yield {"type": "token", "content": display_token}

            trailing_response = self.untagged_reasoning_filter.finish()
            if trailing_response:
                full_response += trailing_response
                tokens_generated += 1
                yield {"type": "token", "content": trailing_response}

            thread.join(timeout=1.0)

            # Clean up response
            full_response = self._clean_response(full_response)
            self._current_response = full_response

            total_tokens = tokens_generated

            yield {
                "type": "done",
                "content": full_response,
                "thinking": "",
                "tokens_used": total_tokens,
                "response_tokens": tokens_generated,
            }

        except Exception as e:
            yield {"type": "error", "content": str(e)}

        finally:
            self._generating = False

    def generate_runtime_stream(
        self, prompt: str, max_new_tokens: int = None
    ) -> Generator[dict, None, None]:
        """Stream an already-expanded agent prompt without scanning it for files."""
        payload = InputPayload(prompt=prompt, enhanced_prompt=prompt)
        yield from self.generate_stream(payload, max_new_tokens=max_new_tokens)

    def _clean_response(self, text: str) -> str:
        """Clean up model response"""
        # Remove any leaked system prompts
        patterns_to_remove = [
            r'You are OpenCLI[^.]*\.',
            r'by Biwa[^.]*\.',
            r'Never reveal[^.]*\.',
            r'Never output[^.]*\.',
            r'system:.*?(?=user:|assistant:|$)',
            r'^assistant\s*',  # Remove leading "assistant" role tag
            r'<\|assistant\|>',
            r'<\|user\|>',
            r'<\|system\|>',
            r'</?think>',
        ]

        for pattern in patterns_to_remove:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)

        # Remove repeated newlines
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Remove garbage patterns (repeated special chars)
        text = re.sub(r'[{}\[\]()]{5,}', '', text)
        text = re.sub(r'[\'\"]{5,}', '', text)

        return text.strip()

    def _is_garbage_output(self, text: str) -> bool:
        """Detect if output is garbage/repetitive"""
        # Check for repeated special characters
        if re.search(r'[{}\[\]()\'\"]{10,}', text):
            return True
        # Check for repeated words
        words = text.split()
        if len(words) > 5:
            unique_words = set(words[-10:])
            if len(unique_words) <= 2:  # Same 1-2 words repeated
                return True
        return False

    def stop_generation(self):
        """Stop current generation"""
        self.interrupt_handler.interrupt()

    def is_generating(self) -> bool:
        return self._generating

    def prepare_input_payload(self, prompt: str) -> InputPayload:
        """
        Build a unified multimodal input payload from text and explicitly
        referenced files. Clipboard images are never read implicitly.
        """
        normalized_prompt = (prompt or "").strip()
        paths = self.file_handler.extract_paths(normalized_prompt)

        file_contents: List[str] = []
        file_paths: List[Path] = []
        image_attachments: List[ImageAttachment] = []
        clipboard_image_used = False

        for path_str in paths:
            path = self.file_handler.resolve_path(path_str)
            if not path:
                continue

            file_type = self.file_handler.get_file_type(path)
            if file_type == "image":
                ok, image, err = self.file_handler.load_image(path)
                if ok and image is not None:
                    image_attachments.append(
                        ImageAttachment(source=path.name, image=image, path=path)
                    )
                    file_paths.append(path)
                continue

            success, content = self.file_handler.read_file(path)
            if success:
                file_paths.append(path)
                file_contents.append(f"\n--- File: {path.name} ({file_type}) ---\n{content}\n---")

        enriched_prompt = normalized_prompt
        if not enriched_prompt and image_attachments:
            enriched_prompt = "Please analyze the attached image."

        if image_attachments:
            image_summary = ", ".join(attachment.source for attachment in image_attachments)
            enriched_prompt = (
                (enriched_prompt + "\n\n" if enriched_prompt else "")
                + f"Visual context attached: {image_summary}"
            )

        if file_contents:
            enriched_prompt = (
                (enriched_prompt + "\n\n" if enriched_prompt else "")
                + "Workspace context:\n"
                + "\n".join(file_contents)
            )

        return InputPayload(
            prompt=normalized_prompt,
            enhanced_prompt=enriched_prompt or normalized_prompt,
            file_contents=file_contents,
            file_paths=file_paths,
            image_attachments=image_attachments,
            clipboard_image_used=clipboard_image_used,
        )

    def get_auto_clipboard_prompt(self) -> Optional[str]:
        """Clipboard inspection is opt-in; never synthesize a prompt from it."""
        return None

    def get_model_info(self) -> dict:
        """Get current model info"""
        if self.current_mode not in self.MODELS:
            return {}

        info = self.MODELS[self.current_mode].copy()
        info["mode"] = self.current_mode
        info["quant"] = self.current_quant
        info["device"] = self.device
        return info

# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

_engine = None

def get_engine() -> OpenCLIEngine:
    global _engine
    if _engine is None:
        _engine = OpenCLIEngine()
    return _engine


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    'OpenCLIEngine',
    'get_engine',
    'get_interrupt_handler',
    'InterruptHandler',
    'ML_AVAILABLE',
    'BNB_AVAILABLE',
]
