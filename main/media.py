"""Pillow-backed media normalization for model-bound visual context."""

from __future__ import annotations

from pathlib import Path
from typing import Any


MAX_IMAGE_BYTES = 50 * 1024 * 1024
MAX_IMAGE_EDGE = 4096
MAX_IMAGE_PIXELS = 24_000_000


class MediaError(ValueError):
    """Image cannot be safely normalized for a vision-capable model."""


def normalize_image(image: Any, *, max_edge: int = MAX_IMAGE_EDGE) -> Any:
    """Correct orientation, bound dimensions, and return detached RGB pixels."""
    try:
        from PIL import Image, ImageOps, UnidentifiedImageError
    except ImportError as error:  # pragma: no cover - optional engine dependency
        raise MediaError("Pillow is not available") from error
    try:
        width, height = image.size
        if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
            raise MediaError("Image dimensions exceed safe model-input limits")
        normalized = ImageOps.exif_transpose(image).convert("RGB")
        normalized.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        return normalized.copy()
    except MediaError:
        raise
    except (OSError, ValueError, UnidentifiedImageError) as error:
        raise MediaError(f"Invalid image: {error}") from error


def load_model_image(path: Path) -> Any:
    """Load and normalize one existing image without adding any UI input flow."""
    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError as error:  # pragma: no cover - optional engine dependency
        raise MediaError("Pillow is not available") from error
    try:
        if path.stat().st_size > MAX_IMAGE_BYTES:
            raise MediaError("Image file exceeds 50 MB limit")
        with Image.open(path) as image:
            return normalize_image(image)
    except MediaError:
        raise
    except (OSError, UnidentifiedImageError) as error:
        raise MediaError(f"Invalid image: {error}") from error


__all__ = [
    "MAX_IMAGE_BYTES",
    "MAX_IMAGE_EDGE",
    "MAX_IMAGE_PIXELS",
    "MediaError",
    "load_model_image",
    "normalize_image",
]
