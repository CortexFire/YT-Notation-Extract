from __future__ import annotations

import cv2
import numpy as np


def to_grayscale(frame: np.ndarray, *, color_order: str = "BGR") -> np.ndarray:
    """Return an 8-bit grayscale image from a grayscale, BGR, or RGB frame."""
    if frame.ndim == 2:
        return _as_uint8(frame)

    if frame.ndim != 3 or frame.shape[2] not in (3, 4):
        raise ValueError("Expected a grayscale, RGB, or BGR image array")

    order = color_order.upper()
    if order == "BGR":
        code = cv2.COLOR_BGRA2GRAY if frame.shape[2] == 4 else cv2.COLOR_BGR2GRAY
    elif order == "RGB":
        code = cv2.COLOR_RGBA2GRAY if frame.shape[2] == 4 else cv2.COLOR_RGB2GRAY
    else:
        raise ValueError("color_order must be 'BGR' or 'RGB'")

    return cv2.cvtColor(_as_uint8(frame), code)


def resize_for_comparison(image: np.ndarray, *, max_dimension: int = 640) -> np.ndarray:
    """Resize an image to a stable maximum dimension while preserving aspect ratio."""
    if max_dimension <= 0:
        raise ValueError("max_dimension must be positive")

    height, width = image.shape[:2]
    current_max = max(height, width)
    if current_max == 0:
        raise ValueError("image dimensions must be non-zero")
    if current_max == max_dimension:
        return image.copy()

    scale = max_dimension / current_max
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(image, new_size, interpolation=interpolation)


def denoise_grayscale(image: np.ndarray, *, kernel_size: int = 3) -> np.ndarray:
    """Apply light denoising suitable for compressed score-video frames."""
    gray = to_grayscale(image) if image.ndim == 3 else _as_uint8(image)
    if kernel_size <= 1:
        return gray.copy()
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")
    return cv2.medianBlur(gray, kernel_size)


def prepare_for_comparison(
    frame: np.ndarray,
    *,
    color_order: str = "BGR",
    max_dimension: int = 640,
) -> np.ndarray:
    gray = to_grayscale(frame, color_order=color_order)
    resized = resize_for_comparison(gray, max_dimension=max_dimension)
    return denoise_grayscale(resized)


def _as_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    return np.clip(image, 0, 255).astype(np.uint8)
