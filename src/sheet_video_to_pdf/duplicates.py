from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from sheet_video_to_pdf.models import DuplicateFlags, ExtractedRegion

ImageLoader = Callable[[Path], np.ndarray]

_COMPARISON_SIZE = (128, 128)
_FOREGROUND_THRESHOLD = 220
_MIN_CONTENT_DENSITY = 0.002


@dataclass(frozen=True)
class _PreparedRegion:
    region: ExtractedRegion
    full_image: np.ndarray
    content_image: np.ndarray
    content_density: float


def flag_duplicate_regions(
    regions: Sequence[ExtractedRegion],
    *,
    image_loader: ImageLoader | None = None,
    exact_threshold: float = 0.995,
    near_threshold: float = 0.90,
    repeat_time_gap_seconds: float = 3.0,
) -> list[DuplicateFlags]:
    """Return advisory duplicate flags aligned to the input region order."""
    if image_loader is None:
        image_loader = _load_image

    flags: list[DuplicateFlags] = []
    prior: list[_PreparedRegion] = []

    for region in regions:
        prepared = _prepare_region(region, image_loader)
        if prepared is None:
            flags.append(DuplicateFlags())
            continue

        match, similarity, exact_duplicate, near_duplicate = _best_prior_match(
            prepared,
            prior,
            exact_threshold=exact_threshold,
            near_threshold=near_threshold,
        )
        repeat_candidate = False
        matched_region_id = None

        if match is not None and near_duplicate:
            matched_region_id = match.region.id
            time_gap = abs(
                prepared.region.source_timestamp_seconds
                - match.region.source_timestamp_seconds
            )
            repeat_candidate = time_gap >= repeat_time_gap_seconds

        flags.append(
            DuplicateFlags(
                exact_duplicate=exact_duplicate,
                near_duplicate=near_duplicate,
                repeat_candidate=repeat_candidate,
                matched_region_id=matched_region_id,
                similarity=similarity if match is not None else None,
            )
        )
        prior.append(prepared)

    return flags


def _best_prior_match(
    current: _PreparedRegion,
    prior: Sequence[_PreparedRegion],
    *,
    exact_threshold: float,
    near_threshold: float,
) -> tuple[_PreparedRegion | None, float | None, bool, bool]:
    best_match: _PreparedRegion | None = None
    best_similarity: float | None = None
    best_exact = False

    for candidate in prior:
        similarity, exact_similarity = _region_similarity(current, candidate)
        if best_similarity is None or similarity > best_similarity:
            best_match = candidate
            best_similarity = similarity
            best_exact = exact_similarity >= exact_threshold

    if best_match is None or best_similarity is None:
        return None, None, False, False

    near_duplicate = best_similarity >= near_threshold or best_exact
    return best_match, best_similarity, best_exact, near_duplicate


def _region_similarity(
    first: _PreparedRegion,
    second: _PreparedRegion,
) -> tuple[float, float]:
    if _has_content_mismatch(first, second):
        return 0.0, 0.0

    full_similarity = _normalized_pixel_similarity(first.full_image, second.full_image)
    content_similarity = _normalized_pixel_similarity(
        first.content_image,
        second.content_image,
    )

    blended_content_similarity = (full_similarity + content_similarity) / 2.0
    return max(full_similarity, blended_content_similarity), full_similarity


def _has_content_mismatch(first: _PreparedRegion, second: _PreparedRegion) -> bool:
    first_has_content = first.content_density >= _MIN_CONTENT_DENSITY
    second_has_content = second.content_density >= _MIN_CONTENT_DENSITY
    return first_has_content != second_has_content


def _prepare_region(
    region: ExtractedRegion,
    image_loader: ImageLoader,
) -> _PreparedRegion | None:
    if region.image_path is None:
        return None

    image = image_loader(region.image_path)
    grayscale = _to_grayscale(image)
    content = _crop_content(grayscale)
    density = float(np.mean(grayscale < _FOREGROUND_THRESHOLD))

    return _PreparedRegion(
        region=region,
        full_image=_resize(grayscale, _COMPARISON_SIZE),
        content_image=_resize(content, _COMPARISON_SIZE),
        content_density=density,
    )


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        grayscale = array
    elif array.ndim == 3 and array.shape[2] >= 3:
        channels = array[:, :, :3].astype(np.float32)
        grayscale = (
            0.299 * channels[:, :, 0]
            + 0.587 * channels[:, :, 1]
            + 0.114 * channels[:, :, 2]
        )
    else:
        raise ValueError("Region images must be 2D grayscale or 3-channel color arrays.")

    return np.clip(grayscale, 0, 255).astype(np.uint8)


def _crop_content(image: np.ndarray) -> np.ndarray:
    mask = image < _FOREGROUND_THRESHOLD
    if not np.any(mask):
        return image

    y_indices, x_indices = np.where(mask)
    height, width = image.shape
    pad_y = max(4, int(height * 0.03))
    pad_x = max(4, int(width * 0.03))
    y0 = max(0, int(y_indices.min()) - pad_y)
    y1 = min(height, int(y_indices.max()) + pad_y + 1)
    x0 = max(0, int(x_indices.min()) - pad_x)
    x1 = min(width, int(x_indices.max()) + pad_x + 1)

    return image[y0:y1, x0:x1]


def _resize(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    target_height, target_width = size
    source_height, source_width = image.shape
    if source_height == target_height and source_width == target_width:
        return image.astype(np.float32) / 255.0

    y_positions = np.linspace(0, source_height - 1, target_height)
    x_positions = np.linspace(0, source_width - 1, target_width)
    y0 = np.floor(y_positions).astype(int)
    x0 = np.floor(x_positions).astype(int)
    y1 = np.minimum(y0 + 1, source_height - 1)
    x1 = np.minimum(x0 + 1, source_width - 1)
    y_weight = (y_positions - y0)[:, None]
    x_weight = (x_positions - x0)[None, :]

    top = (1.0 - x_weight) * image[y0[:, None], x0[None, :]] + x_weight * image[
        y0[:, None],
        x1[None, :],
    ]
    bottom = (1.0 - x_weight) * image[y1[:, None], x0[None, :]] + x_weight * image[
        y1[:, None],
        x1[None, :],
    ]
    resized = (1.0 - y_weight) * top + y_weight * bottom

    return resized.astype(np.float32) / 255.0


def _normalized_pixel_similarity(first: np.ndarray, second: np.ndarray) -> float:
    difference = np.mean(np.abs(first - second))
    return float(max(0.0, 1.0 - difference))


def _load_image(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("L"))


__all__ = ["flag_duplicate_regions"]
