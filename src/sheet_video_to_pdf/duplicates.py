from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from sheet_video_to_pdf.models import DuplicateFlags, DuplicatePolicy, ExtractedRegion

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
    images_by_region_id: Mapping[str, np.ndarray] | None = None,
    image_loader: ImageLoader | None = None,
    exact_threshold: float = 0.995,
    near_threshold: float = 0.90,
    repeat_time_gap_seconds: float = 3.0,
) -> list[DuplicateFlags]:
    """Return advisory duplicate flags aligned to the input region order."""
    if image_loader is None:
        image_loader = _load_image

    prepared_regions = [
        _prepare_region(region, images_by_region_id, image_loader)
        for region in regions
    ]
    valid_prepared = [
        prepared
        for prepared in prepared_regions
        if prepared is not None
    ]
    valid_region_indexes = [
        index
        for index, prepared in enumerate(prepared_regions)
        if prepared is not None
    ]
    valid_positions_by_region_index = {
        region_index: valid_position
        for valid_position, region_index in enumerate(valid_region_indexes)
    }
    full_stack, content_stack, density_array = _feature_stacks(valid_prepared)

    flags: list[DuplicateFlags] = []

    for region_index, region in enumerate(regions):
        prepared = prepared_regions[region_index]
        if prepared is None:
            flags.append(DuplicateFlags())
            continue

        match, similarity, exact_duplicate, near_duplicate = _best_prior_match_vectorized(
            valid_positions_by_region_index[region_index],
            valid_prepared,
            full_stack,
            content_stack,
            density_array,
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

    return flags


def apply_duplicate_policy(
    regions: Sequence[ExtractedRegion],
    policy: DuplicatePolicy,
) -> list[ExtractedRegion]:
    if policy in {DuplicatePolicy.FLAG, DuplicatePolicy.FLAG_AND_INCLUDE}:
        return list(regions)

    if policy is DuplicatePolicy.FLAG_AND_SUPPRESS_OVERLAP:
        return [
            region
            for region in regions
            if not (
                (region.duplicate_flags.exact_duplicate or region.duplicate_flags.near_duplicate)
                and not region.duplicate_flags.repeat_candidate
            )
        ]

    return list(regions)


def _best_prior_match_vectorized(
    current_position: int,
    prepared_regions: Sequence[_PreparedRegion],
    full_stack: np.ndarray,
    content_stack: np.ndarray,
    content_density: np.ndarray,
    *,
    exact_threshold: float,
    near_threshold: float,
) -> tuple[_PreparedRegion | None, float | None, bool, bool]:
    if current_position <= 0:
        return None, None, False, False

    current_full = full_stack[current_position]
    current_content = content_stack[current_position]
    prior_full = full_stack[:current_position]
    prior_content = content_stack[:current_position]

    full_similarity = 1.0 - np.mean(np.abs(prior_full - current_full), axis=1)
    content_similarity = 1.0 - np.mean(np.abs(prior_content - current_content), axis=1)
    blended_content_similarity = (full_similarity + content_similarity) / 2.0
    similarities = np.maximum(full_similarity, blended_content_similarity)

    current_has_content = content_density[current_position] >= _MIN_CONTENT_DENSITY
    prior_has_content = content_density[:current_position] >= _MIN_CONTENT_DENSITY
    similarities = np.where(prior_has_content == current_has_content, similarities, 0.0)
    similarities = np.clip(similarities, 0.0, 1.0)
    full_similarity = np.clip(full_similarity, 0.0, 1.0)

    best_position = int(np.argmax(similarities))
    best_similarity = float(similarities[best_position])
    best_exact = bool(full_similarity[best_position] >= exact_threshold)
    near_duplicate = best_similarity >= near_threshold or best_exact
    return prepared_regions[best_position], best_similarity, best_exact, near_duplicate


def _prepare_region(
    region: ExtractedRegion,
    images_by_region_id: Mapping[str, np.ndarray] | None,
    image_loader: ImageLoader,
) -> _PreparedRegion | None:
    if images_by_region_id is not None and region.id in images_by_region_id:
        image = images_by_region_id[region.id]
    elif region.image_path is not None:
        image = image_loader(region.image_path)
    else:
        return None

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


def _feature_stacks(
    prepared_regions: Sequence[_PreparedRegion],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not prepared_regions:
        empty = np.empty((0, 0), dtype=np.float32)
        return empty, empty, np.empty((0,), dtype=np.float32)

    full_stack = np.stack(
        [prepared.full_image.reshape(-1) for prepared in prepared_regions],
    ).astype(np.float32, copy=False)
    content_stack = np.stack(
        [prepared.content_image.reshape(-1) for prepared in prepared_regions],
    ).astype(np.float32, copy=False)
    density_array = np.array(
        [prepared.content_density for prepared in prepared_regions],
        dtype=np.float32,
    )
    return full_stack, content_stack, density_array


def _load_image(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("L"))


__all__ = ["apply_duplicate_policy", "flag_duplicate_regions"]
