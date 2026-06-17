from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import cv2
import numpy as np

from .models import ExtractedRegion, RegionKind, StitchPlacement


@dataclass(frozen=True)
class StitchedStrip:
    id: str
    image: np.ndarray
    included_region_ids: list[str]
    source_start_seconds: float
    source_end_seconds: float
    placements: list[StitchPlacement] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class StitchResult:
    strips: list[StitchedStrip]
    warnings: list[str] = field(default_factory=list)


def stitch_regions(
    regions: Sequence[ExtractedRegion],
    images_by_region_id: Mapping[str, np.ndarray],
    *,
    min_alignment_confidence: float = 0.74,
) -> StitchResult:
    strips: list[StitchedStrip] = []
    warnings: list[str] = []

    for region in sorted(regions, key=lambda item: item.source_timestamp_seconds):
        image = _as_gray(images_by_region_id[region.id])
        if region.kind is RegionKind.COMPLETE_PAGE:
            strips.append(_new_strip(region, image, len(strips) + 1))
            continue

        if not strips or strips[-1].included_region_ids and _last_region_was_complete(regions, strips[-1]):
            strips.append(_new_strip(region, image, len(strips) + 1))
            continue

        current = strips[-1]
        if image.shape[1] > current.image.shape[1]:
            resized_current = _resize_to_width(current.image, image.shape[1])
            current = StitchedStrip(
                id=current.id,
                image=resized_current,
                included_region_ids=current.included_region_ids,
                source_start_seconds=current.source_start_seconds,
                source_end_seconds=current.source_end_seconds,
                placements=current.placements,
                warnings=current.warnings,
            )
            strips[-1] = current
        normalized = _resize_to_width(image, current.image.shape[1])
        overlap, confidence = _best_vertical_overlap(current.image, normalized)
        if overlap > 0 and confidence >= min_alignment_confidence:
            merged = np.vstack([current.image, normalized[overlap:, :]])
            placement = StitchPlacement(current.id, current.image.shape[0] - overlap, overlap, confidence, "merged-overlap")
            strips[-1] = _replace_strip(current, merged, region, placement)
        elif _has_staff_like_content(current.image) and _has_staff_like_content(normalized) and confidence >= 0.60:
            placement = StitchPlacement(current.id, current.image.shape[0], 0, confidence, "appended-low-overlap")
            merged = np.vstack([current.image, normalized])
            strips[-1] = _replace_strip(current, merged, region, placement)
        else:
            warning = f"low alignment confidence for {region.id}; starting new strip"
            warnings.append(warning)
            strips.append(_new_strip(region, image, len(strips) + 1, warnings=[warning]))

    return StitchResult(strips=strips, warnings=warnings)


def _new_strip(
    region: ExtractedRegion,
    image: np.ndarray,
    index: int,
    warnings: list[str] | None = None,
) -> StitchedStrip:
    placement = StitchPlacement(f"strip_{index:03d}", 0, 0, 1.0, "started-strip")
    return StitchedStrip(
        id=f"strip_{index:03d}",
        image=image,
        included_region_ids=[region.id],
        source_start_seconds=region.source_timestamp_seconds,
        source_end_seconds=region.source_timestamp_seconds,
        placements=[placement],
        warnings=warnings or [],
    )


def _replace_strip(
    strip: StitchedStrip,
    image: np.ndarray,
    region: ExtractedRegion,
    placement: StitchPlacement,
) -> StitchedStrip:
    return StitchedStrip(
        id=strip.id,
        image=image,
        included_region_ids=[*strip.included_region_ids, region.id],
        source_start_seconds=strip.source_start_seconds,
        source_end_seconds=region.source_timestamp_seconds,
        placements=[*strip.placements, placement],
        warnings=strip.warnings,
    )


def _last_region_was_complete(regions: Sequence[ExtractedRegion], strip: StitchedStrip) -> bool:
    complete_ids = {region.id for region in regions if region.kind is RegionKind.COMPLETE_PAGE}
    return strip.included_region_ids[-1] in complete_ids


def _best_vertical_overlap(base: np.ndarray, incoming: np.ndarray) -> tuple[int, float]:
    comparison_width = 240
    vertical_scale = 1.0
    if max(base.shape[1], incoming.shape[1]) > comparison_width:
        base_cmp = _resize_to_width(base, comparison_width)
        incoming_cmp = _resize_to_width(incoming, comparison_width)
        vertical_scale = incoming_cmp.shape[0] / incoming.shape[0]
    else:
        base_cmp = base
        incoming_cmp = incoming

    max_overlap = min(base_cmp.shape[0], incoming_cmp.shape[0], max(12, int(incoming_cmp.shape[0] * 0.8)))
    min_overlap = min(24, max_overlap)
    step = 4
    best_overlap = 0
    best_confidence = 0.0
    for overlap in range(min_overlap, max_overlap + 1, step):
        tail = base_cmp[-overlap:, :]
        head = incoming_cmp[:overlap, :]
        confidence = _similarity(tail, head)
        if confidence > best_confidence or (
            confidence >= best_confidence - 0.10 and overlap > best_overlap
        ):
            best_confidence = confidence
            best_overlap = overlap
    return int(round((best_overlap / vertical_scale) * 0.95)), best_confidence


def _similarity(first: np.ndarray, second: np.ndarray) -> float:
    first_dark = (first < 200).astype(np.float32)
    second_dark = (second < 200).astype(np.float32)
    union = np.logical_or(first_dark > 0, second_dark > 0).sum()
    if union == 0:
        return 0.0
    intersection = np.logical_and(first_dark > 0, second_dark > 0).sum()
    mask_iou = float(intersection / union)

    first_projection = first_dark.mean(axis=1)
    second_projection = second_dark.mean(axis=1)
    projection_similarity = 1.0 - float(np.mean(np.abs(first_projection - second_projection)))
    staff_similarity = _staff_pattern_similarity(first, second)
    structural_similarity = 0.65 * mask_iou + 0.35 * projection_similarity
    return max(0.0, min(1.0, max(structural_similarity, staff_similarity)))


def _staff_pattern_similarity(first: np.ndarray, second: np.ndarray) -> float:
    first_centers = _staff_line_centers(first)
    second_centers = _staff_line_centers(second)
    if len(first_centers) < 3 or len(second_centers) < 3:
        return 0.0
    if abs(len(first_centers) - len(second_centers)) > 1:
        return 0.0

    count = min(len(first_centers), len(second_centers))
    first_selected = np.array(first_centers[-count:], dtype=np.float32)
    second_selected = np.array(second_centers[:count], dtype=np.float32)
    offset = float(np.mean(first_selected - second_selected))
    aligned = second_selected + offset
    distance = float(np.mean(np.abs(first_selected - aligned)))
    return max(0.0, 1.0 - distance / 6.0)


def _staff_line_centers(image: np.ndarray) -> list[float]:
    dark = image < 80
    row_counts = dark.sum(axis=1)
    threshold = max(16, int(image.shape[1] * 0.35))
    rows = row_counts >= threshold
    centers: list[float] = []
    start: int | None = None
    for index, value in enumerate(rows):
        if value and start is None:
            start = index
        elif not value and start is not None:
            centers.append((start + index - 1) / 2)
            start = None
    if start is not None:
        centers.append((start + len(rows) - 1) / 2)
    return centers


def _resize_to_width(image: np.ndarray, width: int) -> np.ndarray:
    if image.shape[1] == width:
        return image.copy()
    scale = width / image.shape[1]
    height = max(1, int(round(image.shape[0] * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(image, (width, height), interpolation=interpolation)


def _as_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.uint8)
    return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY)


def _has_staff_like_content(image: np.ndarray) -> bool:
    return bool(np.mean(image < 230) > 0.01)


__all__ = ["StitchResult", "StitchedStrip", "stitch_regions"]
