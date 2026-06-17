from __future__ import annotations

import cv2
import numpy as np

from sheet_video_to_pdf.models import BoundingBox, ExtractedRegion, RegionKind
from sheet_video_to_pdf.preprocess import denoise_grayscale, to_grayscale


LOW_CONFIDENCE_WARNING = "low confidence notation detection; using full frame fallback"


def detect_notation_region(
    frame: np.ndarray,
    *,
    region_id: str,
    stable_view_id: str,
    source_timestamp_seconds: float,
    color_order: str = "BGR",
) -> ExtractedRegion:
    gray = to_grayscale(frame, color_order=color_order)
    box, confidence = _find_notation_bounds(gray)
    warnings: list[str] = []
    fallback_used = False

    if box is None or confidence < 0.15:
        height, width = gray.shape
        box = BoundingBox(0, 0, width, height)
        confidence = 0.0
        fallback_used = True
        warnings.append(LOW_CONFIDENCE_WARNING)
        kind = RegionKind.UNKNOWN
    else:
        kind = classify_region_kind(gray, box)

    return ExtractedRegion(
        id=region_id,
        stable_view_id=stable_view_id,
        source_timestamp_seconds=source_timestamp_seconds,
        image_path=None,
        bounding_box=box,
        confidence=round(confidence, 3),
        kind=kind,
        fallback_used=fallback_used,
        warnings=warnings,
    )


def classify_region_kind(gray_image: np.ndarray, box: BoundingBox) -> RegionKind:
    height, width = gray_image.shape[:2]
    if box.width <= 0 or box.height <= 0:
        return RegionKind.UNKNOWN

    system_count = _count_system_like_groups(gray_image, box)
    region_aspect = box.width / box.height
    frame_fill_y = box.height / height
    frame_fill_x = box.width / width

    if 0.55 <= region_aspect <= 0.95 and system_count >= 4 and frame_fill_y <= 0.92:
        return RegionKind.COMPLETE_PAGE

    if system_count >= 1 and (region_aspect >= 1.35 or frame_fill_y >= 0.42 or frame_fill_x >= 0.75):
        return RegionKind.PARTIAL_VIEW

    return RegionKind.UNKNOWN


def _find_notation_bounds(gray_image: np.ndarray) -> tuple[BoundingBox | None, float]:
    gray = denoise_grayscale(gray_image)
    trimmed = _trim_video_borders(gray)
    x_offset, y_offset, inner = trimmed

    dark_mask = inner < 190
    dark_mask = _remove_tiny_components(dark_mask)
    if int(dark_mask.sum()) < 40:
        return None, 0.0

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    expanded = cv2.dilate(dark_mask.astype(np.uint8), kernel, iterations=1) > 0
    rows, cols = np.where(expanded)
    if rows.size == 0 or cols.size == 0:
        return None, 0.0

    padding_x = max(8, round(inner.shape[1] * 0.015))
    padding_y = max(8, round(inner.shape[0] * 0.025))
    x0 = max(0, int(cols.min()) - padding_x)
    x1 = min(inner.shape[1], int(cols.max()) + padding_x + 1)
    y0 = max(0, int(rows.min()) - padding_y)
    y1 = min(inner.shape[0], int(rows.max()) + padding_y + 1)

    box = BoundingBox(
        x=x_offset + x0,
        y=y_offset + y0,
        width=max(1, x1 - x0),
        height=max(1, y1 - y0),
    )
    confidence = _region_confidence(gray, box)
    return box, confidence


def _trim_video_borders(gray: np.ndarray) -> tuple[int, int, np.ndarray]:
    dark = gray < 64
    height, width = gray.shape
    row_dark = dark.mean(axis=1)
    col_dark = dark.mean(axis=0)

    top = _edge_run(row_dark, threshold=0.70, from_start=True)
    bottom = _edge_run(row_dark, threshold=0.70, from_start=False)
    left = _edge_run(col_dark, threshold=0.70, from_start=True)
    right = _edge_run(col_dark, threshold=0.70, from_start=False)

    y0 = min(top, max(0, height - 1))
    y1 = max(y0 + 1, height - bottom)
    x0 = min(left, max(0, width - 1))
    x1 = max(x0 + 1, width - right)
    return x0, y0, gray[y0:y1, x0:x1]


def _edge_run(values: np.ndarray, *, threshold: float, from_start: bool) -> int:
    iterable = values if from_start else values[::-1]
    count = 0
    for value in iterable:
        if value < threshold:
            break
        count += 1
    return count


def _remove_tiny_components(mask: np.ndarray, *, min_area: int = 12) -> np.ndarray:
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )
    cleaned = np.zeros(mask.shape, dtype=bool)
    for label in range(1, component_count):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = True
    return cleaned


def _region_confidence(gray: np.ndarray, box: BoundingBox) -> float:
    crop = gray[box.y : box.y + box.height, box.x : box.x + box.width]
    if crop.size == 0:
        return 0.0

    dark_density = float((crop < 190).mean())
    system_count = _count_system_like_groups(gray, box)
    line_score = min(1.0, system_count / 2.0)
    density_score = min(1.0, dark_density / 0.08)
    size_score = min(1.0, (box.width * box.height) / (gray.shape[0] * gray.shape[1] * 0.15))
    return max(0.0, min(1.0, 0.45 * line_score + 0.35 * density_score + 0.20 * size_score))


def _count_system_like_groups(gray: np.ndarray, box: BoundingBox) -> int:
    crop = gray[box.y : box.y + box.height, box.x : box.x + box.width]
    if crop.size == 0:
        return 0

    dark = crop < 110
    row_counts = dark.sum(axis=1)
    row_threshold = max(20, int(crop.shape[1] * 0.35))
    staff_line_rows = row_counts >= row_threshold
    line_groups = _boolean_runs(staff_line_rows, min_length=1)
    if not line_groups:
        return 0

    line_centers = [(start + end) / 2 for start, end in line_groups]
    systems = 0
    current_lines = 1
    for previous, current in zip(line_centers, line_centers[1:]):
        if current - previous <= 16:
            current_lines += 1
        else:
            if current_lines >= 3:
                systems += 1
            current_lines = 1
    if current_lines >= 3:
        systems += 1
    return systems


def _boolean_runs(mask: np.ndarray, *, min_length: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if index - start >= min_length:
                runs.append((start, index - 1))
            start = None
    if start is not None and len(mask) - start >= min_length:
        runs.append((start, len(mask) - 1))
    return runs
