from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from typing import Sequence

import cv2
import numpy as np

from .models import AppConfig, PageOrientation
from .stitching import StitchedStrip


@dataclass(frozen=True)
class PaginatedPage:
    id: str
    image: np.ndarray
    source_strip_id: str
    included_region_ids: list[str]
    break_confidence: float
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _PageSegment:
    image: np.ndarray
    source_strip_id: str
    included_region_ids: list[str]
    system_count: int
    break_confidence: float


def paginate_strips(strips: Sequence[StitchedStrip], config: AppConfig) -> list[PaginatedPage]:
    pages: list[PaginatedPage] = []
    pending: list[_PageSegment] = []
    for strip in strips:
        for segment in _segments_for_strip(strip, config):
            if pending and _packed_system_count(pending) + segment.system_count > _target_systems_per_page(config):
                pages.append(_page_from_segments(len(pages) + 1, pending, config))
                pending = []
            pending.append(segment)
            if _packed_system_count(pending) >= _target_systems_per_page(config):
                pages.append(_page_from_segments(len(pages) + 1, pending, config))
                pending = []
    if pending:
        pages.append(_page_from_segments(len(pages) + 1, pending, config))
    return pages


def _segments_for_strip(strip: StitchedStrip, config: AppConfig) -> list[_PageSegment]:
    gray = strip.image if strip.image.ndim == 2 else cv2.cvtColor(strip.image, cv2.COLOR_BGR2GRAY)
    system_ranges = _score_system_ranges(gray)
    systems_per_page = _target_systems_per_page(config)

    if len(system_ranges) <= systems_per_page:
        return [
            _PageSegment(
                image=gray,
                source_strip_id=strip.id,
                included_region_ids=list(strip.included_region_ids),
                system_count=max(1, len(system_ranges)),
                break_confidence=1.0,
            )
        ]

    segments: list[_PageSegment] = []
    start = 0
    for chunk_size in _system_chunk_sizes(len(system_ranges), systems_per_page):
        selected = system_ranges[start : start + chunk_size]
        y0 = max(0, selected[0][0] - 12)
        y1 = min(gray.shape[0], selected[-1][1] + 12)
        segments.append(
            _PageSegment(
                image=gray[y0:y1, :],
                source_strip_id=strip.id,
                included_region_ids=list(strip.included_region_ids),
                system_count=len(selected),
                break_confidence=0.8,
            )
        )
        start += chunk_size
    return segments


def _system_chunk_sizes(system_count: int, target_per_page: int) -> list[int]:
    if system_count <= 0:
        return []
    if system_count <= target_per_page:
        return [system_count]

    page_count = max(1, ceil(system_count / target_per_page))

    base_size = system_count // page_count
    extra = system_count % page_count
    return [base_size + (1 if index < extra else 0) for index in range(page_count)]


def _target_systems_per_page(config: AppConfig) -> int:
    if config.target_systems_per_page == "auto":
        return 6
    return max(1, int(config.target_systems_per_page))


def _packed_system_count(segments: Sequence[_PageSegment]) -> int:
    return sum(segment.system_count for segment in segments)


def _page_from_segments(
    page_index: int,
    segments: Sequence[_PageSegment],
    config: AppConfig,
) -> PaginatedPage:
    return PaginatedPage(
        id=f"page_{page_index:03d}",
        image=_render_page_group([segment.image for segment in segments], config),
        source_strip_id=",".join(segment.source_strip_id for segment in segments),
        included_region_ids=[
            region_id
            for segment in segments
            for region_id in segment.included_region_ids
        ],
        break_confidence=min(segment.break_confidence for segment in segments),
    )


def _system_ranges(gray: np.ndarray) -> list[tuple[int, int]]:
    dark = gray < 80
    row_counts = dark.sum(axis=1)
    row_threshold = max(20, int(gray.shape[1] * 0.35))
    line_rows = row_counts >= row_threshold
    line_runs = _runs(line_rows)
    if not line_runs:
        return [(0, gray.shape[0])]

    centers = [(start + end) // 2 for start, end in line_runs]
    systems: list[tuple[int, int]] = []
    current = [centers[0]]
    for center in centers[1:]:
        if center - current[-1] <= 18:
            current.append(center)
        else:
            if len(current) >= 3:
                systems.append((max(0, current[0] - 12), min(gray.shape[0], current[-1] + 12)))
            current = [center]
    if len(current) >= 3:
        systems.append((max(0, current[0] - 12), min(gray.shape[0], current[-1] + 12)))
    return systems or [(0, gray.shape[0])]


def _score_system_ranges(gray: np.ndarray) -> list[tuple[int, int]]:
    staff_ranges = _system_ranges(gray)
    if len(staff_ranges) < 4:
        return staff_ranges

    gaps = [staff_ranges[index + 1][0] - staff_ranges[index][1] for index in range(len(staff_ranges) - 1)]
    min_gap = min(gaps)
    max_gap = max(gaps)
    if min_gap <= 0 or max_gap < min_gap * 1.6:
        return staff_ranges

    pair_gap_threshold = (min_gap + max_gap) / 2
    systems: list[tuple[int, int]] = []
    index = 0
    while index < len(staff_ranges):
        if index + 1 < len(staff_ranges) and gaps[index] <= pair_gap_threshold:
            systems.append((staff_ranges[index][0], staff_ranges[index + 1][1]))
            index += 2
        else:
            systems.append(staff_ranges[index])
            index += 1
    return systems


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _render_page(segment: np.ndarray, config: AppConfig) -> np.ndarray:
    return _render_page_group([segment], config)


def _render_page_group(segments: Sequence[np.ndarray], config: AppConfig) -> np.ndarray:
    width_inches, height_inches = (8.5, 11.0)
    if config.page_orientation is PageOrientation.LANDSCAPE:
        width_inches, height_inches = height_inches, width_inches

    page_width = int(round(width_inches * config.pdf_dpi))
    page_height = int(round(height_inches * config.pdf_dpi))
    margin = int(round(config.page_margin_inches * config.pdf_dpi))
    content_width = max(1, page_width - 2 * margin)
    content_height = max(1, page_height - 2 * margin)
    gap = max(8, int(round(config.pdf_dpi * 0.12))) if len(segments) > 1 else 0

    max_segment_width = max(segment.shape[1] for segment in segments)
    total_segment_height = sum(segment.shape[0] for segment in segments) + gap * (len(segments) - 1)
    scale = min(content_width / max_segment_width, content_height / total_segment_height)

    page = np.full((page_height, page_width), 255, dtype=np.uint8)
    y0 = margin
    for segment in segments:
        rendered_width = max(1, int(round(segment.shape[1] * scale)))
        rendered_height = max(1, int(round(segment.shape[0] * scale)))
        resized = cv2.resize(
            segment,
            (rendered_width, rendered_height),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
        )
        x0 = margin + max(0, (content_width - rendered_width) // 2)
        page[y0 : y0 + rendered_height, x0 : x0 + rendered_width] = resized
        page[y0, x0 : x0 + rendered_width] = 249
        page[y0 + rendered_height - 1, x0 : x0 + rendered_width] = 249
        page[y0 : y0 + rendered_height, x0] = 249
        page[y0 : y0 + rendered_height, x0 + rendered_width - 1] = 249
        y0 += rendered_height + gap
    return page


__all__ = ["PaginatedPage", "paginate_strips"]
