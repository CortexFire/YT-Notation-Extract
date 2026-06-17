from __future__ import annotations

from pathlib import Path

import numpy as np

from sheet_video_to_pdf.config import DEFAULT_CONFIG
from sheet_video_to_pdf.models import (
    BoundingBox,
    ExtractedRegion,
    PageOrientation,
    RegionKind,
    StitchPlacement,
)
from sheet_video_to_pdf.pagination import _score_system_ranges, _system_ranges, paginate_strips
from sheet_video_to_pdf.stitching import StitchedStrip, stitch_regions


def _region(
    region_id: str,
    *,
    timestamp: float,
    image: np.ndarray,
    kind: RegionKind = RegionKind.PARTIAL_VIEW,
) -> ExtractedRegion:
    return ExtractedRegion(
        id=region_id,
        stable_view_id=f"view_{region_id[-3:]}",
        source_timestamp_seconds=timestamp,
        image_path=Path(f"{region_id}.jpg"),
        bounding_box=BoundingBox(0, 0, image.shape[1], image.shape[0]),
        confidence=0.9,
        kind=kind,
    )


def _viewport(start_system: int, *, systems: int = 2, width: int = 360) -> np.ndarray:
    image = np.full((systems * 70, width), 255, dtype=np.uint8)
    for local_system in range(systems):
        y0 = local_system * 70 + 14
        label_x = 24 + (start_system + local_system) * 7
        for line in range(5):
            image[y0 + line * 6 : y0 + line * 6 + 2, 44 : width - 34] = 12
        image[y0 + 8 : y0 + 18, label_x : label_x + 8] = 18
        image[y0 + 20 : y0 + 28, label_x + 48 : label_x + 58] = 18
    return image


def _grand_staff_viewport(start_system: int, *, systems: int = 4, width: int = 520) -> np.ndarray:
    image = np.full((systems * 150, width), 255, dtype=np.uint8)
    for local_system in range(systems):
        for staff_offset in (20, 80):
            y0 = local_system * 150 + staff_offset
            for line in range(5):
                image[y0 + line * 6 : y0 + line * 6 + 2, 44 : width - 34] = 12
        label_x = 24 + (start_system + local_system) * 7
        image[local_system * 150 + 34 : local_system * 150 + 44, label_x : label_x + 8] = 18
        image[local_system * 150 + 96 : local_system * 150 + 106, label_x + 48 : label_x + 58] = 18
    return image


def test_stitch_regions_normalizes_widths_and_merges_confident_vertical_overlap() -> None:
    first = _viewport(0, width=320)
    second = _viewport(1, width=400)
    regions = [
        _region("region_001", timestamp=0.0, image=first),
        _region("region_002", timestamp=2.0, image=second),
    ]

    result = stitch_regions(regions, {regions[0].id: first, regions[1].id: second})

    assert len(result.strips) == 1
    strip = result.strips[0]
    assert strip.image.shape[1] == 400
    assert 230 <= strip.image.shape[0] <= 260
    assert strip.included_region_ids == ["region_001", "region_002"]
    assert strip.placements[1].overlap_pixels > 0
    assert strip.placements[1].decision == "merged-overlap"
    assert strip.placements[1].alignment_confidence >= 0.74


def test_stitch_region_width_normalization_preserves_aspect_ratio() -> None:
    first = np.full((90, 300), 255, dtype=np.uint8)
    first[20:30, 40:260] = 0
    second = np.full((90, 400), 255, dtype=np.uint8)
    second[52:72, 90:310] = 0
    regions = [
        _region("region_001", timestamp=0.0, image=first),
        _region("region_002", timestamp=2.0, image=second),
    ]

    result = stitch_regions(regions, {regions[0].id: first, regions[1].id: second})

    assert len(result.strips) == 2
    first_strip = result.strips[0]
    assert first_strip.image.shape[1] == 400
    assert 118 <= first_strip.image.shape[0] <= 122


def test_stitch_regions_warns_and_starts_new_strip_for_low_confidence_alignment() -> None:
    first = _viewport(0)
    unrelated = np.full(first.shape, 255, dtype=np.uint8)
    unrelated[22:42, 70:290] = 20
    regions = [
        _region("region_001", timestamp=0.0, image=first),
        _region("region_002", timestamp=2.0, image=unrelated),
    ]

    result = stitch_regions(regions, {regions[0].id: first, regions[1].id: unrelated})

    assert len(result.strips) == 2
    assert result.strips[0].included_region_ids == ["region_001"]
    assert result.strips[1].included_region_ids == ["region_002"]
    assert any("low alignment" in warning for warning in result.warnings)


def test_complete_page_regions_pass_through_without_forced_stitching() -> None:
    first_page = _viewport(0, systems=5, width=300)
    second_page = _viewport(5, systems=5, width=300)
    regions = [
        _region("region_001", timestamp=0.0, image=first_page, kind=RegionKind.COMPLETE_PAGE),
        _region("region_002", timestamp=4.0, image=second_page, kind=RegionKind.COMPLETE_PAGE),
    ]

    result = stitch_regions(regions, {regions[0].id: first_page, regions[1].id: second_page})

    assert len(result.strips) == 2
    assert [strip.included_region_ids for strip in result.strips] == [["region_001"], ["region_002"]]
    assert all(strip.warnings == [] for strip in result.strips)


def test_paginate_strips_prefers_whitespace_gaps_and_applies_letter_margins() -> None:
    strips = stitch_regions(
        [_region("region_001", timestamp=0.0, image=_viewport(0, systems=6))],
        {"region_001": _viewport(0, systems=6)},
    ).strips
    config = DEFAULT_CONFIG

    pages = paginate_strips(strips, config)

    assert len(pages) == 1
    target_width = int(round(8.5 * config.pdf_dpi))
    target_height = int(round(11.0 * config.pdf_dpi))
    margin = int(round(config.page_margin_inches * config.pdf_dpi))
    assert all(page.image.shape[:2] == (target_height, target_width) for page in pages)
    assert all(page.source_strip_id == "strip_001" for page in pages)
    assert pages[0].break_confidence >= 0.5
    assert np.all(pages[0].image[:margin, :] == 255)
    assert np.all(pages[0].image[:, :margin] == 255)
    assert pages[0].included_region_ids == ["region_001"]


def test_paginate_strips_packs_short_score_strips_into_portrait_pages() -> None:
    strips = [
        StitchedStrip(
            id=f"strip_{index:03d}",
            image=_viewport(index * 2, systems=2, width=520),
            included_region_ids=[f"region_{index:03d}"],
            source_start_seconds=float(index),
            source_end_seconds=float(index),
            placements=[StitchPlacement(f"strip_{index:03d}", 0, 0, 1.0, "started-strip")],
        )
        for index in range(1, 6)
    ]
    config = DEFAULT_CONFIG.__class__(
        page_orientation=PageOrientation.PORTRAIT,
        page_margin_inches=0.25,
        pdf_dpi=100,
        target_systems_per_page="auto",
    )

    pages = paginate_strips(strips, config)

    assert len(pages) == 2
    assert all(page.image.shape[:2] == (1100, 850) for page in pages)
    assert pages[0].included_region_ids == ["region_001", "region_002", "region_003"]
    assert pages[1].included_region_ids == ["region_004", "region_005"]

    margin = int(round(config.page_margin_inches * config.pdf_dpi))
    for page in pages:
        rows, cols = np.where(page.image < 80)
        assert rows.min() >= margin
        assert rows.max() < page.image.shape[0] - margin
        assert cols.min() >= margin
        assert cols.max() < page.image.shape[1] - margin


def test_paginate_strips_balances_long_strip_chunks_to_avoid_sparse_portrait_pages() -> None:
    strip = StitchedStrip(
        id="strip_001",
        image=_viewport(0, systems=8, width=520),
        included_region_ids=["region_001"],
        source_start_seconds=0.0,
        source_end_seconds=0.0,
        placements=[StitchPlacement("strip_001", 0, 0, 1.0, "started-strip")],
    )
    config = DEFAULT_CONFIG.__class__(
        page_orientation=PageOrientation.PORTRAIT,
        page_margin_inches=0.25,
        pdf_dpi=100,
        target_systems_per_page="auto",
    )

    pages = paginate_strips([strip], config)

    assert len(pages) == 2
    assert [_system_ranges(page.image).__len__() for page in pages] == [4, 4]


def test_paginate_strips_counts_paired_staves_as_double_staff_systems() -> None:
    strip = StitchedStrip(
        id="strip_001",
        image=_grand_staff_viewport(0, systems=8),
        included_region_ids=["region_001"],
        source_start_seconds=0.0,
        source_end_seconds=0.0,
        placements=[StitchPlacement("strip_001", 0, 0, 1.0, "started-strip")],
    )
    config = DEFAULT_CONFIG.__class__(
        page_orientation=PageOrientation.PORTRAIT,
        page_margin_inches=0.25,
        pdf_dpi=100,
        target_systems_per_page="auto",
    )

    pages = paginate_strips([strip], config)

    assert len(pages) == 2
    assert [len(_score_system_ranges(page.image)) for page in pages] == [4, 4]


def test_paginate_strips_preserves_white_background_inside_rendered_notation_area() -> None:
    image = np.full((100, 200), 255, dtype=np.uint8)
    image[10:12, 20:180] = 0
    strip = StitchedStrip(
        id="strip_001",
        image=image,
        included_region_ids=["region_001"],
        source_start_seconds=0.0,
        source_end_seconds=0.0,
        placements=[StitchPlacement("strip_001", 0, 0, 1.0, "started-strip")],
    )
    config = DEFAULT_CONFIG.__class__(page_margin_inches=0.25, pdf_dpi=100)

    pages = paginate_strips([strip], config)

    page = pages[0].image
    page_height, page_width = page.shape
    margin = int(round(config.page_margin_inches * config.pdf_dpi))
    content_width = page_width - 2 * margin
    content_height = page_height - 2 * margin
    scale = min(content_width / image.shape[1], content_height / image.shape[0])
    rendered_width = int(round(image.shape[1] * scale))
    rendered_height = int(round(image.shape[0] * scale))
    x0 = margin + (content_width - rendered_width) // 2
    y0 = margin

    assert page[y0 + rendered_height - 8, x0 + rendered_width // 2] == 255


def test_paginate_strips_uses_landscape_dimensions_without_disproportionate_stretching() -> None:
    strip = stitch_regions(
        [_region("region_001", timestamp=0.0, image=_viewport(0, systems=2, width=520))],
        {"region_001": _viewport(0, systems=2, width=520)},
    ).strips[0]
    config = DEFAULT_CONFIG.__class__(
        page_orientation=PageOrientation.LANDSCAPE,
        page_margin_inches=0.25,
        pdf_dpi=100,
    )

    pages = paginate_strips([strip], config)

    assert len(pages) == 1
    assert pages[0].image.shape[:2] == (850, 1100)
    original_rows, original_cols = np.where(strip.image < 80)
    content_rows, content_cols = np.where(pages[0].image < 80)
    rendered_height = content_rows.max() - content_rows.min() + 1
    rendered_width = content_cols.max() - content_cols.min() + 1
    original_height = original_rows.max() - original_rows.min() + 1
    original_width = original_cols.max() - original_cols.min() + 1
    assert abs((rendered_width / rendered_height) - (original_width / original_height)) < 0.08
