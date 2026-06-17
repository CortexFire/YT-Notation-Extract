from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from sheet_video_to_pdf.duplicates import apply_duplicate_policy, flag_duplicate_regions
from sheet_video_to_pdf.models import BoundingBox, DuplicateFlags, DuplicatePolicy, ExtractedRegion, RegionKind


def _region(region_id: str, image_path: str, timestamp: float) -> ExtractedRegion:
    return ExtractedRegion(
        id=region_id,
        stable_view_id=f"view_{region_id[-3:]}",
        source_timestamp_seconds=timestamp,
        image_path=Path(image_path),
        bounding_box=BoundingBox(0, 0, 120, 80),
        confidence=0.95,
        kind=RegionKind.PARTIAL_VIEW,
    )


def _notation_image(*, shifted: bool = False, width: int = 120, height: int = 80) -> np.ndarray:
    image = np.full((height, width, 3), 245, dtype=np.uint8)
    y_shift = 2 if shifted else 0
    for y in (18, 25, 32, 50, 57):
        image[y + y_shift : y + y_shift + 2, 12:108] = 18
    for x, y in ((35, 24), (62, 31), (87, 55)):
        image[y + y_shift - 3 : y + y_shift + 4, x - 3 : x + 4] = 30
    return image


def test_flags_exact_duplicates_without_removing_regions() -> None:
    base = _notation_image()
    regions = [
        _region("region_001", "region_001.png", 1.0),
        _region("region_002", "region_002.png", 1.4),
    ]
    images = {
        Path("region_001.png"): base,
        Path("region_002.png"): base.copy(),
    }

    flags = flag_duplicate_regions(regions, image_loader=images.__getitem__)

    assert len(flags) == len(regions)
    assert [region.id for region in regions] == ["region_001", "region_002"]
    assert flags[0].exact_duplicate is False
    assert flags[0].near_duplicate is False
    assert flags[0].repeat_candidate is False
    assert flags[1].exact_duplicate is True
    assert flags[1].near_duplicate is True
    assert flags[1].repeat_candidate is False
    assert flags[1].matched_region_id == "region_001"
    assert flags[1].similarity is not None
    assert flags[1].similarity >= 0.995


def test_flags_near_duplicate_repeats_as_advisory_candidates() -> None:
    regions = [
        _region("region_001", "region_001.png", 1.0),
        _region("region_002", "region_002.png", 7.0),
        _region("region_003", "region_003.png", 8.0),
    ]
    images = {
        Path("region_001.png"): _notation_image(),
        Path("region_002.png"): _notation_image(shifted=True),
        Path("region_003.png"): np.full((80, 120, 3), 245, dtype=np.uint8),
    }

    flags = flag_duplicate_regions(
        regions,
        image_loader=images.__getitem__,
        repeat_time_gap_seconds=3.0,
    )

    assert len(flags) == 3
    assert flags[1].exact_duplicate is False
    assert flags[1].near_duplicate is True
    assert flags[1].repeat_candidate is True
    assert flags[1].matched_region_id == "region_001"
    assert 0.90 <= flags[1].similarity < 0.995
    assert flags[2].exact_duplicate is False
    assert flags[2].near_duplicate is False
    assert flags[2].repeat_candidate is False
    assert flags[2].matched_region_id is None


def test_comparison_normalizes_grayscale_and_size() -> None:
    base = _notation_image(width=180, height=120)
    smaller = _notation_image(width=120, height=80)
    regions = [
        _region("region_001", "region_001.png", 0.0),
        _region("region_002", "region_002.png", 0.5),
    ]
    images = {
        Path("region_001.png"): base[:, :, ::-1],
        Path("region_002.png"): smaller,
    }

    flags = flag_duplicate_regions(regions, image_loader=images.__getitem__)

    assert flags[1].near_duplicate is True
    assert flags[1].matched_region_id == "region_001"


def test_flags_duplicates_from_in_memory_images_without_region_paths() -> None:
    first = _region("region_001", "unused_001.png", 1.0).__class__(
        id="region_001",
        stable_view_id="view_001",
        source_timestamp_seconds=1.0,
        image_path=None,
        bounding_box=BoundingBox(0, 0, 120, 80),
        confidence=0.95,
        kind=RegionKind.PARTIAL_VIEW,
    )
    second = first.__class__(
        id="region_002",
        stable_view_id="view_002",
        source_timestamp_seconds=2.0,
        image_path=None,
        bounding_box=BoundingBox(0, 0, 120, 80),
        confidence=0.95,
        kind=RegionKind.PARTIAL_VIEW,
    )
    image = _notation_image()

    flags = flag_duplicate_regions(
        [first, second],
        images_by_region_id={"region_001": image, "region_002": image.copy()},
    )

    assert flags[1].exact_duplicate is True
    assert flags[1].matched_region_id == "region_001"


def test_duplicate_policy_only_suppresses_non_repeat_overlap_candidates() -> None:
    original = _region("region_001", "region_001.png", 1.0)
    duplicate = replace(
        _region("region_002", "region_002.png", 1.4),
        duplicate_flags=DuplicateFlags(
            near_duplicate=True,
            matched_region_id="region_001",
            repeat_candidate=False,
        ),
    )
    repeat = replace(
        _region("region_003", "region_003.png", 8.0),
        duplicate_flags=DuplicateFlags(
            near_duplicate=True,
            matched_region_id="region_001",
            repeat_candidate=True,
        ),
    )
    regions = [original, duplicate, repeat]

    assert apply_duplicate_policy(regions, DuplicatePolicy.FLAG) == regions
    assert apply_duplicate_policy(regions, DuplicatePolicy.FLAG_AND_INCLUDE) == regions
    assert apply_duplicate_policy(regions, DuplicatePolicy.FLAG_AND_SUPPRESS_OVERLAP) == [original, repeat]
