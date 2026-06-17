import numpy as np

from sheet_video_to_pdf.models import BoundingBox, RegionKind
from sheet_video_to_pdf.preprocess import (
    denoise_grayscale,
    resize_for_comparison,
    to_grayscale,
)
from sheet_video_to_pdf.regions import classify_region_kind, detect_notation_region


def _staff_frame(
    height=320,
    width=700,
    *,
    top=70,
    system_count=2,
    include_border=True,
    include_lyrics=True,
):
    frame = np.full((height, width, 3), 255, dtype=np.uint8)
    if include_border:
        frame[0:8, :] = 0
        frame[-8:, :] = 0
        frame[:, 0:10] = 0
        frame[:, -10:] = 0

    for system in range(system_count):
        y0 = top + system * 95
        for line in range(5):
            y = y0 + line * 8
            frame[y : y + 2, 82:620] = 0
        frame[y0 + 12 : y0 + 20, 185:197] = 0
        frame[y0 + 20 : y0 + 30, 300:312] = 0
        if include_lyrics:
            lyric_y = y0 + 48
            frame[lyric_y : lyric_y + 5, 215:330] = 35

    return frame


def test_to_grayscale_accepts_rgb_and_bgr_color_orders():
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    rgb[:, :, 0] = 255
    bgr = rgb[:, :, ::-1]

    rgb_gray = to_grayscale(rgb, color_order="RGB")
    bgr_gray = to_grayscale(bgr, color_order="BGR")

    assert rgb_gray.shape == (2, 2)
    assert rgb_gray.dtype == np.uint8
    assert np.array_equal(rgb_gray, bgr_gray)


def test_resize_for_comparison_uses_consistent_max_dimension():
    image = np.zeros((800, 400), dtype=np.uint8)

    resized = resize_for_comparison(image, max_dimension=200)

    assert resized.shape == (200, 100)


def test_denoise_grayscale_smooths_compression_speckles_without_resizing():
    image = np.full((41, 41), 255, dtype=np.uint8)
    image[20, 20] = 0

    denoised = denoise_grayscale(image)

    assert denoised.shape == image.shape
    assert denoised.dtype == np.uint8
    assert denoised[20, 20] > image[20, 20]


def test_detect_notation_region_trims_borders_and_preserves_nearby_lyrics():
    frame = _staff_frame()

    region = detect_notation_region(
        frame,
        region_id="region_001",
        stable_view_id="view_001",
        source_timestamp_seconds=12.5,
    )

    assert region.id == "region_001"
    assert region.stable_view_id == "view_001"
    assert region.source_timestamp_seconds == 12.5
    assert region.fallback_used is False
    assert region.confidence >= 0.5
    assert region.bounding_box.x > 10
    assert region.bounding_box.y > 8
    assert region.bounding_box.x + region.bounding_box.width < frame.shape[1] - 10
    assert region.bounding_box.y + region.bounding_box.height < frame.shape[0] - 8
    assert region.bounding_box.y + region.bounding_box.height >= 218
    assert region.kind is RegionKind.PARTIAL_VIEW


def test_detect_notation_region_falls_back_to_full_frame_when_confidence_is_low():
    frame = np.full((240, 320, 3), 255, dtype=np.uint8)

    region = detect_notation_region(
        frame,
        region_id="region_002",
        stable_view_id="view_002",
        source_timestamp_seconds=1.0,
    )

    assert region.bounding_box == BoundingBox(0, 0, 320, 240)
    assert region.fallback_used is True
    assert region.kind is RegionKind.UNKNOWN
    assert region.confidence == 0.0
    assert any("low confidence" in warning for warning in region.warnings)


def test_classify_region_kind_identifies_complete_page_from_page_shape_and_systems():
    frame = _staff_frame(
        height=850,
        width=620,
        top=80,
        system_count=5,
        include_border=False,
    )
    box = BoundingBox(54, 45, 520, 720)

    kind = classify_region_kind(to_grayscale(frame, color_order="BGR"), box)

    assert kind is RegionKind.COMPLETE_PAGE
