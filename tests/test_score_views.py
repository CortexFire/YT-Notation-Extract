from __future__ import annotations

import cv2
import numpy as np
import pytest

from sheet_video_to_pdf.errors import NotationLocalizationError
from sheet_video_to_pdf.models import BoundingBox
from sheet_video_to_pdf.score_views import fuse_score_frames, group_score_states


ROI = BoundingBox(20, 15, 480, 150)


def _score_crop(*, variant: int = 0, cursor_x: int | None = None) -> np.ndarray:
    score = np.full((150, 480, 3), 250, dtype=np.uint8)
    for staff_top in (35, 92):
        for line in range(5):
            y = staff_top + line * 5
            cv2.line(score, (12, y), (466, y), (20, 20, 20), 1)
        note_positions = (80, 170, 280, 390) if variant == 0 else (120, 220, 330, 430)
        for note_x in note_positions:
            cv2.rectangle(score, (note_x, staff_top + 4), (note_x + 8, staff_top + 19), (15, 15, 15), -1)
    if cursor_x is not None:
        cv2.line(score, (cursor_x, 20), (cursor_x, 135), (20, 225, 40), 5)
    return score


def _full_frame(*, variant: int, cursor_x: int, clutter_seed: int) -> np.ndarray:
    frame = np.full((260, 540, 3), 30, dtype=np.uint8)
    frame[ROI.y : ROI.y + ROI.height, ROI.x : ROI.x + ROI.width] = _score_crop(
        variant=variant,
        cursor_x=cursor_x,
    )
    rng = np.random.default_rng(clutter_seed)
    frame[180:250] = rng.integers(0, 255, size=frame[180:250].shape, dtype=np.uint8)
    return frame


def test_group_score_states_ignores_moving_cursor_and_outside_roi_motion():
    frames = [
        *[
            _full_frame(variant=0, cursor_x=55 + index * 55, clutter_seed=index)
            for index in range(5)
        ],
        *[
            _full_frame(variant=1, cursor_x=65 + index * 55, clutter_seed=100 + index)
            for index in range(5)
        ],
    ]

    groups = group_score_states(
        frames,
        timestamps_seconds=[index * 0.5 for index in range(10)],
        notation_roi=ROI,
    )

    assert len(groups) == 2
    assert groups[0].sample_indexes == [0, 1, 2, 3, 4]
    assert groups[0].start_seconds == 0.0
    assert groups[0].end_seconds == 2.0
    assert groups[1].sample_indexes == [5, 6, 7, 8, 9]
    assert groups[1].start_seconds == 2.5
    assert groups[1].end_seconds == 4.5


def test_fuse_score_frames_removes_moving_overlay_and_preserves_covered_note():
    clean = _score_crop(variant=0)
    frames = [
        _score_crop(variant=0, cursor_x=84),
        _score_crop(variant=0, cursor_x=204),
        _score_crop(variant=0, cursor_x=324),
    ]

    fused = fuse_score_frames(frames, start_seconds=0.0, end_seconds=2.0)

    assert fused.source_frame_count == 3
    assert fused.cleanup_confidence >= 0.95
    assert fused.image[45, 84] < 50
    assert fused.image[25, 84] > 235
    assert np.mean(np.abs(fused.image.astype(np.int16) - cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY))) < 2.0


def test_fuse_score_frames_fails_closed_for_single_frame_with_colored_overlay():
    frame = _score_crop(variant=0, cursor_x=84)

    with pytest.raises(NotationLocalizationError, match="0.0s-0.5s"):
        fuse_score_frames([frame], start_seconds=0.0, end_seconds=0.5)


def test_fuse_score_frames_accepts_clean_single_frame():
    clean = _score_crop(variant=0)

    fused = fuse_score_frames([clean], start_seconds=0.0, end_seconds=0.5)

    assert fused.source_frame_count == 1
    assert fused.cleanup_confidence == 1.0
    assert np.array_equal(fused.image, cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY))
