from __future__ import annotations

import cv2
import numpy as np
import pytest

from sheet_video_to_pdf.errors import NotationLocalizationError
from sheet_video_to_pdf.localization import localize_notation_area, localize_notation_from_frames
from sheet_video_to_pdf.models import BoundingBox, LocalizationSource, VideoMetadata


def _cluttered_score_frame(*, cursor_x: int = 90, score_y: int = 18) -> np.ndarray:
    frame = np.full((360, 640, 3), 32, dtype=np.uint8)
    frame[190:250, 20:220] = (55, 55, 55)
    cv2.circle(frame, (120 + cursor_x // 8, 220), 28, (120, 170, 220), -1)
    frame[275:350, 0:640] = 245
    cv2.line(frame, (0, 275), (639, 275), (10, 10, 10), 2)
    for x in range(0, 640, 24):
        cv2.line(frame, (x, 275), (x, 350), (15, 15, 15), 2)

    frame[score_y : score_y + 132, 10:630] = 250

    for staff_top in (score_y + 38, score_y + 86):
        for line in range(5):
            y = staff_top + line * 5
            cv2.line(frame, (24, y), (616, y), (20, 20, 20), 1)
        cv2.rectangle(frame, (150, staff_top + 4), (158, staff_top + 18), (20, 20, 20), -1)
        cv2.rectangle(frame, (350, staff_top + 9), (358, staff_top + 23), (20, 20, 20), -1)

    cv2.line(
        frame,
        (cursor_x, score_y + 25),
        (cursor_x, score_y + 122),
        (20, 220, 40),
        4,
    )
    return frame


def test_auto_localization_selects_staff_panel_and_excludes_keyboard_and_visualizer():
    frames = [_cluttered_score_frame(cursor_x=70 + index * 55) for index in range(6)]

    result = localize_notation_from_frames(frames, timestamps_seconds=[float(i) for i in range(6)])

    box = result.bounding_box
    assert result.source is LocalizationSource.AUTO
    assert result.support_count == 6
    assert result.confidence >= 0.7
    assert box.x <= 16
    assert 18 <= box.y <= 24
    assert box.x + box.width >= 624
    assert 145 <= box.y + box.height <= 150


def test_manual_localization_returns_exact_validated_box():
    frames = [_cluttered_score_frame(cursor_x=90), _cluttered_score_frame(cursor_x=300)]
    manual = BoundingBox(10, 18, 620, 132)

    result = localize_notation_from_frames(frames, manual_roi=manual)

    assert result.source is LocalizationSource.MANUAL
    assert result.bounding_box == manual
    assert result.support_count == 2


def test_manual_localization_rejects_box_outside_source_frame():
    frames = [_cluttered_score_frame()]

    with pytest.raises(NotationLocalizationError, match="outside the video frame"):
        localize_notation_from_frames(frames, manual_roi=BoundingBox(500, 20, 200, 100))


def test_manual_localization_rejects_box_without_staff_evidence():
    frames = [_cluttered_score_frame()]

    with pytest.raises(NotationLocalizationError, match="staff lines"):
        localize_notation_from_frames(frames, manual_roi=BoundingBox(0, 180, 240, 75))


def test_auto_localization_fails_when_no_consistent_staff_panel_exists():
    frames = [np.full((240, 320, 3), 245, dtype=np.uint8) for _ in range(6)]

    with pytest.raises(NotationLocalizationError, match="consistent notation area"):
        localize_notation_from_frames(frames)


def test_auto_localization_fails_when_score_panel_moves_between_probe_groups():
    frames = [
        *[_cluttered_score_frame(score_y=18) for _ in range(3)],
        *[_cluttered_score_frame(score_y=180) for _ in range(3)],
    ]

    with pytest.raises(NotationLocalizationError, match="consistent notation area"):
        localize_notation_from_frames(frames)


def test_localize_notation_area_reads_evenly_spaced_video_probes(tmp_path):
    video_path = tmp_path / "cluttered.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        2.0,
        (640, 360),
    )
    for index in range(20):
        writer.write(_cluttered_score_frame(cursor_x=55 + index * 22))
    writer.release()
    metadata = VideoMetadata(
        path=video_path,
        duration_seconds=10.0,
        frame_rate=2.0,
        frame_count=20,
        width=640,
        height=360,
    )

    result = localize_notation_area(video_path, metadata, max_probes=6)

    assert result.support_count == 6
    assert len(result.probe_timestamps_seconds) == 6
    assert result.probe_timestamps_seconds[0] == 0.0
    assert result.probe_timestamps_seconds[-1] == 9.5
