from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from sheet_video_to_pdf.errors import VideoReadError
from sheet_video_to_pdf.video import decode_first_frame, validate_mp4
from tests.fixtures.synthetic_video import (
    create_moving_sheet_music_video,
    create_static_sheet_music_video,
)


def test_validate_mp4_rejects_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.mp4"

    with pytest.raises(VideoReadError, match="does not exist"):
        validate_mp4(missing)


def test_validate_mp4_requires_mp4_extension(tmp_path: Path) -> None:
    not_mp4 = tmp_path / "score.mov"
    not_mp4.write_bytes(b"not really a movie")

    with pytest.raises(VideoReadError, match=r"\.mp4"):
        validate_mp4(not_mp4)


def test_validate_mp4_reads_metadata_and_first_frame(tmp_path: Path) -> None:
    video_path = create_static_sheet_music_video(
        tmp_path / "static.mp4",
        frame_count=6,
        fps=6.0,
        frame_size=(144, 96),
    )

    metadata = validate_mp4(video_path)
    first_frame = decode_first_frame(video_path)

    assert metadata.path == video_path
    assert metadata.frame_count == 6
    assert metadata.frame_rate == pytest.approx(6.0, rel=0.05)
    assert metadata.duration_seconds == pytest.approx(1.0, rel=0.15)
    assert metadata.width == 144
    assert metadata.height == 96
    assert first_frame.shape == (96, 144, 3)
    assert first_frame.dtype == np.uint8
    assert first_frame.mean() < 255.0


def test_unreadable_mp4_error_mentions_codec_or_ffmpeg(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_bytes(b"this is not a decodable mp4")

    with pytest.raises(VideoReadError, match=r"(?i)(codec|ffmpeg)"):
        validate_mp4(corrupt)


def test_synthetic_moving_video_contains_holds_transitions_and_vertical_movement(
    tmp_path: Path,
) -> None:
    video_path = create_moving_sheet_music_video(
        tmp_path / "moving.mp4",
        positions=(0, 12, 24),
        hold_frames=2,
        transition_frames=1,
        fps=5.0,
        frame_size=(144, 96),
    )

    capture = cv2.VideoCapture(str(video_path))
    try:
        frames: list[np.ndarray] = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()

    assert len(frames) == 8
    assert np.mean(cv2.absdiff(frames[0], frames[1])) < 1.0
    assert np.mean(cv2.absdiff(frames[1], frames[2])) > 1.0
    assert np.mean(cv2.absdiff(frames[3], frames[5])) > 1.0
