from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from sheet_video_to_pdf.errors import VideoReadError
from sheet_video_to_pdf.models import VideoMetadata


def validate_mp4(path: str | Path) -> VideoMetadata:
    """Validate that a local MP4 can be opened, inspected, and decoded."""
    video_path = _validate_mp4_path(path)
    capture = _open_capture(video_path)
    try:
        metadata = _read_metadata(video_path, capture)
        ok, _frame = capture.read()
        if not ok:
            raise _codec_error(video_path, "OpenCV opened the file but could not decode a frame")
        return metadata
    finally:
        capture.release()


def decode_first_frame(path: str | Path) -> np.ndarray:
    """Decode and return the first frame of a valid local MP4."""
    video_path = _validate_mp4_path(path)
    capture = _open_capture(video_path)
    try:
        ok, frame = capture.read()
        if not ok or frame is None:
            raise _codec_error(video_path, "OpenCV could not decode the first frame")
        return frame
    finally:
        capture.release()


def _validate_mp4_path(path: str | Path) -> Path:
    video_path = Path(path)
    if not video_path.exists():
        raise VideoReadError(f"Input video does not exist: {video_path}")
    if not video_path.is_file():
        raise VideoReadError(f"Input video is not a file: {video_path}")
    if video_path.suffix.lower() != ".mp4":
        raise VideoReadError(f"Input video must use the .mp4 extension: {video_path}")
    return video_path


def _open_capture(path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise _codec_error(path, "OpenCV could not open the MP4")
    return capture


def _read_metadata(path: Path, capture: cv2.VideoCapture) -> VideoMetadata:
    frame_rate = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    if not _positive_finite(frame_rate):
        raise VideoReadError(f"Video metadata is unreadable: frame rate is missing for {path}")
    if frame_count <= 0:
        raise VideoReadError(f"Video metadata is unreadable: frame count is missing for {path}")
    if width <= 0 or height <= 0:
        raise VideoReadError(f"Video metadata is unreadable: frame size is missing for {path}")

    duration_seconds = frame_count / frame_rate
    if not _positive_finite(duration_seconds):
        raise VideoReadError(f"Video metadata is unreadable: duration is missing for {path}")

    return VideoMetadata(
        path=path,
        duration_seconds=duration_seconds,
        frame_rate=frame_rate,
        frame_count=frame_count,
        width=width,
        height=height,
    )


def _positive_finite(value: float) -> bool:
    return value > 0 and math.isfinite(value)


def _codec_error(path: Path, reason: str) -> VideoReadError:
    return VideoReadError(
        f"{reason}: {path}. Verify that the file uses a supported MP4 codec "
        "and that FFmpeg support is installed for OpenCV."
    )
