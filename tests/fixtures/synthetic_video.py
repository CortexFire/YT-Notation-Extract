from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def create_static_sheet_music_video(
    path: str | Path,
    *,
    frame_count: int = 8,
    fps: float = 8.0,
    frame_size: tuple[int, int] = (160, 120),
) -> Path:
    """Create a deterministic MP4 with a held sheet-music-like frame."""
    output_path = Path(path)
    frames = [_sheet_frame(frame_size=frame_size, vertical_offset=0) for _ in range(frame_count)]
    _write_mp4(output_path, frames, fps)
    return output_path


def create_moving_sheet_music_video(
    path: str | Path,
    *,
    positions: tuple[int, ...] = (0, 10, 20),
    hold_frames: int = 2,
    transition_frames: int = 1,
    fps: float = 8.0,
    frame_size: tuple[int, int] = (160, 120),
) -> Path:
    """Create deterministic vertical movement with light gray transition frames."""
    output_path = Path(path)
    frames: list[np.ndarray] = []
    previous: np.ndarray | None = None
    for vertical_offset in positions:
        current = _sheet_frame(frame_size=frame_size, vertical_offset=vertical_offset)
        if previous is not None:
            for transition_index in range(transition_frames):
                alpha = (transition_index + 1) / (transition_frames + 1)
                frames.append(cv2.addWeighted(previous, 1.0 - alpha, current, alpha, 0.0))
        frames.extend(current.copy() for _ in range(hold_frames))
        previous = current
    _write_mp4(output_path, frames, fps)
    return output_path


def _sheet_frame(
    *,
    frame_size: tuple[int, int],
    vertical_offset: int,
) -> np.ndarray:
    width, height = frame_size
    frame = np.full((height, width, 3), 255, dtype=np.uint8)
    page_left = 16
    page_right = width - 16
    system_gap = 34
    staff_gap = 5
    top = 18 - vertical_offset

    for system_index in range(5):
        staff_top = top + system_index * system_gap
        for line_index in range(5):
            y = staff_top + line_index * staff_gap
            if 0 <= y < height:
                cv2.line(frame, (page_left, y), (page_right, y), (25, 25, 25), 1)
        for note_index in range(5):
            x = page_left + 12 + note_index * 22
            y = staff_top + 4 + ((note_index + system_index) % 5) * 3
            if 0 <= y < height:
                cv2.circle(frame, (x, y), 3, (15, 15, 15), -1)
                cv2.line(frame, (x + 3, y), (x + 3, y - 14), (15, 15, 15), 1)
    return frame


def _write_mp4(path: Path, frames: list[np.ndarray], fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError("synthetic videos require at least one frame")
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not create an MP4 writer for synthetic fixtures")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()
