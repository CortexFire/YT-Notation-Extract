from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

from .errors import VideoReadError
from .preprocess import prepare_for_comparison


@dataclass(frozen=True)
class SampledFrameRef:
    sample_index: int
    source_frame_index: int
    timestamp_seconds: float


@dataclass(frozen=True)
class SampleAnalysis:
    refs: list[SampledFrameRef]
    prepared_frames: list[np.ndarray]
    sampled_fps: float
    frame_step: int


def analyze_sampled_frames(
    path: str | Path,
    source_fps: float,
    target_fps: float = 2.0,
) -> SampleAnalysis:
    if source_fps <= 0:
        raise VideoReadError("Video frame rate is unreadable; cannot sample frames")
    if target_fps <= 0:
        raise ValueError("target_fps must be positive")

    frame_step = max(1, int(round(source_fps / target_fps)))
    sampled_fps = source_fps / frame_step
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise VideoReadError(
            f"OpenCV could not open the MP4: {path}. Verify codec and FFmpeg support."
        )

    refs: list[SampledFrameRef] = []
    prepared_frames: list[np.ndarray] = []
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % frame_step == 0:
                sample_index = len(refs)
                refs.append(
                    SampledFrameRef(
                        sample_index=sample_index,
                        source_frame_index=frame_index,
                        timestamp_seconds=frame_index / source_fps,
                    )
                )
                prepared_frames.append(prepare_for_comparison(frame, max_dimension=160))
            frame_index += 1
    finally:
        capture.release()

    if not refs:
        raise VideoReadError(
            f"OpenCV could not decode sampled frames from {path}. Verify codec and FFmpeg support."
        )

    return SampleAnalysis(
        refs=refs,
        prepared_frames=prepared_frames,
        sampled_fps=sampled_fps,
        frame_step=frame_step,
    )


def read_sampled_frames_by_index(
    path: str | Path,
    refs: Sequence[SampledFrameRef],
    sample_indexes: Iterable[int],
) -> dict[int, np.ndarray]:
    requested = set(sample_indexes)
    if not requested:
        return {}

    refs_by_source_index = {
        ref.source_frame_index: ref.sample_index
        for ref in refs
        if ref.sample_index in requested
    }
    if not refs_by_source_index:
        return {}

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise VideoReadError(
            f"OpenCV could not open the MP4: {path}. Verify codec and FFmpeg support."
        )

    frames: dict[int, np.ndarray] = {}
    final_source_index = max(refs_by_source_index)
    frame_index = 0
    try:
        while frame_index <= final_source_index:
            ok, frame = capture.read()
            if not ok:
                break
            sample_index = refs_by_source_index.get(frame_index)
            if sample_index is not None:
                frames[sample_index] = frame
            frame_index += 1
    finally:
        capture.release()

    missing = requested - set(frames)
    if missing:
        raise VideoReadError(
            f"OpenCV could not decode requested sampled frames from {path}. "
            "Verify codec and FFmpeg support."
        )

    return frames
