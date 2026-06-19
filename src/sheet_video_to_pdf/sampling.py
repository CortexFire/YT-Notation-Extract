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
            if frame_index % frame_step == 0:
                ok, frame = capture.read()
                if not ok:
                    break
                sample_index = len(refs)
                refs.append(
                    SampledFrameRef(
                        sample_index=sample_index,
                        source_frame_index=frame_index,
                        timestamp_seconds=frame_index / source_fps,
                    )
                )
                prepared_frames.append(prepare_for_comparison(frame, max_dimension=160))
            else:
                ok = capture.grab()
                if not ok:
                    break
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

    final_source_index = max(refs_by_source_index)
    if _should_seek_for_sparse_reads(
        requested_count=len(refs_by_source_index),
        final_source_index=final_source_index,
    ):
        frames = _read_requested_frames_by_seek(path, refs_by_source_index)
        if set(frames) == requested:
            return frames

    return _read_requested_frames_sequentially(path, refs_by_source_index)


def _should_seek_for_sparse_reads(*, requested_count: int, final_source_index: int) -> bool:
    if requested_count <= 0:
        return False
    return final_source_index + 1 > requested_count * 8


def _read_requested_frames_by_seek(
    path: str | Path,
    refs_by_source_index: dict[int, int],
) -> dict[int, np.ndarray]:
    capture = _open_capture(path)
    frames: dict[int, np.ndarray] = {}
    try:
        for source_index, sample_index in sorted(refs_by_source_index.items()):
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, source_index):
                return frames
            ok, frame = capture.read()
            if not ok:
                return frames
            frames[sample_index] = frame
    finally:
        capture.release()
    return frames


def _read_requested_frames_sequentially(
    path: str | Path,
    refs_by_source_index: dict[int, int],
) -> dict[int, np.ndarray]:
    capture = _open_capture(path)
    requested = set(refs_by_source_index.values())
    final_source_index = max(refs_by_source_index)
    frames: dict[int, np.ndarray] = {}
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


def _open_capture(path: str | Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise VideoReadError(
            f"OpenCV could not open the MP4: {path}. Verify codec and FFmpeg support."
        )
    return capture
