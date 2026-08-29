from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import warnings

import cv2
import numpy as np

from .errors import NotationLocalizationError
from .localization import has_staff_evidence
from .models import BoundingBox
from .preprocess import resize_for_comparison, to_grayscale


@dataclass(frozen=True)
class ScoreStateGroup:
    sample_indexes: list[int]
    start_seconds: float
    end_seconds: float
    representative_sample_index: int


@dataclass(frozen=True)
class FusedScoreView:
    image: np.ndarray
    cleanup_confidence: float
    source_frame_count: int
    overlay_occupancy: float


def group_score_states(
    frames: Sequence[np.ndarray],
    *,
    timestamps_seconds: Sequence[float],
    notation_roi: BoundingBox,
    change_threshold: float | None = None,
) -> list[ScoreStateGroup]:
    if len(frames) != len(timestamps_seconds):
        raise ValueError("timestamps_seconds must match the number of frames")
    if not frames:
        return []

    prepared = [
        prepare_score_for_comparison(_crop(frame, notation_roi))
        for frame in frames
    ]
    return group_score_states_from_prepared(
        prepared,
        timestamps_seconds=timestamps_seconds,
        change_threshold=change_threshold,
    )


def group_score_states_from_prepared(
    prepared_frames: Sequence[np.ndarray],
    *,
    timestamps_seconds: Sequence[float],
    change_threshold: float | None = None,
) -> list[ScoreStateGroup]:
    if len(prepared_frames) != len(timestamps_seconds):
        raise ValueError("timestamps_seconds must match the number of prepared frames")
    if not prepared_frames:
        return []

    distances = [
        _score_distance(prepared_frames[index - 1], prepared_frames[index])
        for index in range(1, len(prepared_frames))
    ]
    threshold = change_threshold if change_threshold is not None else _adaptive_change_threshold(distances)
    groups: list[list[int]] = [[0]]
    for index, distance in enumerate(distances, start=1):
        if distance >= threshold:
            groups.append([index])
        else:
            groups[-1].append(index)

    return [
        ScoreStateGroup(
            sample_indexes=indexes,
            start_seconds=float(timestamps_seconds[indexes[0]]),
            end_seconds=float(timestamps_seconds[indexes[-1]]),
            representative_sample_index=indexes[len(indexes) // 2],
        )
        for indexes in groups
    ]


def _adaptive_change_threshold(distances: Sequence[float]) -> float:
    if len(distances) < 2:
        return 0.08
    ordered = sorted(float(value) for value in distances)
    start = max(0, len(ordered) // 2 - 1)
    best_low = 0.0
    best_high = 0.0
    best_gap = 0.0
    for low, high in zip(ordered[start:-1], ordered[start + 1 :]):
        gap = high - low
        if gap > best_gap:
            best_low = low
            best_high = high
            best_gap = gap
    if best_gap >= 0.02 and best_high >= 0.05 and best_high >= max(0.001, best_low) * 1.8:
        return (best_low + best_high) / 2.0
    return 0.08


def prepare_score_for_comparison(frame: np.ndarray, *, max_dimension: int = 320) -> np.ndarray:
    bgr = _as_bgr(frame)
    saturation_mask = _saturated_overlay_mask(bgr)
    gray = to_grayscale(bgr)
    gray[saturation_mask] = 255
    resized = resize_for_comparison(gray, max_dimension=max_dimension)
    dark = (resized < 190).astype(np.uint8)
    return cv2.morphologyEx(
        dark,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)),
    )


def fuse_score_frames(
    frames: Sequence[np.ndarray],
    *,
    start_seconds: float,
    end_seconds: float,
) -> FusedScoreView:
    if not frames:
        raise NotationLocalizationError(
            f"No score frames were available for {start_seconds:.1f}s-{end_seconds:.1f}s"
        )
    shape = frames[0].shape[:2]
    if any(frame.shape[:2] != shape for frame in frames):
        raise ValueError("score frames must have matching dimensions")

    bgr_frames = [_as_bgr(frame) for frame in frames]
    overlay_masks = [_saturated_overlay_mask(frame) for frame in bgr_frames]
    overlay_occupancy = float(np.mean(np.stack(overlay_masks)))
    gray_frames = [to_grayscale(frame) for frame in bgr_frames]

    if len(frames) < 3:
        if overlay_occupancy > 0.001:
            raise NotationLocalizationError(
                "Could not remove notation overlays in the short score view at "
                f"{start_seconds:.1f}s-{end_seconds:.1f}s"
            )
        sharpest = max(gray_frames, key=_sharpness)
        if not has_staff_evidence(sharpest):
            raise _cleanup_failure(start_seconds, end_seconds)
        return FusedScoreView(
            image=sharpest.copy(),
            cleanup_confidence=1.0,
            source_frame_count=len(frames),
            overlay_occupancy=round(overlay_occupancy, 4),
        )

    stack = np.stack(gray_frames).astype(np.float32)
    masks = np.stack(overlay_masks)
    masked = stack.copy()
    masked[masks] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        fused = np.nanmedian(masked, axis=0)
    all_masked = np.isnan(fused)
    if bool(all_masked.any()):
        fallback = np.median(stack, axis=0)
        fused[all_masked] = fallback[all_masked]

    required_samples = len(frames) // 2 + 1
    valid_counts = (~masks).sum(axis=0)
    cleanup_confidence = float(np.mean(valid_counts >= required_samples))
    all_masked_fraction = float(np.mean(valid_counts == 0))
    image = np.clip(fused, 0, 255).astype(np.uint8)
    if cleanup_confidence < 0.95 or all_masked_fraction > 0.001 or not has_staff_evidence(image):
        raise _cleanup_failure(start_seconds, end_seconds)

    return FusedScoreView(
        image=image,
        cleanup_confidence=round(cleanup_confidence, 3),
        source_frame_count=len(frames),
        overlay_occupancy=round(overlay_occupancy, 4),
    )


def _score_distance(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape != second.shape:
        raise ValueError("prepared score frames must have matching dimensions")
    first_dark = first > 0
    second_dark = second > 0
    union = np.logical_or(first_dark, second_dark).sum()
    if union == 0:
        return 0.0
    changed = np.logical_xor(first_dark, second_dark).sum()
    return float(changed / union)


def _saturated_overlay_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return (hsv[:, :, 1] >= 70) & (hsv[:, :, 2] >= 70)


def _sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def _crop(frame: np.ndarray, box: BoundingBox) -> np.ndarray:
    height, width = frame.shape[:2]
    if (
        box.x < 0
        or box.y < 0
        or box.x + box.width > width
        or box.y + box.height > height
    ):
        raise ValueError("notation_roi is outside the frame")
    return frame[box.y : box.y + box.height, box.x : box.x + box.width]


def _as_bgr(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 3:
        return frame.astype(np.uint8)
    raise ValueError("Expected a grayscale or BGR score frame")


def _cleanup_failure(start_seconds: float, end_seconds: float) -> NotationLocalizationError:
    return NotationLocalizationError(
        "Could not reconstruct clean notation with recognizable staff lines at "
        f"{start_seconds:.1f}s-{end_seconds:.1f}s"
    )


__all__ = [
    "FusedScoreView",
    "ScoreStateGroup",
    "fuse_score_frames",
    "group_score_states",
    "group_score_states_from_prepared",
    "prepare_score_for_comparison",
]
