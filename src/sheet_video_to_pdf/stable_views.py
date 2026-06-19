from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from .cadence import CandidateFrame
from .models import StableView
from .preprocess import to_grayscale


@dataclass(frozen=True)
class RejectedStableCandidate:
    frame_index: int
    timestamp_seconds: float
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class StableViewSelection:
    accepted: list[StableView]
    rejected: list[RejectedStableCandidate]


@dataclass(frozen=True)
class StableCandidatePreselection:
    candidates: list[CandidateFrame]
    rejected: list[RejectedStableCandidate]


def preselect_stable_candidates_from_prepared(
    prepared_frames: Sequence[np.ndarray],
    candidates: Sequence[CandidateFrame],
    *,
    min_content_score: float = 0.015,
) -> StableCandidatePreselection:
    accepted: list[CandidateFrame] = []
    rejected: list[RejectedStableCandidate] = []
    last_signature: np.ndarray | None = None

    for candidate in candidates:
        notes = list(candidate.notes)
        if candidate.frame_index < 0 or candidate.frame_index >= len(prepared_frames):
            notes.append("frame-index-out-of-range")
            rejected.append(RejectedStableCandidate(candidate.frame_index, candidate.timestamp_seconds, notes))
            continue

        prepared = _normalize_prepared_frame(prepared_frames[candidate.frame_index])
        content_score = _prepared_content_score(prepared)
        if "static-hold" in notes:
            notes.append("static-hold-collapsed")
            rejected.append(RejectedStableCandidate(candidate.frame_index, candidate.timestamp_seconds, notes))
            continue
        if "motion-heavy" in notes or candidate.stability_score < 0.45:
            if "motion-heavy" not in notes:
                notes.append("motion-heavy")
            rejected.append(RejectedStableCandidate(candidate.frame_index, candidate.timestamp_seconds, notes))
            continue
        if content_score < min_content_score:
            notes.append("too-small-content")
            rejected.append(RejectedStableCandidate(candidate.frame_index, candidate.timestamp_seconds, notes))
            continue

        signature = _prepared_signature(prepared)
        if last_signature is not None and "change-adjacent" not in notes:
            if float(np.mean(np.abs(signature - last_signature))) < 0.01:
                notes.append("prepared-static-hold-collapsed")
                rejected.append(RejectedStableCandidate(candidate.frame_index, candidate.timestamp_seconds, notes))
                continue

        accepted.append(candidate)
        last_signature = signature

    return StableCandidatePreselection(candidates=accepted, rejected=rejected)


def select_stable_views(
    frames: Sequence[np.ndarray],
    candidates: Sequence[CandidateFrame],
    *,
    min_content_score: float = 0.015,
) -> StableViewSelection:
    return select_stable_views_from_frame_map(
        {index: frame for index, frame in enumerate(frames)},
        candidates,
        min_content_score=min_content_score,
    )


def select_stable_views_from_frame_map(
    frames_by_sample_index: Mapping[int, np.ndarray],
    candidates: Sequence[CandidateFrame],
    *,
    source_frame_indexes: Mapping[int, int] | None = None,
    min_content_score: float = 0.015,
) -> StableViewSelection:
    accepted: list[StableView] = []
    rejected: list[RejectedStableCandidate] = []
    last_signature: np.ndarray | None = None

    for candidate in candidates:
        notes = list(candidate.notes)
        if candidate.frame_index not in frames_by_sample_index:
            notes.append("frame-index-out-of-range")
            rejected.append(RejectedStableCandidate(candidate.frame_index, candidate.timestamp_seconds, notes))
            continue

        frame = frames_by_sample_index[candidate.frame_index]
        content_score = _content_score(frame)
        if "static-hold" in notes:
            notes.append("static-hold-collapsed")
            rejected.append(RejectedStableCandidate(candidate.frame_index, candidate.timestamp_seconds, notes))
            continue
        if "motion-heavy" in notes or candidate.stability_score < 0.45:
            if "motion-heavy" not in notes:
                notes.append("motion-heavy")
            rejected.append(RejectedStableCandidate(candidate.frame_index, candidate.timestamp_seconds, notes))
            continue
        if content_score < min_content_score:
            notes.append("too-small-content")
            rejected.append(RejectedStableCandidate(candidate.frame_index, candidate.timestamp_seconds, notes))
            continue
        if _blur_score(frame) < _blur_threshold(frame):
            notes.append("motion-blur")
            rejected.append(RejectedStableCandidate(candidate.frame_index, candidate.timestamp_seconds, notes))
            continue

        signature = _signature(frame)
        if (
            last_signature is not None
            and "change-adjacent" not in notes
        ):
            if float(np.mean(np.abs(signature - last_signature))) < 0.01:
                notes.append("static-hold-collapsed")
                rejected.append(RejectedStableCandidate(candidate.frame_index, candidate.timestamp_seconds, notes))
                continue

        accepted.append(
            StableView(
                id=f"view_{len(accepted) + 1:03d}",
                timestamp_seconds=candidate.timestamp_seconds,
                frame_index=candidate.frame_index,
                frame_path=None,
                stability_score=round(candidate.stability_score, 3),
                source_frame_index=(
                    source_frame_indexes.get(candidate.frame_index)
                    if source_frame_indexes is not None
                    else None
                ),
                rejection_notes=[],
            )
        )
        last_signature = signature

    return StableViewSelection(accepted=accepted, rejected=rejected)


def _content_score(frame: np.ndarray) -> float:
    gray = to_grayscale(frame)
    return float(np.mean(gray < 230))


def _normalize_prepared_frame(frame: np.ndarray) -> np.ndarray:
    prepared = np.asarray(frame)
    if prepared.ndim != 2:
        prepared = to_grayscale(prepared)
    if prepared.dtype == np.float32 and prepared.max(initial=0.0) <= 1.0:
        return prepared
    return prepared.astype(np.float32) / 255.0


def _prepared_content_score(frame: np.ndarray) -> float:
    return float(np.mean(frame < 0.90))


def _prepared_signature(frame: np.ndarray) -> np.ndarray:
    y_idx = np.linspace(0, frame.shape[0] - 1, 48).astype(int)
    x_idx = np.linspace(0, frame.shape[1] - 1, 48).astype(int)
    return frame[y_idx[:, None], x_idx[None, :]]


def _signature(frame: np.ndarray) -> np.ndarray:
    gray = to_grayscale(frame).astype(np.float32) / 255.0
    y_idx = np.linspace(0, gray.shape[0] - 1, 48).astype(int)
    x_idx = np.linspace(0, gray.shape[1] - 1, 48).astype(int)
    return gray[y_idx[:, None], x_idx[None, :]]


def _blur_score(frame: np.ndarray) -> float:
    gray = to_grayscale(frame).astype(np.float32) / 255.0
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    center = gray[1:-1, 1:-1]
    laplacian = (
        4 * center
        - gray[:-2, 1:-1]
        - gray[2:, 1:-1]
        - gray[1:-1, :-2]
        - gray[1:-1, 2:]
    )
    return float(np.var(laplacian))


def _blur_threshold(frame: np.ndarray) -> float:
    height, width = frame.shape[:2]
    if max(height, width) >= 720:
        return 0.005
    return 0.155


__all__ = [
    "RejectedStableCandidate",
    "StableCandidatePreselection",
    "StableViewSelection",
    "preselect_stable_candidates_from_prepared",
    "select_stable_views",
    "select_stable_views_from_frame_map",
]
