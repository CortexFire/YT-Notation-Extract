from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .models import CadenceDecision
from .preprocess import prepare_for_comparison


@dataclass(frozen=True)
class CandidateFrame:
    frame_index: int
    timestamp_seconds: float
    change_score: float
    stability_score: float
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CadenceAnalysis:
    decisions: list[CadenceDecision]
    candidates: list[CandidateFrame]
    rejected_candidates: list[CandidateFrame]


def determine_adaptive_cadence(
    frames: Sequence[np.ndarray],
    *,
    fps: float,
    max_dimension: int = 160,
) -> CadenceAnalysis:
    if fps <= 0:
        raise ValueError("fps must be positive")
    if not frames:
        return CadenceAnalysis([], [], [])

    prepared = [
        prepare_for_comparison(frame, max_dimension=max_dimension).astype(np.float32) / 255.0
        for frame in frames
    ]
    changes = [0.0]
    changes.extend(_change_score(prepared[index - 1], prepared[index]) for index in range(1, len(prepared)))
    content_scores = [_content_score(frame) for frame in prepared]

    candidates: list[CandidateFrame] = []
    rejected: list[CandidateFrame] = []
    decisions: list[CadenceDecision] = []
    pending_change = False
    last_candidate_signature: np.ndarray | None = None
    periodic_interval_frames = max(1, int(round(fps * 2.0)))

    for index, change in enumerate(changes):
        timestamp = index / fps
        stability = max(0.0, 1.0 - change)
        notes: list[str] = []

        if _is_low_content_transition(index, content_scores):
            notes.append("transition")
            rejected.append(CandidateFrame(index, timestamp, change, stability, notes))
            decisions.append(_decision(index, fps, "transition low-content range", change))
            pending_change = True
            continue

        if change >= 0.16:
            notes.append("motion-heavy")
            rejected.append(CandidateFrame(index, timestamp, change, stability, notes))
            decisions.append(_decision(index, fps, "motion-heavy", change))
            pending_change = True
            continue

        frame_signature = prepared[index]
        is_static_hold = (
            last_candidate_signature is not None
            and _change_score(last_candidate_signature, frame_signature) < 0.035
        )

        if index == 0:
            notes.append("initial")
        elif pending_change or (index > 0 and changes[index - 1] >= 0.12):
            notes.append("change-adjacent")
            decisions.append(_decision(index, fps, "change-adjacent dense analysis", change))
            pending_change = False
        elif index % periodic_interval_frames == 0 and content_scores[index] >= 0.02:
            notes.append("periodic-sample")
            decisions.append(_decision(index, fps, "periodic content sample", change))
        elif is_static_hold:
            notes.append("static-hold")
            rejected.append(CandidateFrame(index, timestamp, change, stability, notes))
            decisions.append(_decision(index, fps, "static-hold reduced analysis", change))
            continue
        else:
            notes.append("stable")

        candidate = CandidateFrame(index, timestamp, change, stability, notes)
        candidates.append(candidate)
        last_candidate_signature = frame_signature

    return CadenceAnalysis(decisions, candidates, rejected)


def _change_score(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape != second.shape:
        raise ValueError("comparison frames must have matching shapes")
    return float(np.mean(np.abs(first - second)))


def _content_score(frame: np.ndarray) -> float:
    return float(np.mean(frame < 0.90))


def _is_low_content_transition(index: int, content_scores: list[float]) -> bool:
    if content_scores[index] >= 0.01:
        return False
    previous_has_content = index > 0 and content_scores[index - 1] >= 0.02
    next_has_content = index + 1 < len(content_scores) and content_scores[index + 1] >= 0.02
    return previous_has_content or next_has_content


def _decision(index: int, fps: float, reason: str, change: float) -> CadenceDecision:
    timestamp = index / fps
    return CadenceDecision(
        start_seconds=timestamp,
        end_seconds=timestamp,
        interval_seconds=1.0 / fps,
        reason=reason,
        average_change=round(change, 4),
    )


__all__ = ["CadenceAnalysis", "CandidateFrame", "determine_adaptive_cadence"]
