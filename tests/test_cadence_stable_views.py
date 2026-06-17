from __future__ import annotations

import inspect

import numpy as np

from sheet_video_to_pdf.cadence import CandidateFrame, determine_adaptive_cadence
from sheet_video_to_pdf.models import CadenceDecision, StableView
from sheet_video_to_pdf.stable_views import select_stable_views


def _notation_frame(y_offset: int = 0, height: int = 96, width: int = 128) -> np.ndarray:
    frame = np.full((height, width, 3), 245, dtype=np.uint8)
    for y in (20 + y_offset, 28 + y_offset, 36 + y_offset, 56 + y_offset, 64 + y_offset):
        frame[max(0, y) : min(height, y + 2), 12:116] = 10
    for x, y in ((36, 26 + y_offset), (64, 34 + y_offset), (84, 62 + y_offset)):
        frame[max(0, y - 3) : min(height, y + 4), max(0, x - 3) : min(width, x + 4)] = 20
    return frame


def _motion_frame() -> np.ndarray:
    frame = np.full((96, 128, 3), 220, dtype=np.uint8)
    for y in range(12, 86, 12):
        frame[y : y + 1, 8:120] = 70
        frame[y + 2 : y + 3, 8:120] = 150
    return frame


def _blurred_notation_frame() -> np.ndarray:
    sharp = _notation_frame(0)
    blurred = sharp.copy()
    for shift in range(1, 8):
        blurred = ((blurred.astype(np.uint16) + np.roll(sharp, shift=shift, axis=1)) // 2).astype(np.uint8)
    return blurred


def _high_resolution_real_style_frame(y_offset: int = 0) -> np.ndarray:
    small = _notation_frame(y_offset=y_offset, height=96, width=128)
    large = np.full((1080, 1920, 3), 250, dtype=np.uint8)
    scaled = np.repeat(np.repeat(small, 8, axis=0), 8, axis=1)
    scaled = scaled[:768, :1024]
    large[120 : 120 + scaled.shape[0], 80 : 80 + scaled.shape[1]] = scaled
    return large


def test_adaptive_cadence_has_no_user_sample_fps_and_records_density_decisions() -> None:
    signature = inspect.signature(determine_adaptive_cadence)

    assert "sample_fps" not in signature.parameters

    first_view = [_notation_frame(0) for _ in range(6)]
    transition = [_motion_frame(), np.roll(_notation_frame(20), shift=5, axis=0)]
    second_view = [_notation_frame(20) for _ in range(6)]

    analysis = determine_adaptive_cadence(first_view + transition + second_view, fps=2.0)

    assert all(isinstance(decision, CadenceDecision) for decision in analysis.decisions)
    assert {candidate.frame_index for candidate in analysis.candidates} >= {0, 8}
    assert sum(candidate.frame_index < 6 for candidate in analysis.candidates) < 6
    assert any("static-hold" in decision.reason for decision in analysis.decisions)
    assert any("change-adjacent" in decision.reason for decision in analysis.decisions)
    assert any("motion-heavy" in decision.reason for decision in analysis.decisions)
    assert any("motion-heavy" in candidate.notes for candidate in analysis.rejected_candidates)


def test_adaptive_cadence_marks_low_content_fade_as_transition_range() -> None:
    frames = [
        _notation_frame(0),
        np.full((96, 128, 3), 250, dtype=np.uint8),
        _notation_frame(24),
    ]

    analysis = determine_adaptive_cadence(frames, fps=3.0)

    rejected_notes = {note for candidate in analysis.rejected_candidates for note in candidate.notes}
    assert "transition" in rejected_notes
    assert any("transition" in decision.reason for decision in analysis.decisions)


def test_adaptive_cadence_keeps_periodic_candidates_for_low_delta_notation_changes() -> None:
    frames: list[np.ndarray] = []
    for y_offset in (0, 4, 8):
        frames.extend([_high_resolution_real_style_frame(y_offset)] * 120)

    analysis = determine_adaptive_cadence(frames, fps=60.0)

    candidate_indexes = {candidate.frame_index for candidate in analysis.candidates}
    assert any(110 <= index <= 130 for index in candidate_indexes)
    assert any(230 <= index <= 250 for index in candidate_indexes)
    assert len(analysis.candidates) >= 3
    assert any("periodic-sample" in candidate.notes for candidate in analysis.candidates)


def test_adaptive_cadence_periodically_samples_visually_similar_score_frames() -> None:
    frames = [_high_resolution_real_style_frame(0) for _ in range(181)]

    analysis = determine_adaptive_cadence(frames, fps=60.0)

    periodic_indexes = [
        candidate.frame_index
        for candidate in analysis.candidates
        if "periodic-sample" in candidate.notes
    ]
    assert periodic_indexes == [120]


def test_select_stable_views_keeps_first_after_change_and_rejects_bad_candidates() -> None:
    frames = [
        _notation_frame(0),
        _notation_frame(0),
        _motion_frame(),
        _notation_frame(20),
        np.full((96, 128, 3), 245, dtype=np.uint8),
    ]
    candidates = [
        CandidateFrame(0, 0.0, change_score=0.0, stability_score=0.98, notes=["initial"]),
        CandidateFrame(1, 0.5, change_score=0.01, stability_score=0.99, notes=["static-hold"]),
        CandidateFrame(2, 1.0, change_score=0.7, stability_score=0.2, notes=["motion-heavy"]),
        CandidateFrame(3, 1.5, change_score=0.62, stability_score=0.96, notes=["change-adjacent"]),
        CandidateFrame(4, 2.0, change_score=0.02, stability_score=0.95, notes=[]),
    ]

    selection = select_stable_views(frames, candidates, min_content_score=0.02)

    assert [view.frame_index for view in selection.accepted] == [0, 3]
    assert all(isinstance(view, StableView) for view in selection.accepted)
    assert selection.accepted[0].id == "view_001"
    assert selection.accepted[1].id == "view_002"
    rejected_notes = {item.frame_index: item.notes for item in selection.rejected}
    assert "static-hold-collapsed" in rejected_notes[1]
    assert "motion-heavy" in rejected_notes[2]
    assert "too-small-content" in rejected_notes[4]


def test_select_stable_views_rejects_blurred_notation_even_without_upstream_motion_note() -> None:
    frames = [_notation_frame(0), _blurred_notation_frame(), _notation_frame(18)]
    candidates = [
        CandidateFrame(0, 0.0, change_score=0.0, stability_score=0.98, notes=["initial"]),
        CandidateFrame(1, 0.5, change_score=0.08, stability_score=0.92, notes=["change-adjacent"]),
        CandidateFrame(2, 1.0, change_score=0.4, stability_score=0.97, notes=["change-adjacent"]),
    ]

    selection = select_stable_views(frames, candidates, min_content_score=0.02)

    assert [view.frame_index for view in selection.accepted] == [0, 2]
    rejected_notes = {item.frame_index: item.notes for item in selection.rejected}
    assert "motion-blur" in rejected_notes[1]


def test_select_stable_views_accepts_high_resolution_antialiased_sheet_music() -> None:
    frames = [_high_resolution_real_style_frame(0), _high_resolution_real_style_frame(8)]
    candidates = [
        CandidateFrame(0, 0.0, change_score=0.0, stability_score=0.98, notes=["initial"]),
        CandidateFrame(1, 2.0, change_score=0.02, stability_score=0.98, notes=["periodic-sample"]),
    ]

    selection = select_stable_views(frames, candidates, min_content_score=0.02)

    assert [view.frame_index for view in selection.accepted] == [0, 1]
