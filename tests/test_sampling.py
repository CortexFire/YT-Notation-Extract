from __future__ import annotations

import cv2

from sheet_video_to_pdf.sampling import analyze_sampled_frames, read_sampled_frames_by_index
from tests.fixtures.synthetic_video import create_moving_sheet_music_video


def test_analyze_sampled_frames_records_refs_and_prepared_images(tmp_path):
    video_path = create_moving_sheet_music_video(
        tmp_path / "sampled.mp4",
        positions=(0, 4),
        hold_frames=12,
        transition_frames=0,
        fps=6.0,
        frame_size=(120, 90),
    )

    analysis = analyze_sampled_frames(video_path, source_fps=6.0, target_fps=2.0)

    assert analysis.frame_step == 3
    assert analysis.sampled_fps == 2.0
    assert len(analysis.refs) == len(analysis.prepared_frames)
    assert [ref.sample_index for ref in analysis.refs[:3]] == [0, 1, 2]
    assert [ref.source_frame_index for ref in analysis.refs[:3]] == [0, 3, 6]
    assert [ref.timestamp_seconds for ref in analysis.refs[:3]] == [0.0, 0.5, 1.0]
    assert all(frame.ndim == 2 for frame in analysis.prepared_frames)
    assert max(analysis.prepared_frames[0].shape) <= 160


def test_read_sampled_frames_by_index_decodes_only_requested_frames(tmp_path):
    video_path = create_moving_sheet_music_video(
        tmp_path / "selected.mp4",
        positions=(0, 6, 12),
        hold_frames=4,
        transition_frames=0,
        fps=6.0,
        frame_size=(120, 90),
    )
    analysis = analyze_sampled_frames(video_path, source_fps=6.0, target_fps=2.0)

    frames = read_sampled_frames_by_index(video_path, analysis.refs, {1, 3})

    assert sorted(frames) == [1, 3]
    assert all(frame.shape == (90, 120, 3) for frame in frames.values())

    capture = cv2.VideoCapture(str(video_path))
    try:
        decoded = {}
        for source_index in (analysis.refs[1].source_frame_index, analysis.refs[3].source_frame_index):
            capture.set(cv2.CAP_PROP_POS_FRAMES, source_index)
            ok, frame = capture.read()
            assert ok
            decoded[source_index] = frame
    finally:
        capture.release()

    assert frames[1].shape == decoded[analysis.refs[1].source_frame_index].shape
    assert frames[3].shape == decoded[analysis.refs[3].source_frame_index].shape
