from __future__ import annotations

import cv2
import numpy as np

from sheet_video_to_pdf.sampling import (
    SampledFrameRef,
    analyze_sampled_frames,
    read_sampled_frames_by_index,
)
from tests.fixtures.synthetic_video import create_moving_sheet_music_video


class _FakeCapture:
    def __init__(self, frame_count: int, frame_size: tuple[int, int] = (12, 8)) -> None:
        self.frame_count = frame_count
        self.frame_size = frame_size
        self.index = 0
        self.read_indexes: list[int] = []
        self.grab_indexes: list[int] = []
        self.set_positions: list[int] = []
        self.released = False

    def isOpened(self) -> bool:
        return True

    def read(self):
        if self.index >= self.frame_count:
            return False, None
        frame_index = self.index
        self.read_indexes.append(frame_index)
        self.index += 1
        width, height = self.frame_size
        return True, np.full((height, width, 3), frame_index % 255, dtype=np.uint8)

    def grab(self) -> bool:
        if self.index >= self.frame_count:
            return False
        self.grab_indexes.append(self.index)
        self.index += 1
        return True

    def set(self, prop_id, value) -> bool:
        if prop_id != cv2.CAP_PROP_POS_FRAMES:
            return False
        self.index = int(value)
        self.set_positions.append(self.index)
        return True

    def release(self) -> None:
        self.released = True


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


def test_analyze_sampled_frames_grabs_skipped_frames_without_decoding(monkeypatch):
    capture = _FakeCapture(frame_count=9)
    monkeypatch.setattr(cv2, "VideoCapture", lambda _path: capture)

    analysis = analyze_sampled_frames("input.mp4", source_fps=6.0, target_fps=2.0)

    assert [ref.source_frame_index for ref in analysis.refs] == [0, 3, 6]
    assert capture.read_indexes == [0, 3, 6]
    assert capture.grab_indexes == [1, 2, 4, 5, 7, 8]
    assert capture.released is True


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


def test_read_sampled_frames_by_index_seeks_for_sparse_requests(monkeypatch):
    capture = _FakeCapture(frame_count=100)
    monkeypatch.setattr(cv2, "VideoCapture", lambda _path: capture)
    refs = [
        SampledFrameRef(sample_index=0, source_frame_index=0, timestamp_seconds=0.0),
        SampledFrameRef(sample_index=1, source_frame_index=30, timestamp_seconds=1.0),
        SampledFrameRef(sample_index=2, source_frame_index=60, timestamp_seconds=2.0),
        SampledFrameRef(sample_index=3, source_frame_index=90, timestamp_seconds=3.0),
    ]

    frames = read_sampled_frames_by_index("input.mp4", refs, {1, 3})

    assert sorted(frames) == [1, 3]
    assert capture.set_positions == [30, 90]
    assert capture.read_indexes == [30, 90]
