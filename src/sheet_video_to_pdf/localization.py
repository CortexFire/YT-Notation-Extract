from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .errors import NotationLocalizationError
from .models import BoundingBox, LocalizationSource, NotationLocalization, VideoMetadata
from .preprocess import to_grayscale


@dataclass(frozen=True)
class _PanelCandidate:
    box: BoundingBox
    staff_group_count: int


def localize_notation_area(
    path: str | Path,
    metadata: VideoMetadata,
    *,
    manual_roi: BoundingBox | None = None,
    max_probes: int = 16,
) -> NotationLocalization:
    if max_probes < 1:
        raise ValueError("max_probes must be positive")
    if metadata.frame_count < 1 or metadata.frame_rate <= 0:
        raise NotationLocalizationError("Video metadata is incomplete; cannot probe notation area")

    probe_count = min(max_probes, metadata.frame_count)
    source_indexes = sorted(
        set(
            int(round(value))
            for value in np.linspace(0, metadata.frame_count - 1, probe_count)
        )
    )
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise NotationLocalizationError("OpenCV could not open the video for notation localization")

    frames: list[np.ndarray] = []
    timestamps: list[float] = []
    try:
        for source_index in source_indexes:
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, source_index):
                continue
            ok, frame = capture.read()
            if not ok:
                continue
            frames.append(frame)
            timestamps.append(source_index / metadata.frame_rate)
    finally:
        capture.release()

    if not frames:
        raise NotationLocalizationError("OpenCV could not decode notation probe frames")
    return localize_notation_from_frames(
        frames,
        timestamps_seconds=timestamps,
        manual_roi=manual_roi,
    )


def localize_notation_from_frames(
    frames: Sequence[np.ndarray],
    *,
    timestamps_seconds: Sequence[float] | None = None,
    manual_roi: BoundingBox | None = None,
) -> NotationLocalization:
    if not frames:
        raise NotationLocalizationError("No probe frames were available for notation localization")

    height, width = frames[0].shape[:2]
    if any(frame.shape[:2] != (height, width) for frame in frames):
        raise NotationLocalizationError("Notation probe frames have inconsistent dimensions")

    timestamps = list(timestamps_seconds or [float(index) for index in range(len(frames))])
    if len(timestamps) != len(frames):
        raise ValueError("timestamps_seconds must match the number of frames")

    if manual_roi is not None:
        _validate_box_inside_frame(manual_roi, width=width, height=height)
        support = sum(
            bool(_find_staff_groups(_crop(frame, manual_roi)))
            for frame in frames
        )
        if support == 0:
            raise NotationLocalizationError(
                "The manual notation ROI does not contain recognizable staff lines"
            )
        return NotationLocalization(
            bounding_box=manual_roi,
            confidence=round(support / len(frames), 3),
            source=LocalizationSource.MANUAL,
            support_count=support,
            probe_timestamps_seconds=timestamps,
        )

    detected: list[tuple[int, _PanelCandidate]] = []
    for frame_index, frame in enumerate(frames):
        candidate = _detect_panel(frame)
        if candidate is not None:
            detected.append((frame_index, candidate))

    if not detected:
        raise _automatic_failure()

    clusters: list[list[tuple[int, _PanelCandidate]]] = []
    for item in detected:
        for cluster in clusters:
            if _intersection_over_union(item[1].box, _median_box(cluster)) >= 0.60:
                cluster.append(item)
                break
        else:
            clusters.append([item])

    winner = max(
        clusters,
        key=lambda cluster: (
            len(cluster),
            sum(candidate.staff_group_count for _, candidate in cluster),
        ),
    )
    if len(winner) < 3 or len(winner) <= len(detected) / 2:
        raise _automatic_failure()

    box = _median_box(winner)
    support_ratio = len(winner) / len(detected)
    average_groups = float(np.mean([candidate.staff_group_count for _, candidate in winner]))
    group_score = min(1.0, average_groups / 2.0)
    confidence = min(1.0, 0.65 * support_ratio + 0.35 * group_score)
    return NotationLocalization(
        bounding_box=box,
        confidence=round(confidence, 3),
        source=LocalizationSource.AUTO,
        support_count=len(winner),
        probe_timestamps_seconds=[timestamps[index] for index, _ in winner],
    )


def _detect_panel(frame: np.ndarray) -> _PanelCandidate | None:
    gray = to_grayscale(frame)
    groups, horizontal_mask = _find_staff_groups_with_mask(gray)
    if not groups:
        return None

    staff_centers = [center for group in groups for center in group]
    staff_y = int(round(float(np.median(staff_centers))))
    y0, y1 = _light_panel_row_bounds(gray, staff_y)

    supported_cols = np.where(horizontal_mask[y0:y1].any(axis=0))[0]
    if supported_cols.size == 0:
        return None
    padding_x = max(4, round(gray.shape[1] * 0.02))
    x0 = max(0, int(supported_cols.min()) - padding_x)
    x1 = min(gray.shape[1], int(supported_cols.max()) + padding_x + 1)
    if x1 - x0 < gray.shape[1] * 0.20:
        return None

    groups_in_panel = [
        group
        for group in groups
        if y0 <= float(np.mean(group)) < y1
    ]
    return _PanelCandidate(
        BoundingBox(x0, y0, x1 - x0, y1 - y0),
        staff_group_count=len(groups_in_panel),
    )


def _find_staff_groups(image: np.ndarray) -> list[list[float]]:
    groups, _ = _find_staff_groups_with_mask(to_grayscale(image) if image.ndim == 3 else image)
    return groups


def has_staff_evidence(image: np.ndarray) -> bool:
    return bool(_find_staff_groups(image))


def _find_staff_groups_with_mask(gray: np.ndarray) -> tuple[list[list[float]], np.ndarray]:
    dark = (gray < 170).astype(np.uint8) * 255
    kernel_width = max(12, int(round(gray.shape[1] * 0.05)))
    horizontal = cv2.morphologyEx(
        dark,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1)),
    )
    row_support = (horizontal > 0).mean(axis=1)
    rows = np.where(row_support >= 0.20)[0]
    centers = _run_centers(rows)
    return _regular_staff_groups(centers, gray.shape[0]), horizontal > 0


def _regular_staff_groups(centers: list[float], image_height: int) -> list[list[float]]:
    groups: list[list[float]] = []
    index = 0
    max_spacing = max(8.0, image_height * 0.04)
    while index <= len(centers) - 4:
        best: list[float] | None = None
        for count in (6, 5, 4):
            candidate = centers[index : index + count]
            if len(candidate) != count:
                continue
            gaps = np.diff(candidate)
            median_gap = float(np.median(gaps))
            tolerance = max(1.5, median_gap * 0.30)
            if 2.0 <= median_gap <= max_spacing and bool(np.all(np.abs(gaps - median_gap) <= tolerance)):
                best = candidate
                break
        if best is None:
            index += 1
            continue
        groups.append(best)
        index += len(best)
    return groups


def _run_centers(rows: np.ndarray) -> list[float]:
    if rows.size == 0:
        return []
    runs: list[list[int]] = [[int(rows[0])]]
    for row in rows[1:]:
        if int(row) > runs[-1][-1] + 1:
            runs.append([int(row)])
        else:
            runs[-1].append(int(row))
    return [float(np.mean(run)) for run in runs]


def _light_panel_row_bounds(gray: np.ndarray, staff_y: int) -> tuple[int, int]:
    smoothing_height = max(9, int(round(gray.shape[0] * 0.03)) | 1)
    smoothed = cv2.blur(gray, (1, smoothing_height)).mean(axis=1)
    light_rows = smoothed >= 145
    y0 = staff_y
    while y0 > 0 and light_rows[y0 - 1]:
        y0 -= 1
    y1 = staff_y + 1
    while y1 < gray.shape[0] and light_rows[y1]:
        y1 += 1
    return y0, y1


def _median_box(cluster: Sequence[tuple[int, _PanelCandidate]]) -> BoundingBox:
    boxes = [candidate.box for _, candidate in cluster]
    return BoundingBox(
        x=int(round(float(np.median([box.x for box in boxes])))),
        y=int(round(float(np.median([box.y for box in boxes])))),
        width=int(round(float(np.median([box.width for box in boxes])))),
        height=int(round(float(np.median([box.height for box in boxes])))),
    )


def _intersection_over_union(first: BoundingBox, second: BoundingBox) -> float:
    x0 = max(first.x, second.x)
    y0 = max(first.y, second.y)
    x1 = min(first.x + first.width, second.x + second.width)
    y1 = min(first.y + first.height, second.y + second.height)
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    if intersection == 0:
        return 0.0
    union = first.width * first.height + second.width * second.height - intersection
    return float(intersection / union)


def _validate_box_inside_frame(box: BoundingBox, *, width: int, height: int) -> None:
    if (
        box.x < 0
        or box.y < 0
        or box.width <= 0
        or box.height <= 0
        or box.x + box.width > width
        or box.y + box.height > height
    ):
        raise NotationLocalizationError("The manual notation ROI is outside the video frame")


def _crop(frame: np.ndarray, box: BoundingBox) -> np.ndarray:
    return frame[box.y : box.y + box.height, box.x : box.x + box.width]


def _automatic_failure() -> NotationLocalizationError:
    return NotationLocalizationError(
        "Could not find one consistent notation area with recognizable staff lines; "
        "provide a manual notation ROI"
    )


__all__ = ["has_staff_evidence", "localize_notation_area", "localize_notation_from_frames"]
