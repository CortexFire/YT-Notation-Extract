from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .cadence import determine_adaptive_cadence
from .duplicates import flag_duplicate_regions
from .errors import NoNotationError, VideoReadError
from .manifest import write_manifest
from .models import AppConfig, ExtractedRegion, RunManifest, StableView, StitchedPage
from .output import (
    generate_pdf_from_pages,
    prepare_output_dirs,
    write_region_image,
    write_stable_view_image,
    write_stitched_page_image,
)
from .pagination import paginate_strips
from .regions import detect_notation_region
from .stable_views import select_stable_views
from .stitching import stitch_regions
from .video import validate_mp4


def run_pipeline(config: AppConfig) -> Path:
    metadata = validate_mp4(config.input_video)
    frames, analysis_fps = _read_sampled_frames(config.input_video, metadata.frame_rate)
    output_paths = prepare_output_dirs(config)

    cadence = determine_adaptive_cadence(frames, fps=analysis_fps)
    stable_selection = select_stable_views(frames, cadence.candidates)
    stable_views = _write_stable_views(
        stable_selection.accepted,
        frames,
        output_paths,
        config,
    )
    if not stable_views:
        raise NoNotationError("No stable sheet music views were detected")

    regions, region_images = _extract_regions(stable_views, frames, output_paths, config)
    if not regions:
        raise NoNotationError("No reconstructable notation regions were detected")

    duplicate_flags = flag_duplicate_regions(regions)
    regions = [
        replace(region, duplicate_flags=flags)
        for region, flags in zip(regions, duplicate_flags)
    ]

    stitch_result = stitch_regions(regions, region_images)
    pages = paginate_strips(stitch_result.strips, config)
    if not pages:
        raise NoNotationError("No stitched pages were produced")

    stitched_pages: list[StitchedPage] = []
    for page in pages:
        page_path = write_stitched_page_image(
            _gray_to_pil(page.image),
            output_paths,
            config.jpeg_quality,
        )
        source_times = [
            region.source_timestamp_seconds
            for region in regions
            if region.id in page.included_region_ids
        ]
        stitched_pages.append(
            StitchedPage(
                id=page.id,
                image_path=page_path,
                included_region_ids=page.included_region_ids,
                source_start_seconds=min(source_times) if source_times else 0.0,
                source_end_seconds=max(source_times) if source_times else 0.0,
                warnings=page.warnings,
            )
        )

    manifest = RunManifest(
        video=metadata,
        cadence_decisions=cadence.decisions,
        stable_views=stable_views,
        extracted_regions=regions,
        stitched_pages=stitched_pages,
        warnings=[
            *(f"candidate {item.frame_index}: {', '.join(item.notes)}" for item in cadence.rejected_candidates),
            *(f"stable candidate {item.frame_index}: {', '.join(item.notes)}" for item in stable_selection.rejected),
            *stitch_result.warnings,
        ],
    )
    write_manifest(manifest, output_paths.output_dir)
    return generate_pdf_from_pages(output_paths, config.output_pdf, config.pdf_dpi)


def _read_all_frames(path: str | Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise VideoReadError(
            f"OpenCV could not open the MP4: {path}. Verify codec and FFmpeg support."
        )
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    if not frames:
        raise VideoReadError(
            f"OpenCV could not decode frames from {path}. Verify codec and FFmpeg support."
        )
    return frames


def _read_sampled_frames(
    path: str | Path,
    source_fps: float,
    *,
    target_fps: float = 2.0,
) -> tuple[list[np.ndarray], float]:
    if source_fps <= 0:
        raise VideoReadError("Video frame rate is unreadable; cannot sample frames")

    frame_step = max(1, int(round(source_fps / target_fps)))
    sampled_fps = source_fps / frame_step
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise VideoReadError(
            f"OpenCV could not open the MP4: {path}. Verify codec and FFmpeg support."
        )

    frames: list[np.ndarray] = []
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % frame_step == 0:
                frames.append(frame)
            frame_index += 1
    finally:
        capture.release()

    if not frames:
        raise VideoReadError(
            f"OpenCV could not decode sampled frames from {path}. Verify codec and FFmpeg support."
        )
    return frames, sampled_fps


def _write_stable_views(
    stable_views: list[StableView],
    frames: list[np.ndarray],
    output_paths,
    config: AppConfig,
) -> list[StableView]:
    written: list[StableView] = []
    for stable_view in stable_views:
        frame_path = None
        if config.generate_review_assets:
            frame_path = write_stable_view_image(
                _bgr_to_pil(frames[stable_view.frame_index]),
                output_paths,
                config.jpeg_quality,
            )
        written.append(replace(stable_view, frame_path=frame_path))
    return written


def _extract_regions(
    stable_views: list[StableView],
    frames: list[np.ndarray],
    output_paths,
    config: AppConfig,
) -> tuple[list[ExtractedRegion], dict[str, np.ndarray]]:
    regions: list[ExtractedRegion] = []
    images: dict[str, np.ndarray] = {}
    for stable_view in stable_views:
        frame = frames[stable_view.frame_index]
        region = detect_notation_region(
            frame,
            region_id=f"region_{len(regions) + 1:03d}",
            stable_view_id=stable_view.id,
            source_timestamp_seconds=stable_view.timestamp_seconds,
        )
        box = region.bounding_box
        crop = frame[box.y : box.y + box.height, box.x : box.x + box.width]
        region_path = None
        if config.generate_review_assets:
            region_path = write_region_image(
                _bgr_to_pil(crop),
                output_paths,
                config.jpeg_quality,
            )
        region = replace(region, image_path=region_path)
        regions.append(region)
        images[region.id] = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return regions, images


def _bgr_to_pil(frame: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _gray_to_pil(image: np.ndarray) -> Image.Image:
    return Image.fromarray(image.astype(np.uint8), mode="L").convert("RGB")
