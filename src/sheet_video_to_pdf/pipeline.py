from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import tempfile
from typing import Iterator

import cv2
import numpy as np
from PIL import Image

from .cadence import determine_adaptive_cadence_from_prepared
from .duplicates import apply_duplicate_policy, flag_duplicate_regions
from .errors import NoNotationError, VideoReadError
from .manifest import write_manifest
from .models import AppConfig, ExtractedRegion, RunManifest, StableView, StitchedPage
from .sampling import analyze_sampled_frames, read_sampled_frames_by_index
from .output import (
    ArtifactWriter,
    OutputPaths,
    generate_pdf_from_page_images,
    prepare_output_dirs,
)
from .pagination import paginate_strips
from .regions import detect_notation_region
from .stable_views import (
    preselect_stable_candidates_from_prepared,
    select_stable_views_from_frame_map,
)
from .stitching import stitch_regions
from .video import validate_mp4


def run_pipeline(config: AppConfig) -> Path:
    metadata = validate_mp4(config.input_video)
    output_paths = prepare_output_dirs(config)

    with _active_output_paths(config, output_paths) as active_output_paths:
        writer = ArtifactWriter(active_output_paths, config.jpeg_quality)

        sample_analysis = analyze_sampled_frames(config.input_video, metadata.frame_rate)
        cadence = determine_adaptive_cadence_from_prepared(
            sample_analysis.prepared_frames,
            fps=sample_analysis.sampled_fps,
        )
        stable_preselection = preselect_stable_candidates_from_prepared(
            sample_analysis.prepared_frames,
            cadence.candidates,
        )
        candidate_sample_indexes = {
            candidate.frame_index
            for candidate in stable_preselection.candidates
        }
        frames_by_sample_index = read_sampled_frames_by_index(
            config.input_video,
            sample_analysis.refs,
            candidate_sample_indexes,
        )
        source_frame_indexes = {
            ref.sample_index: ref.source_frame_index
            for ref in sample_analysis.refs
        }
        stable_selection = select_stable_views_from_frame_map(
            frames_by_sample_index,
            stable_preselection.candidates,
            source_frame_indexes=source_frame_indexes,
        )
        stable_views = _write_stable_views(
            stable_selection.accepted,
            frames_by_sample_index,
            writer,
            config,
        )
        if not stable_views:
            raise NoNotationError("No stable sheet music views were detected")

        regions, region_images = _extract_regions(stable_views, frames_by_sample_index, writer, config)
        if not regions:
            raise NoNotationError("No reconstructable notation regions were detected")

        duplicate_flags = flag_duplicate_regions(regions, images_by_region_id=region_images)
        regions = [
            replace(region, duplicate_flags=flags)
            for region, flags in zip(regions, duplicate_flags)
        ]

        stitching_regions = apply_duplicate_policy(regions, config.duplicate_policy)
        stitch_result = stitch_regions(stitching_regions, region_images)
        pages = paginate_strips(stitch_result.strips, config)
        if not pages:
            raise NoNotationError("No stitched pages were produced")

        stitched_pages: list[StitchedPage] = []
        page_image_paths: list[Path] = []
        for page in pages:
            page_path = writer.write_stitched_page_image(
                _gray_to_pil(page.image),
            )
            page_image_paths.append(page_path)
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

        if config.output_debug_files:
            manifest = RunManifest(
                video=metadata,
                cadence_decisions=cadence.decisions,
                stable_views=stable_views,
                extracted_regions=regions,
                stitched_pages=stitched_pages,
                warnings=[
                    *(f"candidate {item.frame_index}: {', '.join(item.notes)}" for item in cadence.rejected_candidates),
                    *(f"stable candidate {item.frame_index}: {', '.join(item.notes)}" for item in stable_preselection.rejected),
                    *(f"stable candidate {item.frame_index}: {', '.join(item.notes)}" for item in stable_selection.rejected),
                    *stitch_result.warnings,
                ],
            )
            write_manifest(manifest, output_paths.output_dir)
        return generate_pdf_from_page_images(page_image_paths, config.output_pdf, config.pdf_dpi)


@contextmanager
def _active_output_paths(config: AppConfig, output_paths: OutputPaths) -> Iterator[OutputPaths]:
    if config.output_debug_files:
        yield output_paths
        return

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        yield OutputPaths(
            output_dir=root,
            stable_views_dir=root / "stable_views",
            extracted_regions_dir=root / "extracted_regions",
            stitched_pages_dir=root / "stitched_pages",
        )


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
    analysis = analyze_sampled_frames(path, source_fps, target_fps)
    frames_by_sample_index = read_sampled_frames_by_index(
        path,
        analysis.refs,
        (ref.sample_index for ref in analysis.refs),
    )
    return [
        frames_by_sample_index[ref.sample_index]
        for ref in analysis.refs
    ], analysis.sampled_fps


def _write_stable_views(
    stable_views: list[StableView],
    frames: dict[int, np.ndarray],
    writer: ArtifactWriter,
    config: AppConfig,
) -> list[StableView]:
    written: list[StableView] = []
    for stable_view in stable_views:
        frame_path = None
        if config.output_debug_files and config.generate_review_assets:
            frame_path = writer.write_stable_view_image(
                _bgr_to_pil(frames[stable_view.frame_index]),
            )
        written.append(replace(stable_view, frame_path=frame_path))
    return written


def _extract_regions(
    stable_views: list[StableView],
    frames: dict[int, np.ndarray],
    writer: ArtifactWriter,
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
        if config.output_debug_files and config.generate_review_assets:
            region_path = writer.write_region_image(
                _bgr_to_pil(crop),
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


def _should_write_review_assets(config: AppConfig) -> bool:
    return config.generate_review_assets and config.output_debug_files
