from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import tempfile
from typing import Iterator, Sequence

import cv2
import numpy as np
from PIL import Image

from .duplicates import apply_duplicate_policy, flag_duplicate_regions
from .errors import NoNotationError, VideoReadError
from .localization import has_staff_evidence, localize_notation_area
from .manifest import write_manifest
from .models import (
    AppConfig,
    BoundingBox,
    CadenceDecision,
    ExtractedRegion,
    RegionKind,
    RunManifest,
    StableView,
    StitchedPage,
)
from .output import ArtifactWriter, OutputPaths, generate_pdf_from_page_images, prepare_output_dirs
from .pagination import paginate_strips
from .regions import classify_region_kind
from .sampling import SampledFrameRef, analyze_sampled_frames, read_sampled_frames_by_index
from .score_views import ScoreStateGroup, fuse_score_frames, group_score_states_from_prepared
from .stitching import stitch_regions
from .video import validate_mp4


def run_pipeline(config: AppConfig) -> Path:
    metadata = validate_mp4(config.input_video)
    localization = localize_notation_area(
        config.input_video,
        metadata,
        manual_roi=config.notation_roi,
    )
    output_paths = prepare_output_dirs(config)

    with _active_output_paths(config, output_paths) as active_output_paths:
        writer = ArtifactWriter(active_output_paths, config.jpeg_quality)
        _write_localization_preview(config.input_video, localization.bounding_box, writer, config)

        sample_analysis = analyze_sampled_frames(
            config.input_video,
            metadata.frame_rate,
            notation_roi=localization.bounding_box,
        )
        timestamps = [ref.timestamp_seconds for ref in sample_analysis.refs]
        score_states = group_score_states_from_prepared(
            sample_analysis.prepared_frames,
            timestamps_seconds=timestamps,
        )
        requested_indexes = {
            sample_index
            for state in score_states
            for sample_index in _representative_indexes(state.sample_indexes)
        }
        frames_by_sample_index = read_sampled_frames_by_index(
            config.input_video,
            sample_analysis.refs,
            requested_indexes,
        )
        stable_views, regions, region_images, view_warnings = _build_clean_score_views(
            score_states,
            frames_by_sample_index,
            sample_analysis.refs,
            localization.bounding_box,
            writer,
            config,
            localization.confidence,
        )
        if not regions:
            raise NoNotationError("No clean score views with recognizable staff lines were detected")

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
            page_path = writer.write_stitched_page_image(_gray_to_pil(page.image))
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
                notation_localization=localization,
                cadence_decisions=[
                    CadenceDecision(
                        start_seconds=state.start_seconds,
                        end_seconds=state.end_seconds,
                        interval_seconds=1.0 / sample_analysis.sampled_fps,
                        reason="distinct score state",
                    )
                    for state in score_states
                ],
                stable_views=stable_views,
                extracted_regions=regions,
                stitched_pages=stitched_pages,
                warnings=[*view_warnings, *stitch_result.warnings],
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
            localization_dir=root / "localization",
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
    return [frames_by_sample_index[ref.sample_index] for ref in analysis.refs], analysis.sampled_fps


def _build_clean_score_views(
    states: Sequence[ScoreStateGroup],
    frames: dict[int, np.ndarray],
    refs: Sequence[SampledFrameRef],
    notation_roi: BoundingBox,
    writer: ArtifactWriter,
    config: AppConfig,
    localization_confidence: float,
) -> tuple[list[StableView], list[ExtractedRegion], dict[str, np.ndarray], list[str]]:
    refs_by_sample_index = {ref.sample_index: ref for ref in refs}
    stable_views: list[StableView] = []
    regions: list[ExtractedRegion] = []
    images: dict[str, np.ndarray] = {}
    view_warnings: list[str] = []
    for state in states:
        selected_indexes = [
            index
            for index in _representative_indexes(state.sample_indexes)
            if index in frames
        ]
        score_frames: list[np.ndarray] = []
        clean_indexes: list[int] = []
        for sample_index in selected_indexes:
            crop = _crop_to_box(frames[sample_index], notation_roi)
            if has_staff_evidence(crop):
                score_frames.append(crop)
                clean_indexes.append(sample_index)
        if not score_frames:
            view_warnings.append(
                f"ignored non-notation score state {state.start_seconds:.1f}s-{state.end_seconds:.1f}s"
            )
            continue

        fused = fuse_score_frames(
            score_frames,
            start_seconds=state.start_seconds,
            end_seconds=state.end_seconds,
        )
        view_id = f"view_{len(stable_views) + 1:03d}"
        representative_sample_index = clean_indexes[len(clean_indexes) // 2]
        representative_ref = refs_by_sample_index[representative_sample_index]
        frame_path = None
        if _should_write_review_assets(config):
            frame_path = writer.write_stable_view_image(_gray_to_pil(fused.image))
        stable_views.append(
            StableView(
                id=view_id,
                timestamp_seconds=representative_ref.timestamp_seconds,
                frame_index=representative_sample_index,
                frame_path=frame_path,
                stability_score=1.0,
                source_frame_index=representative_ref.source_frame_index,
                source_start_seconds=state.start_seconds,
                source_end_seconds=state.end_seconds,
                source_frame_indexes=[
                    refs_by_sample_index[index].source_frame_index
                    for index in clean_indexes
                ],
                composite_frame_count=fused.source_frame_count,
                cleanup_confidence=fused.cleanup_confidence,
            )
        )

        local_box = BoundingBox(0, 0, fused.image.shape[1], fused.image.shape[0])
        kind = classify_region_kind(fused.image, local_box)
        if kind is RegionKind.UNKNOWN:
            kind = RegionKind.PARTIAL_VIEW
        region_id = f"region_{len(regions) + 1:03d}"
        region_path = None
        if _should_write_review_assets(config):
            region_path = writer.write_region_image(_gray_to_pil(fused.image))
        regions.append(
            ExtractedRegion(
                id=region_id,
                stable_view_id=view_id,
                source_timestamp_seconds=representative_ref.timestamp_seconds,
                image_path=region_path,
                bounding_box=notation_roi,
                confidence=round(min(localization_confidence, fused.cleanup_confidence), 3),
                kind=kind,
            )
        )
        images[region_id] = fused.image
    return stable_views, regions, images, view_warnings


def _representative_indexes(indexes: Sequence[int], *, maximum: int = 9) -> list[int]:
    if len(indexes) <= maximum:
        return list(indexes)
    return sorted(
        set(int(round(value)) for value in np.linspace(indexes[0], indexes[-1], maximum))
    )


def _crop_to_box(frame: np.ndarray, box: BoundingBox) -> np.ndarray:
    return frame[box.y : box.y + box.height, box.x : box.x + box.width]


def _write_localization_preview(
    path: str | Path,
    box: BoundingBox,
    writer: ArtifactWriter,
    config: AppConfig,
) -> None:
    if not _should_write_review_assets(config):
        return
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        return
    preview = frame.copy()
    cv2.rectangle(
        preview,
        (box.x, box.y),
        (box.x + box.width - 1, box.y + box.height - 1),
        (0, 0, 255),
        3,
    )
    writer.write_localization_preview(_bgr_to_pil(preview))


def _bgr_to_pil(frame: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _gray_to_pil(image: np.ndarray) -> Image.Image:
    return Image.fromarray(image.astype(np.uint8), mode="L").convert("RGB")


def _should_write_review_assets(config: AppConfig) -> bool:
    return config.generate_review_assets and config.output_debug_files
