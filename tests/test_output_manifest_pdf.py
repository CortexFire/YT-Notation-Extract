from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from sheet_video_to_pdf.errors import NoNotationError
from sheet_video_to_pdf.manifest import write_manifest
from sheet_video_to_pdf.models import (
    AppConfig,
    BoundingBox,
    CadenceDecision,
    ExtractedRegion,
    RegionKind,
    RunManifest,
    StableView,
    StitchedPage,
    VideoMetadata,
)
from sheet_video_to_pdf.output import (
    generate_pdf_from_pages,
    prepare_output_dirs,
    write_region_image,
    write_stable_view_image,
    write_stitched_page_image,
)


def test_prepare_output_dirs_creates_expected_subdirectories(tmp_path: Path) -> None:
    config = AppConfig(output_dir=tmp_path / "out", output_pdf=tmp_path / "out" / "sheet_music.pdf")

    paths = prepare_output_dirs(config)

    assert paths.output_dir == tmp_path / "out"
    assert paths.stable_views_dir.is_dir()
    assert paths.extracted_regions_dir.is_dir()
    assert paths.stitched_pages_dir.is_dir()


def test_prepare_output_dirs_safely_cleans_only_generated_locations(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    stable_dir = output_dir / "stable_views"
    regions_dir = output_dir / "extracted_regions"
    pages_dir = output_dir / "stitched_pages"
    stable_dir.mkdir(parents=True)
    regions_dir.mkdir()
    pages_dir.mkdir()
    stale_view = stable_dir / "view_001.jpg"
    stale_region = regions_dir / "region_001.jpg"
    stale_page = pages_dir / "page_001.jpg"
    manifest = output_dir / "manifest.json"
    pdf = output_dir / "sheet_music.pdf"
    outside_file = output_dir / "keep_me.txt"
    nested_outside = tmp_path / "outside.pdf"
    for path in [stale_view, stale_region, stale_page, manifest, pdf, outside_file, nested_outside]:
        path.write_bytes(b"stale")
    config = AppConfig(output_dir=output_dir, output_pdf=pdf, clean_output=True)

    prepare_output_dirs(config)

    assert not stale_view.exists()
    assert not stale_region.exists()
    assert not stale_page.exists()
    assert not manifest.exists()
    assert not pdf.exists()
    assert outside_file.exists()
    assert nested_outside.exists()


def test_writes_sequential_jpegs_to_review_folders(tmp_path: Path) -> None:
    paths = prepare_output_dirs(AppConfig(output_dir=tmp_path / "out", output_pdf=tmp_path / "out" / "sheet_music.pdf"))
    image = Image.new("RGB", (12, 8), "white")

    first_view = write_stable_view_image(image, paths, jpeg_quality=80)
    second_view = write_stable_view_image(image, paths, jpeg_quality=80)
    first_region = write_region_image(image, paths, jpeg_quality=80)
    first_page = write_stitched_page_image(image, paths, jpeg_quality=80)

    assert first_view == paths.stable_views_dir / "view_001.jpg"
    assert second_view == paths.stable_views_dir / "view_002.jpg"
    assert first_region == paths.extracted_regions_dir / "region_001.jpg"
    assert first_page == paths.stitched_pages_dir / "page_001.jpg"
    assert first_view.exists()
    assert second_view.exists()
    assert first_region.exists()
    assert first_page.exists()


def test_write_manifest_serializes_run_manifest_json(tmp_path: Path) -> None:
    manifest = RunManifest(
        video=VideoMetadata(
            path=Path("input/video.mp4"),
            duration_seconds=12.5,
            frame_rate=30.0,
            frame_count=375,
            width=1920,
            height=1080,
        ),
        cadence_decisions=[
            CadenceDecision(start_seconds=0.0, end_seconds=12.5, interval_seconds=0.5, reason="stable")
        ],
        stable_views=[
            StableView(
                id="view-1",
                timestamp_seconds=1.25,
                frame_index=38,
                frame_path=Path("output/stable_views/view_001.jpg"),
                stability_score=0.98,
            )
        ],
        extracted_regions=[
            ExtractedRegion(
                id="region-1",
                stable_view_id="view-1",
                source_timestamp_seconds=1.25,
                image_path=Path("output/extracted_regions/region_001.jpg"),
                bounding_box=BoundingBox(x=1, y=2, width=300, height=120),
                confidence=0.91,
                kind=RegionKind.PARTIAL_VIEW,
            )
        ],
        stitched_pages=[
            StitchedPage(
                id="page-1",
                image_path=Path("output/stitched_pages/page_001.jpg"),
                included_region_ids=["region-1"],
                source_start_seconds=1.25,
                source_end_seconds=1.25,
                warnings=["low whitespace confidence"],
            )
        ],
    )

    manifest_path = write_manifest(manifest, tmp_path / "out")

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_path == tmp_path / "out" / "manifest.json"
    assert data["video"]["path"] == "input/video.mp4"
    assert data["cadence_decisions"][0]["reason"] == "stable"
    assert data["stable_views"][0]["frame_path"] == "output/stable_views/view_001.jpg"
    assert data["extracted_regions"][0]["bounding_box"] == {"x": 1, "y": 2, "width": 300, "height": 120}
    assert data["extracted_regions"][0]["kind"] == "partial_view"
    assert data["stitched_pages"][0]["warnings"] == ["low whitespace confidence"]


def test_generate_pdf_from_stitched_pages_uses_page_images_in_order(tmp_path: Path) -> None:
    paths = prepare_output_dirs(AppConfig(output_dir=tmp_path / "out", output_pdf=tmp_path / "out" / "sheet_music.pdf"))
    write_stitched_page_image(Image.new("RGB", (20, 30), "white"), paths, jpeg_quality=90)
    write_stitched_page_image(Image.new("RGB", (20, 30), "black"), paths, jpeg_quality=90)

    pdf_path = generate_pdf_from_pages(paths, tmp_path / "out" / "sheet_music.pdf", pdf_dpi=200)

    assert pdf_path == tmp_path / "out" / "sheet_music.pdf"
    pdf_bytes = pdf_path.read_bytes()
    assert pdf_bytes.startswith(b"%PDF")
    assert pdf_bytes.count(b"/Type /Page") >= 2


def test_generate_pdf_from_stitched_pages_fails_when_no_pages_exist(tmp_path: Path) -> None:
    paths = prepare_output_dirs(AppConfig(output_dir=tmp_path / "out", output_pdf=tmp_path / "out" / "sheet_music.pdf"))

    with pytest.raises(NoNotationError, match="No stitched page images"):
        generate_pdf_from_pages(paths, tmp_path / "out" / "sheet_music.pdf", pdf_dpi=200)
