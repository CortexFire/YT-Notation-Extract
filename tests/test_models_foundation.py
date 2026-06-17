from pathlib import Path

from sheet_video_to_pdf.models import (
    BoundingBox,
    DuplicateFlags,
    RegionKind,
    RunManifest,
    StableView,
    VideoMetadata,
)


def test_manifest_serializes_nested_records_to_json_ready_dicts():
    manifest = RunManifest(
        video=VideoMetadata(
            path=Path("input/video.mp4"),
            duration_seconds=12.5,
            frame_rate=30.0,
            frame_count=375,
            width=1920,
            height=1080,
        ),
        stable_views=[
            StableView(
                id="view_001",
                timestamp_seconds=1.2,
                frame_index=36,
                frame_path=Path("output/stable_views/view_001.jpg"),
                stability_score=0.96,
                rejection_notes=[],
            )
        ],
    )

    data = manifest.to_dict()

    assert data["video"]["path"] == "input/video.mp4"
    assert data["stable_views"][0]["id"] == "view_001"
    assert data["stable_views"][0]["frame_path"] == "output/stable_views/view_001.jpg"
    assert data["cadence_decisions"] == []
    assert data["warnings"] == []


def test_region_and_duplicate_models_capture_spec_fields():
    flags = DuplicateFlags(
        exact_duplicate=False,
        near_duplicate=True,
        repeat_candidate=True,
        matched_region_id="region_001",
        similarity=0.91,
    )

    region = flags.to_dict()

    assert region["near_duplicate"] is True
    assert region["repeat_candidate"] is True
    assert region["matched_region_id"] == "region_001"
    assert BoundingBox(1, 2, 30, 40).to_tuple() == (1, 2, 30, 40)
    assert RegionKind.PARTIAL_VIEW.value == "partial_view"
