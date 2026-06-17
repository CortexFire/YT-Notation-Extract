from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class PagePreset(str, Enum):
    LETTER = "letter"


class PageOrientation(str, Enum):
    PORTRAIT = "portrait"
    LANDSCAPE = "landscape"


class DuplicatePolicy(str, Enum):
    FLAG = "flag"
    FLAG_AND_INCLUDE = "flag-and-include"
    FLAG_AND_SUPPRESS_OVERLAP = "flag-and-suppress-overlap"


class RegionKind(str, Enum):
    PARTIAL_VIEW = "partial_view"
    COMPLETE_PAGE = "complete_page"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class AppConfig:
    input_video: Path = Path("input/video.mp4")
    output_pdf: Path = Path("output/sheet_music.pdf")
    output_dir: Path = Path("output")
    page_preset: PagePreset = PagePreset.LETTER
    page_orientation: PageOrientation = PageOrientation.PORTRAIT
    page_margin_inches: float = 0.35
    target_systems_per_page: int | str = "auto"
    duplicate_policy: DuplicatePolicy = DuplicatePolicy.FLAG
    generate_review_assets: bool = True
    output_debug_files: bool = True
    jpeg_quality: int = 92
    pdf_dpi: int = 200
    clean_output: bool = True


@dataclass(frozen=True)
class BoundingBox:
    x: int
    y: int
    width: int
    height: int

    def to_tuple(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)


@dataclass(frozen=True)
class VideoMetadata:
    path: Path
    duration_seconds: float
    frame_rate: float
    frame_count: int
    width: int
    height: int


@dataclass(frozen=True)
class CadenceDecision:
    start_seconds: float
    end_seconds: float
    interval_seconds: float
    reason: str
    average_change: float | None = None


@dataclass(frozen=True)
class StableView:
    id: str
    timestamp_seconds: float
    frame_index: int
    frame_path: Path | None
    stability_score: float
    source_frame_index: int | None = None
    rejection_notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DuplicateFlags:
    exact_duplicate: bool = False
    near_duplicate: bool = False
    repeat_candidate: bool = False
    matched_region_id: str | None = None
    similarity: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(self)


@dataclass(frozen=True)
class ExtractedRegion:
    id: str
    stable_view_id: str
    source_timestamp_seconds: float
    image_path: Path | None
    bounding_box: BoundingBox
    confidence: float
    kind: RegionKind
    fallback_used: bool = False
    duplicate_flags: DuplicateFlags = field(default_factory=DuplicateFlags)
    alignment_confidence: float | None = None
    stitch_placement: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class StitchPlacement:
    strip_id: str
    y_offset: int
    overlap_pixels: int
    alignment_confidence: float
    decision: str


@dataclass(frozen=True)
class StitchedPage:
    id: str
    image_path: Path
    included_region_ids: list[str]
    source_start_seconds: float
    source_end_seconds: float
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RunManifest:
    video: VideoMetadata
    cadence_decisions: list[CadenceDecision] = field(default_factory=list)
    stable_views: list[StableView] = field(default_factory=list)
    extracted_regions: list[ExtractedRegion] = field(default_factory=list)
    stitched_pages: list[StitchedPage] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(self)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: _json_ready(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value
