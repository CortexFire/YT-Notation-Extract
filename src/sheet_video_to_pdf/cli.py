from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from .config import build_config
from .errors import SheetVideoToPdfError
from .models import AppConfig, DuplicatePolicy, PageOrientation, PagePreset
from .pipeline import run_pipeline
from .progress import ElapsedTimeTracker


@dataclass(frozen=True)
class ParsedCli:
    config_path: Path | None
    overrides: dict[str, object]


def parse_args(argv: Sequence[str] | None = None) -> ParsedCli:
    parser = argparse.ArgumentParser(
        prog="sheet-video-to-pdf",
        description="Reconstruct sheet music pages from a local MP4 and write an image-based PDF.",
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--input", dest="input_video")
    parser.add_argument("--output", dest="output_pdf")
    parser.add_argument("--output-dir")
    parser.add_argument("--page-preset", choices=[item.value for item in PagePreset])
    parser.add_argument("--page-orientation", choices=[item.value for item in PageOrientation])
    parser.add_argument("--page-margin-inches", type=float)
    parser.add_argument("--target-systems-per-page", type=_target_systems)
    parser.add_argument("--duplicate-policy", choices=[item.value for item in DuplicatePolicy])
    parser.add_argument("--jpeg-quality", type=int)
    parser.add_argument("--pdf-dpi", type=int)
    parser.add_argument("--no-review-assets", action="store_true")
    parser.add_argument("--no-clean-output", action="store_true")
    parser.add_argument("--no-debug-files", action="store_true")

    namespace = parser.parse_args(argv)
    raw = vars(namespace)
    config_path = raw.pop("config")
    no_review_assets = raw.pop("no_review_assets")
    no_clean_output = raw.pop("no_clean_output")
    no_debug_files = raw.pop("no_debug_files")

    overrides = {key: value for key, value in raw.items() if value is not None}
    if no_review_assets:
        overrides["generate_review_assets"] = False
    if no_clean_output:
        overrides["clean_output"] = False
    if no_debug_files:
        overrides["output_debug_files"] = False

    return ParsedCli(config_path=config_path, overrides=overrides)


def run_cli(
    argv: Sequence[str] | None = None,
    pipeline: Callable[[AppConfig], Path] = run_pipeline,
    *,
    progress_interval: float = 1.0,
) -> int:
    try:
        parsed = parse_args(argv)
        config = build_config(parsed.config_path, parsed.overrides)
        with ElapsedTimeTracker(interval=progress_interval):
            pipeline(config)
    except SheetVideoToPdfError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


def _target_systems(value: str) -> int | str:
    if value == "auto":
        return value
    return int(value)
