from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import img2pdf
from PIL import Image

from .errors import NoNotationError
from .models import AppConfig

STABLE_VIEWS_DIRNAME = "stable_views"
EXTRACTED_REGIONS_DIRNAME = "extracted_regions"
STITCHED_PAGES_DIRNAME = "stitched_pages"


@dataclass(frozen=True)
class OutputPaths:
    output_dir: Path
    stable_views_dir: Path
    extracted_regions_dir: Path
    stitched_pages_dir: Path


def prepare_output_dirs(config: AppConfig) -> OutputPaths:
    output_dir = Path(config.output_dir)
    paths = OutputPaths(
        output_dir=output_dir,
        stable_views_dir=output_dir / STABLE_VIEWS_DIRNAME,
        extracted_regions_dir=output_dir / EXTRACTED_REGIONS_DIRNAME,
        stitched_pages_dir=output_dir / STITCHED_PAGES_DIRNAME,
    )

    if config.clean_output:
        clean_output(config)

    paths.stable_views_dir.mkdir(parents=True, exist_ok=True)
    paths.extracted_regions_dir.mkdir(parents=True, exist_ok=True)
    paths.stitched_pages_dir.mkdir(parents=True, exist_ok=True)
    return paths


def clean_output(config: AppConfig) -> None:
    output_dir = Path(config.output_dir)
    for dirname in (STABLE_VIEWS_DIRNAME, EXTRACTED_REGIONS_DIRNAME, STITCHED_PAGES_DIRNAME):
        _unlink_files_inside(output_dir / dirname)

    _unlink_if_file(output_dir / "manifest.json")

    output_pdf = Path(config.output_pdf)
    if _is_relative_to(output_pdf, output_dir):
        _unlink_if_file(output_pdf)


def write_stable_view_image(image: Image.Image, paths: OutputPaths, jpeg_quality: int) -> Path:
    return _write_numbered_jpeg(image, paths.stable_views_dir, "view", jpeg_quality)


def write_region_image(image: Image.Image, paths: OutputPaths, jpeg_quality: int) -> Path:
    return _write_numbered_jpeg(image, paths.extracted_regions_dir, "region", jpeg_quality)


def write_stitched_page_image(image: Image.Image, paths: OutputPaths, jpeg_quality: int) -> Path:
    return _write_numbered_jpeg(image, paths.stitched_pages_dir, "page", jpeg_quality)


def generate_pdf_from_pages(paths: OutputPaths, output_pdf: str | Path, pdf_dpi: int) -> Path:
    page_images = _numbered_jpegs(paths.stitched_pages_dir, "page")
    if not page_images:
        raise NoNotationError("No stitched page images exist; cannot generate PDF")

    output_path = Path(output_pdf)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    layout_fun = img2pdf.get_fixed_dpi_layout_fun((pdf_dpi, pdf_dpi))
    with output_path.open("wb") as handle:
        handle.write(img2pdf.convert([str(path) for path in page_images], layout_fun=layout_fun))
    return output_path


def _write_numbered_jpeg(image: Image.Image, directory: Path, prefix: str, jpeg_quality: int) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    next_index = _next_index(directory, prefix)
    path = directory / f"{prefix}_{next_index:03d}.jpg"
    image_to_save = image.convert("RGB") if image.mode != "RGB" else image
    image_to_save.save(path, format="JPEG", quality=jpeg_quality)
    return path


def _next_index(directory: Path, prefix: str) -> int:
    existing = _numbered_jpegs(directory, prefix)
    if not existing:
        return 1
    return max(_numbered_index(path, prefix) for path in existing) + 1


def _numbered_jpegs(directory: Path, prefix: str) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        (
            path
            for path in directory.glob(f"{prefix}_*.jpg")
            if _numbered_index(path, prefix) is not None
        ),
        key=lambda path: _numbered_index(path, prefix) or 0,
    )


def _numbered_index(path: Path, prefix: str) -> int | None:
    match = re.fullmatch(rf"{re.escape(prefix)}_(\d{{3}})\.jpg", path.name)
    return int(match.group(1)) if match else None


def _unlink_files_inside(directory: Path) -> None:
    if not directory.exists():
        return

    for path in sorted(directory.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_file() or path.is_symlink():
            path.unlink()
        elif path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass


def _unlink_if_file(path: Path) -> None:
    if path.is_file() or path.is_symlink():
        path.unlink()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(parent.resolve(strict=False))
    except ValueError:
        return False
    return True
