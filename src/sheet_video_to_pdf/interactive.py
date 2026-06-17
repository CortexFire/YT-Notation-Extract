from __future__ import annotations

from pathlib import Path
from typing import Callable

from .errors import SheetVideoToPdfError
from .models import AppConfig
from .pipeline import run_pipeline


InputFunc = Callable[[str], str]
PauseFunc = Callable[[], None]
PipelineFunc = Callable[[AppConfig], Path]


def run_interactive(
    *,
    pipeline: PipelineFunc = run_pipeline,
    input_func: InputFunc = input,
    pause_func: PauseFunc | None = None,
) -> int:
    if pause_func is None:
        pause_func = _pause

    try:
        return _run_prompt_flow(pipeline, input_func)
    finally:
        pause_func()


def _run_prompt_flow(pipeline: PipelineFunc, input_func: InputFunc) -> int:
    print("Sheet Video to PDF")
    print("==================")
    print("Convert an MP4 of sheet music into a reconstructed PDF.")
    print()

    input_video = _prompt_path(input_func, "MP4 video path: ")
    if input_video is None:
        print("No input video was provided.")
        return 2
    if not input_video.exists():
        print(f"Input video does not exist: {input_video}")
        return 2
    if not input_video.is_file():
        print(f"Input video is not a file: {input_video}")
        return 2
    if input_video.suffix.lower() != ".mp4":
        print(f"Input video must use the .mp4 extension: {input_video}")
        return 2

    default_output_pdf = input_video.with_name(f"{input_video.stem}_sheet_music.pdf")
    default_output_dir = input_video.with_name(f"{input_video.stem}_sheet_music_assets")

    output_pdf = _prompt_path(
        input_func,
        f"Output PDF path [{default_output_pdf}]: ",
        default=default_output_pdf,
    )
    output_dir = _prompt_path(
        input_func,
        f"Review assets folder [{default_output_dir}]: ",
        default=default_output_dir,
    )

    config = AppConfig(
        input_video=input_video,
        output_pdf=output_pdf or default_output_pdf,
        output_dir=output_dir or default_output_dir,
    )

    print()
    print("Working... this can take a little while for longer videos.")
    try:
        pdf_path = pipeline(config)
    except SheetVideoToPdfError as exc:
        print()
        print(f"Could not create PDF: {exc}")
        return 2

    print()
    print("Done!")
    print(f"PDF: {pdf_path}")
    print(f"Review assets: {config.output_dir}")
    return 0


def _prompt_path(input_func: InputFunc, prompt: str, *, default: Path | None = None) -> Path | None:
    raw_value = input_func(prompt).strip()
    if not raw_value:
        return default
    return Path(_strip_wrapping_quotes(raw_value)).expanduser()


def _strip_wrapping_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _pause() -> None:
    input("Press Enter to close...")
