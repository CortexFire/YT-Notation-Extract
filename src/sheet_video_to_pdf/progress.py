from __future__ import annotations

from pathlib import Path
import sys
import threading
import time
from typing import Callable, TextIO

from .models import AppConfig


PipelineFunc = Callable[[AppConfig], Path]
ClockFunc = Callable[[], float]


def run_with_elapsed_tracker(
    pipeline: PipelineFunc,
    config: AppConfig,
    *,
    interval_seconds: float = 1.0,
    clock: ClockFunc = time.monotonic,
    stream: TextIO | None = None,
) -> Path:
    if stream is None:
        stream = sys.stdout

    finished = threading.Event()
    result: list[Path] = []
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            result.append(pipeline(config))
        except BaseException as exc:
            errors.append(exc)
        finally:
            finished.set()

    start_time = clock()
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    _write_elapsed_update(0.0, stream)
    while not finished.wait(max(interval_seconds, 0.01)):
        _write_elapsed_update(clock() - start_time, stream)

    thread.join()
    elapsed_seconds = clock() - start_time
    _write_elapsed_update(elapsed_seconds, stream)
    print(file=stream)
    print(f"Elapsed time: {_format_elapsed(elapsed_seconds)}", file=stream)

    if errors:
        raise errors[0]
    return result[0]


def _write_elapsed_update(elapsed_seconds: float, stream: TextIO) -> None:
    print(f"\rElapsed: {_format_elapsed(elapsed_seconds)}", end="", file=stream, flush=True)


def _format_elapsed(elapsed_seconds: float) -> str:
    total_seconds = max(0, int(elapsed_seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"
