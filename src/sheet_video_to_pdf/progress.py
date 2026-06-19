from __future__ import annotations

import sys
import threading
import time
from types import TracebackType
from typing import Callable, TextIO, TypeVar


TConfig = TypeVar("TConfig")
TResult = TypeVar("TResult")


class ElapsedTimeTracker:
    def __init__(
        self,
        *,
        interval: float = 1.0,
        stream: TextIO | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if interval <= 0:
            raise ValueError("interval must be greater than zero")

        self.interval = interval
        self.stream = stream
        self.clock = clock or time.monotonic
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._started_at: float | None = None

    def __enter__(self) -> ElapsedTimeTracker:
        self._started_at = self.clock()
        self._write_elapsed()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        self.stop()
        return False

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        self._write_finished()

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval):
            self._write_elapsed()

    def _write_elapsed(self) -> None:
        self._write(f"Elapsed time: {self.elapsed_text()}", end="")

    def _write_finished(self) -> None:
        self._write(f"Finished in {self.elapsed_text()}", end="\n")

    def _write(self, message: str, *, end: str) -> None:
        stream = self.stream or sys.stdout
        with self._lock:
            print(f"\r{message}", end=end, file=stream)
            stream.flush()

    def elapsed_text(self) -> str:
        if self._started_at is None:
            elapsed_seconds = 0
        else:
            elapsed_seconds = max(0, int(self.clock() - self._started_at))
        return format_elapsed(elapsed_seconds)


def format_elapsed(total_seconds: int) -> str:
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def run_with_elapsed_tracker(
    operation: Callable[[TConfig], TResult],
    config: TConfig,
    *,
    interval_seconds: float = 1.0,
    stream: TextIO | None = None,
) -> TResult:
    with ElapsedTimeTracker(interval=interval_seconds, stream=stream):
        return operation(config)
