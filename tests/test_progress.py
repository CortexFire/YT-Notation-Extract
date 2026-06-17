from io import StringIO

from sheet_video_to_pdf.progress import ElapsedTimeTracker


def test_elapsed_tracker_updates_same_console_line_until_finished():
    current_time = 0.0
    stream = StringIO()

    def clock():
        return current_time

    with ElapsedTimeTracker(interval=99, stream=stream, clock=clock) as tracker:
        current_time = 1.0
        tracker._write_elapsed()
        current_time = 2.0
        tracker._write_elapsed()

    output = stream.getvalue()
    assert output.count("\n") == 1
    assert output.count("\r") == 4
    assert "\rElapsed time: 0s" in output
    assert "\rElapsed time: 1s" in output
    assert "\rElapsed time: 2s" in output
    assert "\rFinished in 2s\n" in output
