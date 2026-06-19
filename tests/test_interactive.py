from pathlib import Path
import time

from sheet_video_to_pdf.errors import ConfigError
from sheet_video_to_pdf.interactive import run_interactive
from sheet_video_to_pdf.models import PageOrientation


def test_interactive_uses_video_folder_defaults(tmp_path, capsys):
    video_path = tmp_path / "lesson.mp4"
    video_path.write_bytes(b"mp4")
    seen = {}
    pauses = []

    def fake_pipeline(config):
        seen["config"] = config
        return config.output_pdf

    exit_code = run_interactive(
        pipeline=fake_pipeline,
        input_func=_answers(str(video_path), "y", "y", "p"),
        pause_func=lambda: pauses.append(True),
    )

    assert exit_code == 0
    assert seen["config"].input_video == video_path
    assert seen["config"].output_pdf == tmp_path / "lesson_sheet_music.pdf"
    assert seen["config"].output_dir == tmp_path / "lesson_sheet_music_assets"
    assert seen["config"].output_debug_files is True
    assert seen["config"].page_orientation is PageOrientation.PORTRAIT
    assert pauses == [True]
    assert "Done!" in capsys.readouterr().out


def test_interactive_accepts_custom_output_locations_after_declining_defaults(tmp_path):
    video_path = tmp_path / "lesson.mp4"
    video_path.write_bytes(b"mp4")
    output_pdf = tmp_path / "custom.pdf"
    output_dir = tmp_path / "custom_assets"
    seen = {}
    prompts = []

    def fake_pipeline(config):
        seen["config"] = config
        return config.output_pdf

    exit_code = run_interactive(
        pipeline=fake_pipeline,
        input_func=_answers(str(video_path), "n", str(output_pdf), "y", str(output_dir), "l", prompts=prompts),
        pause_func=lambda: None,
    )

    assert exit_code == 0
    assert seen["config"].output_pdf == output_pdf
    assert seen["config"].output_dir == output_dir
    assert seen["config"].page_orientation is PageOrientation.LANDSCAPE
    assert prompts == [
        "MP4 video path: ",
        "Place outputs next to the MP4? [y/n]: ",
        "Output PDF path: ",
        "Output debug files? [y/n]: ",
        "Review assets folder: ",
        "PDF orientation [p/l]: ",
    ]


def test_interactive_skips_custom_debug_folder_when_debug_files_disabled(tmp_path):
    video_path = tmp_path / "lesson.mp4"
    video_path.write_bytes(b"mp4")
    output_pdf = tmp_path / "custom.pdf"
    seen = {}
    prompts = []

    def fake_pipeline(config):
        seen["config"] = config
        return config.output_pdf

    exit_code = run_interactive(
        pipeline=fake_pipeline,
        input_func=_answers(str(video_path), "n", str(output_pdf), "n", "p", prompts=prompts),
        pause_func=lambda: None,
    )

    assert exit_code == 0
    assert seen["config"].output_pdf == output_pdf
    assert seen["config"].output_dir == output_pdf.parent
    assert seen["config"].output_debug_files is False
    assert prompts == [
        "MP4 video path: ",
        "Place outputs next to the MP4? [y/n]: ",
        "Output PDF path: ",
        "Output debug files? [y/n]: ",
        "PDF orientation [p/l]: ",
    ]


def test_interactive_can_disable_debug_files(tmp_path):
    video_path = tmp_path / "lesson.mp4"
    video_path.write_bytes(b"mp4")
    seen = {}
    prompts = []

    def fake_pipeline(config):
        seen["config"] = config
        return config.output_pdf

    exit_code = run_interactive(
        pipeline=fake_pipeline,
        input_func=_answers(str(video_path), "y", "n", "p", prompts=prompts),
        pause_func=lambda: None,
    )

    assert exit_code == 0
    assert seen["config"].output_debug_files is False
    assert prompts == [
        "MP4 video path: ",
        "Place outputs next to the MP4? [y/n]: ",
        "Output debug files? [y/n]: ",
        "PDF orientation [p/l]: ",
    ]


def test_interactive_reprompts_for_pdf_orientation(tmp_path):
    video_path = tmp_path / "lesson.mp4"
    video_path.write_bytes(b"mp4")
    seen = {}
    prompts = []

    def fake_pipeline(config):
        seen["config"] = config
        return config.output_pdf

    exit_code = run_interactive(
        pipeline=fake_pipeline,
        input_func=_answers(str(video_path), "y", "n", "sideways", "l", prompts=prompts),
        pause_func=lambda: None,
    )

    assert exit_code == 0
    assert seen["config"].page_orientation is PageOrientation.LANDSCAPE
    assert prompts == [
        "MP4 video path: ",
        "Place outputs next to the MP4? [y/n]: ",
        "Output debug files? [y/n]: ",
        "PDF orientation [p/l]: ",
        "Please enter p or l: ",
    ]


def test_interactive_reports_elapsed_time_while_pipeline_runs(tmp_path, capsys):
    video_path = tmp_path / "lesson.mp4"
    video_path.write_bytes(b"mp4")

    def fake_pipeline(config):
        time.sleep(0.03)
        return config.output_pdf

    exit_code = run_interactive(
        pipeline=fake_pipeline,
        input_func=_answers(str(video_path), "y", "y", "p"),
        pause_func=lambda: None,
        progress_interval=0.005,
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert output.count("Elapsed time:") >= 2
    assert "\rElapsed time:" in output
    assert "Finished in" in output


def test_interactive_reprompts_for_same_folder_confirmation(tmp_path):
    video_path = tmp_path / "lesson.mp4"
    video_path.write_bytes(b"mp4")
    seen = {}
    prompts = []

    def fake_pipeline(config):
        seen["config"] = config
        return config.output_pdf

    exit_code = run_interactive(
        pipeline=fake_pipeline,
        input_func=_answers(str(video_path), "maybe", "y", "n", "p", prompts=prompts),
        pause_func=lambda: None,
    )

    assert exit_code == 0
    assert seen["config"].output_pdf == tmp_path / "lesson_sheet_music.pdf"
    assert seen["config"].output_dir == tmp_path / "lesson_sheet_music_assets"
    assert prompts == [
        "MP4 video path: ",
        "Place outputs next to the MP4? [y/n]: ",
        "Please enter y or n: ",
        "Output debug files? [y/n]: ",
        "PDF orientation [p/l]: ",
    ]


def test_interactive_reports_missing_input_without_running_pipeline(tmp_path, capsys):
    missing_video = tmp_path / "missing.mp4"
    called = False

    def fake_pipeline(_config):
        nonlocal called
        called = True
        return Path("unused.pdf")

    exit_code = run_interactive(
        pipeline=fake_pipeline,
        input_func=_answers(str(missing_video)),
        pause_func=lambda: None,
    )

    assert exit_code == 2
    assert called is False
    assert "Input video does not exist" in capsys.readouterr().out


def test_interactive_reports_pipeline_errors_and_pauses(tmp_path, capsys):
    video_path = tmp_path / "lesson.mp4"
    video_path.write_bytes(b"mp4")
    pauses = []

    def fake_pipeline(_config):
        raise ConfigError("bad config")

    exit_code = run_interactive(
        pipeline=fake_pipeline,
        input_func=_answers(str(video_path), "y", "y", "p"),
        pause_func=lambda: pauses.append(True),
    )

    assert exit_code == 2
    assert pauses == [True]
    assert "bad config" in capsys.readouterr().out


def _answers(*values, prompts=None):
    answers = iter(values)

    def input_func(prompt):
        if prompts is not None:
            prompts.append(prompt)
        return next(answers)

    return input_func
