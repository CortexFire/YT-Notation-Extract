from pathlib import Path

from sheet_video_to_pdf.errors import ConfigError
from sheet_video_to_pdf.interactive import run_interactive


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
        input_func=_answers(str(video_path), "y"),
        pause_func=lambda: pauses.append(True),
    )

    assert exit_code == 0
    assert seen["config"].input_video == video_path
    assert seen["config"].output_pdf == tmp_path / "lesson_sheet_music.pdf"
    assert seen["config"].output_dir == tmp_path / "lesson_sheet_music_assets"
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
        input_func=_answers(str(video_path), "n", str(output_pdf), str(output_dir), prompts=prompts),
        pause_func=lambda: None,
    )

    assert exit_code == 0
    assert seen["config"].output_pdf == output_pdf
    assert seen["config"].output_dir == output_dir
    assert prompts == [
        "MP4 video path: ",
        "Place outputs next to the MP4? [y/n]: ",
        "Output PDF path: ",
        "Review assets folder: ",
    ]


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
        input_func=_answers(str(video_path), "maybe", "y", prompts=prompts),
        pause_func=lambda: None,
    )

    assert exit_code == 0
    assert seen["config"].output_pdf == tmp_path / "lesson_sheet_music.pdf"
    assert seen["config"].output_dir == tmp_path / "lesson_sheet_music_assets"
    assert prompts == [
        "MP4 video path: ",
        "Place outputs next to the MP4? [y/n]: ",
        "Please enter y or n: ",
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
        input_func=_answers(str(video_path), "y"),
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
