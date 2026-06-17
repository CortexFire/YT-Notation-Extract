from pathlib import Path
import subprocess
import sys
import time

from sheet_video_to_pdf.cli import parse_args, run_cli
from sheet_video_to_pdf.config import DEFAULT_CONFIG
from sheet_video_to_pdf.errors import ConfigError
from sheet_video_to_pdf.models import DuplicatePolicy
from sheet_video_to_pdf.pipeline import _read_sampled_frames, run_pipeline
from tests.fixtures.synthetic_video import create_moving_sheet_music_video


def test_parse_args_maps_output_options_without_sample_fps():
    parsed = parse_args(
        [
            "--input",
            "input/video.mp4",
            "--output",
            "output/custom.pdf",
            "--output-dir",
            "output",
            "--page-preset",
            "letter",
            "--page-orientation",
            "landscape",
            "--page-margin-inches",
            "0.25",
            "--target-systems-per-page",
            "5",
            "--duplicate-policy",
            "flag-and-include",
            "--no-review-assets",
            "--no-clean-output",
            "--no-debug-files",
        ]
    )

    assert parsed.overrides["input_video"] == "input/video.mp4"
    assert parsed.overrides["output_pdf"] == "output/custom.pdf"
    assert parsed.overrides["page_orientation"] == "landscape"
    assert parsed.overrides["page_margin_inches"] == 0.25
    assert parsed.overrides["target_systems_per_page"] == 5
    assert parsed.overrides["duplicate_policy"] == "flag-and-include"
    assert parsed.overrides["generate_review_assets"] is False
    assert parsed.overrides["clean_output"] is False
    assert parsed.overrides["output_debug_files"] is False
    assert "sample_fps" not in parsed.overrides


def test_run_cli_returns_zero_when_pipeline_succeeds(tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text('{"input_video": "input/video.mp4"}', encoding="utf-8")
    seen = {}

    def fake_pipeline(config):
        seen["config"] = config
        return Path("output/sheet_music.pdf")

    exit_code = run_cli(["--config", str(config_file), "--duplicate-policy", "flag"], fake_pipeline)

    assert exit_code == 0
    assert seen["config"].duplicate_policy is DuplicatePolicy.FLAG


def test_run_cli_reports_elapsed_time_while_pipeline_runs_and_when_complete(tmp_path, capsys):
    config_file = tmp_path / "config.json"
    config_file.write_text('{"input_video": "input/video.mp4"}', encoding="utf-8")

    def fake_pipeline(_config):
        time.sleep(0.03)
        return Path("output/sheet_music.pdf")

    exit_code = run_cli(["--config", str(config_file)], fake_pipeline, progress_interval=0.005)

    output = capsys.readouterr().out
    assert exit_code == 0
    assert output.count("Elapsed time:") >= 2
    assert "Finished in" in output


def test_run_cli_returns_nonzero_for_user_facing_errors(capsys):
    def fake_pipeline(_config):
        raise ConfigError("bad config")

    exit_code = run_cli(["--input", "input/video.mp4"], fake_pipeline)

    assert exit_code == 2
    assert "bad config" in capsys.readouterr().err


def test_main_script_runs_without_installed_package():
    result = subprocess.run(
        [sys.executable, "main.py", "--input", "missing.mp4"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "does not exist" in result.stderr


def test_pipeline_processes_synthetic_video_end_to_end(tmp_path):
    video_path = create_moving_sheet_music_video(
        tmp_path / "input.mp4",
        positions=(0, 12, 24),
        hold_frames=2,
        transition_frames=1,
        fps=5.0,
        frame_size=(160, 120),
    )
    config = DEFAULT_CONFIG.__class__(
        input_video=video_path,
        output_dir=tmp_path / "out",
        output_pdf=tmp_path / "out" / "sheet_music.pdf",
        pdf_dpi=80,
    )

    pdf_path = run_pipeline(config)

    assert pdf_path == config.output_pdf
    assert pdf_path.exists()
    assert (config.output_dir / "manifest.json").exists()
    assert list((config.output_dir / "stable_views").glob("view_*.jpg"))
    assert list((config.output_dir / "extracted_regions").glob("region_*.jpg"))
    assert list((config.output_dir / "stitched_pages").glob("page_*.jpg"))


def test_pipeline_pdf_only_mode_leaves_only_final_pdf(tmp_path):
    video_path = create_moving_sheet_music_video(
        tmp_path / "input.mp4",
        positions=(0, 12, 24),
        hold_frames=2,
        transition_frames=1,
        fps=5.0,
        frame_size=(160, 120),
    )
    output_dir = tmp_path / "out"
    config = DEFAULT_CONFIG.__class__(
        input_video=video_path,
        output_dir=output_dir,
        output_pdf=output_dir / "sheet_music.pdf",
        pdf_dpi=80,
        output_debug_files=False,
    )

    pdf_path = run_pipeline(config)

    assert pdf_path == config.output_pdf
    assert pdf_path.exists()
    assert sorted(path.name for path in output_dir.iterdir()) == ["sheet_music.pdf"]


def test_pipeline_pdf_only_mode_does_not_create_separate_debug_folder(tmp_path):
    video_path = create_moving_sheet_music_video(
        tmp_path / "input.mp4",
        positions=(0, 12, 24),
        hold_frames=2,
        transition_frames=1,
        fps=5.0,
        frame_size=(160, 120),
    )
    debug_dir = tmp_path / "debug"
    output_pdf = tmp_path / "pdfs" / "sheet_music.pdf"
    config = DEFAULT_CONFIG.__class__(
        input_video=video_path,
        output_dir=debug_dir,
        output_pdf=output_pdf,
        pdf_dpi=80,
        output_debug_files=False,
    )

    pdf_path = run_pipeline(config)

    assert pdf_path == output_pdf
    assert output_pdf.exists()
    assert not debug_dir.exists()
    assert sorted(path.name for path in output_pdf.parent.iterdir()) == ["sheet_music.pdf"]


def test_read_sampled_frames_reduces_high_frame_rate_video(tmp_path):
    video_path = create_moving_sheet_music_video(
        tmp_path / "many_frames.mp4",
        positions=(0, 4),
        hold_frames=12,
        transition_frames=0,
        fps=6.0,
        frame_size=(120, 90),
    )

    sampled, sampled_fps = _read_sampled_frames(video_path, source_fps=6.0, target_fps=2.0)

    assert sampled_fps == 2.0
    assert 7 <= len(sampled) <= 9
