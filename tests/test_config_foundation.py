from pathlib import Path

import pytest

from sheet_video_to_pdf.config import DEFAULT_CONFIG, build_config
from sheet_video_to_pdf.errors import ConfigError
from sheet_video_to_pdf.models import DuplicatePolicy, PageOrientation, PagePreset


def test_default_config_matches_spec_defaults():
    config = DEFAULT_CONFIG

    assert config.input_video == Path("input/video.mp4")
    assert config.output_pdf == Path("output/sheet_music.pdf")
    assert config.output_dir == Path("output")
    assert config.page_preset is PagePreset.LETTER
    assert config.page_orientation is PageOrientation.PORTRAIT
    assert config.page_margin_inches == 0.35
    assert config.target_systems_per_page == "auto"
    assert config.duplicate_policy is DuplicatePolicy.FLAG
    assert config.generate_review_assets is True
    assert config.jpeg_quality == 92
    assert config.pdf_dpi == 200
    assert config.clean_output is True


def test_config_file_values_override_defaults_and_cli_values_override_file(tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text(
        """
        {
          "input_video": "file-value.mp4",
          "output_pdf": "file-output.pdf",
          "output_dir": "file-output",
          "page_preset": "letter",
          "page_orientation": "landscape",
          "page_margin_inches": 0.5,
          "target_systems_per_page": 5,
          "duplicate_policy": "flag-and-include",
          "generate_review_assets": false,
          "jpeg_quality": 80,
          "pdf_dpi": 150,
          "clean_output": false
        }
        """,
        encoding="utf-8",
    )

    config = build_config(
        config_path=config_file,
        overrides={
            "input_video": "cli-value.mp4",
            "duplicate_policy": "flag",
            "generate_review_assets": True,
        },
    )

    assert config.input_video == Path("cli-value.mp4")
    assert config.output_pdf == Path("file-output.pdf")
    assert config.output_dir == Path("file-output")
    assert config.page_orientation is PageOrientation.LANDSCAPE
    assert config.page_margin_inches == 0.5
    assert config.target_systems_per_page == 5
    assert config.duplicate_policy is DuplicatePolicy.FLAG
    assert config.generate_review_assets is True
    assert config.jpeg_quality == 80
    assert config.pdf_dpi == 150
    assert config.clean_output is False


def test_config_rejects_sample_fps_and_invalid_duplicate_policy(tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text(
        '{"input_video": "video.mp4", "sample_fps": 2, "duplicate_policy": "remove"}',
        encoding="utf-8",
    )

    with pytest.raises(ConfigError) as exc:
        build_config(config_path=config_file)

    message = str(exc.value)
    assert "sample_fps" in message
    assert "duplicate_policy" in message
