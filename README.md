# YT Notation Extract

## Purpose

YT Notation Extract is a local tool for turning a local MP4 video of sheet music into a readable PDF.

The tool analyzes the video, finds stable notation views, extracts notation regions, flags duplicate or near-duplicate regions for review, stitches overlapping partial views into reconstructed page images, and generates an image-based PDF.

It is intended for videos where the sheet music is already visible in the frame, including videos that show only part of the score at a time, such as two systems or staff lines per view.

## How to use

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Place a local MP4 file somewhere on your machine, for example:

```text
input/video.mp4
```

Run the tool:

```bash
python main.py --input input/video.mp4 --output output/sheet_music.pdf
```

On Windows, if `python` is not available but the Python launcher is installed, use:

```bash
py main.py --input input/video.mp4 --output output/sheet_music.pdf
```

Optional output controls:

```bash
python main.py \
  --input input/video.mp4 \
  --output output/sheet_music.pdf \
  --output-dir output \
  --page-preset letter \
  --page-orientation portrait \
  --duplicate-policy flag
```

You can also use a JSON config file:

```bash
python main.py --config config.json
```

Example config:

```json
{
  "input_video": "input/video.mp4",
  "output_pdf": "output/sheet_music.pdf",
  "output_dir": "output",
  "page_preset": "letter",
  "page_orientation": "portrait",
  "page_margin_inches": 0.35,
  "target_systems_per_page": "auto",
  "duplicate_policy": "flag",
  "generate_review_assets": true,
  "jpeg_quality": 92,
  "pdf_dpi": 200,
  "clean_output": true
}
```

The tool automatically decides the frame analysis cadence. You do not need to configure a sample rate.

Generated outputs are written under the output folder:

```text
output/
  stable_views/
  extracted_regions/
  stitched_pages/
  manifest.json
  sheet_music.pdf
```

## Dependencies

Python dependencies are listed in `requirements.txt`:

```text
opencv-python
pillow
img2pdf
numpy
```

FFmpeg is recommended as a system dependency because it improves MP4 codec support for OpenCV.

Example FFmpeg installation commands:

```bash
# macOS
brew install ffmpeg

# Windows
winget install Gyan.FFmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg
```

For development and tests:

```bash
pip install -r requirements-dev.txt
pytest
```

## Notes on downloading MP4s

This project does not download videos and does not integrate with YouTube. It only processes local MP4 files that you already have.

If your source is a YouTube video or another online video, use a separate YouTube downloader, video extraction app, or downloader service to create the MP4 first, then provide that local file to this tool.

Make sure you have the right to download and process the video, and follow the terms of the platform hosting it.
