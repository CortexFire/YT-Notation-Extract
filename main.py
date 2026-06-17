from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from sheet_video_to_pdf.cli import run_cli


if __name__ == "__main__":
    raise SystemExit(run_cli())
