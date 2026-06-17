from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from sheet_video_to_pdf.interactive import run_interactive


if __name__ == "__main__":
    raise SystemExit(run_interactive())
