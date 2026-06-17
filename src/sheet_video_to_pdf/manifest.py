from __future__ import annotations

import json
from pathlib import Path

from .models import RunManifest


def write_manifest(
    manifest: RunManifest,
    output_dir: str | Path,
    filename: str = "manifest.json",
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path / filename
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path
