from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from .errors import ConfigError
from .models import AppConfig, DuplicatePolicy, PageOrientation, PagePreset

DEFAULT_CONFIG = AppConfig()

_ALLOWED_KEYS = set(AppConfig.__dataclass_fields__.keys())


def build_config(
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> AppConfig:
    values: dict[str, Any] = {}

    errors: list[str] = []
    if config_path is not None:
        file_values = _load_json_config(Path(config_path))
        errors.extend(_unknown_key_errors(file_values))
        values.update(file_values)

    clean_overrides = {
        key: value
        for key, value in (overrides or {}).items()
        if value is not None
    }
    errors.extend(_unknown_key_errors(clean_overrides))
    values.update(clean_overrides)

    converted, conversion_errors = _convert_values(values)
    errors.extend(conversion_errors)

    if errors:
        raise ConfigError("; ".join(errors))

    return replace(DEFAULT_CONFIG, **converted)


def _load_json_config(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError as exc:
        raise ConfigError(f"Config file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Config file is not valid JSON: {path}") from exc

    if not isinstance(data, dict):
        raise ConfigError("Config JSON must contain an object at the top level")
    return data


def _unknown_key_errors(values: dict[str, Any]) -> list[str]:
    errors = []
    for key in values:
        if key == "sample_fps":
            errors.append("sample_fps is not supported; analysis cadence is automatic")
        elif key not in _ALLOWED_KEYS:
            errors.append(f"Unknown config field: {key}")
    return errors


def _convert_values(values: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    converted: dict[str, Any] = {}
    errors: list[str] = []

    for key, value in values.items():
        if key not in _ALLOWED_KEYS:
            continue
        try:
            converted[key] = _convert_value(key, value)
        except ValueError as exc:
            errors.append(str(exc))

    return converted, errors


def _convert_value(key: str, value: Any) -> Any:
    if key in {"input_video", "output_pdf", "output_dir"}:
        return Path(value)
    if key == "page_preset":
        return _enum_value(PagePreset, value, key)
    if key == "page_orientation":
        return _enum_value(PageOrientation, value, key)
    if key == "duplicate_policy":
        return _enum_value(DuplicatePolicy, value, key)
    if key == "page_margin_inches":
        margin = float(value)
        if margin < 0:
            raise ValueError("page_margin_inches must be non-negative")
        return margin
    if key == "target_systems_per_page":
        if value == "auto":
            return value
        count = int(value)
        if count < 1:
            raise ValueError("target_systems_per_page must be 'auto' or a positive integer")
        return count
    if key == "jpeg_quality":
        quality = int(value)
        if not 1 <= quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")
        return quality
    if key == "pdf_dpi":
        dpi = int(value)
        if dpi < 1:
            raise ValueError("pdf_dpi must be positive")
        return dpi
    if key in {"generate_review_assets", "clean_output"}:
        if not isinstance(value, bool):
            raise ValueError(f"{key} must be true or false")
        return value
    return value


def _enum_value(enum_type: type, value: Any, key: str) -> Any:
    try:
        return enum_type(value)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise ValueError(f"{key} must be one of: {allowed}") from exc
