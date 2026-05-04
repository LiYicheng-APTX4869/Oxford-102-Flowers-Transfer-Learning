from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        if path.suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(f)
        if path.suffix.lower() == ".json":
            return json.load(f)
    raise ValueError(f"Unsupported config file: {path}")


def save_config(config: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        if path.suffix.lower() in {".yaml", ".yml"}:
            yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
        elif path.suffix.lower() == ".json":
            json.dump(config, f, ensure_ascii=False, indent=2)
        else:
            raise ValueError(f"Unsupported config file: {path}")


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def parse_cli_overrides(unknown_args: list[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    idx = 0
    while idx < len(unknown_args):
        key = unknown_args[idx]
        if not key.startswith("--"):
            raise ValueError(f"Unexpected argument: {key}")
        if idx + 1 >= len(unknown_args):
            raise ValueError(f"Missing value for override: {key}")
        value = unknown_args[idx + 1]
        idx += 2
        key = key[2:]
        current = overrides
        parts = key.split(".")
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = parse_scalar(value)
    return overrides


def parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        if "." in value or "e" in lowered:
            return float(value)
        return int(value)
    except ValueError:
        return value


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="Path to YAML or JSON config.")
    return parser
